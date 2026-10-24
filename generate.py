"""
MultiXR-Synthetic: synthesize multi-user SLAM / IMU / UWB data from LocoVR head trajectories.

Pipeline
  1. Build N_mobile 70 s head trajectories by stitching LocoVR segments tail-to-head inside the room,
     with no user leaving the room and no two users' 0.25 m cylinders overlapping.
  2. Place N_static anchors on the floor (z = 0).
  3. Synthesize 200 Hz accelerometer / gyroscope streams (gnss-ins-sim error model + bias random walk).
  4. Synthesize UWB ranges between all nodes with a Pure ALOHA channel model.
  5. Write everything to generated/mobile_{N_mobile}_static_{N_static}/.

World frame: right-handed, Z-up, room spans [0, W] x [0, H], floor at z = 0, gravity = [0, 0, -9.81].
Head (body) frame: FLU (x forward, y left, z up).
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, RotationSpline

GNSS_INS_SIM_PATH = "/home/antond2/Desktop/Research/gnss-ins-sim"
sys.path.insert(0, GNSS_INS_SIM_PATH)
from gnss_ins_sim.sim import imu_model  # noqa: E402
from gnss_ins_sim.pathgen import pathgen  # noqa: E402

LOCOVR_PATH = "/home/antond2/Desktop/Research/Datasets/LocoVR/LocoVR-20261024T163646Z-1-001/LocoVR"

### Rigid offsets
T_HEAD_TO_DECAWAVE = np.eye(4)
T_HEAD_TO_IMU = np.eye(4)

### LocoVR
FS_POSE = 90.0                  # LocoVR sample rate, Hz
FEET = ("left foot", "right foot")
# LocoVR positions are already Z-up (x, y, z) = (u_z, -u_x, u_y) of Unity's left-handed Y-up frame
# (LocoVR's own transform_axis_pos), but the quaternions (x, y, z, w) are still in the Unity frame.
M_UNITY_TO_WORLD = np.array([[0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]])
# After conjugating with M, the head's columns are (backward, right, up); rotate 180 deg about z for FLU.
R_TO_FLU = Rotation.from_euler("z", np.pi)
SMOOTH_CUTOFF_HZ = 8.0          # low-pass on tracker data before it is differentiated for the IMU
MIN_SEGMENT_S = 1.0

### Placement
USER_RADIUS = 0.25              # m, bounding cylinder around each user
BLEND_S = 0.5                   # s, velocity / tilt / height blend at each stitch
MAX_STITCH_YAW_DEG = 20.0       # max angle between the tail's forward vector and the next segment's first one
MAX_TURN_DEG = 30.0             # max extra yaw applied to a stitched segment ...
MAX_TURN_POSES = 5              # ... over this many consecutive poses
DELTA_STEP_DEG = 5.0            # granularity of candidate extra yaw rotations
SEGMENT_TRIES = 200
USER_TRIES = 50

### IMU (ORB-SLAM3 continuous-time noise densities)
IMU_FS = 200.0
NOISE_GYRO = 0.002049600985797649       # rad/s^0.5
NOISE_ACC = 0.02064189891192468         # m/s^1.5
GYRO_WALK = 0.00003199855545594742      # rad/s^1.5
ACC_WALK = 0.0022919238444020808        # m/s^2.5
GRAVITY = np.array([0.0, 0.0, -9.81])

### UWB
UWB_STD = 0.2                   # m

### Output
VIVE_FS = 30.0                  # written pose rate, Hz (subsampled from FS_POSE)


def per_axis(scalar):
    # ||v|| = |x| sqrt(3) with equal components
    return np.full(3, scalar / math.sqrt(3.0))


def yaw_of(rot):
    m = rot.as_matrix()
    return np.arctan2(m[..., 1, 0], m[..., 0, 0])


def rotate_xy(angles, xy):
    c, s = np.cos(angles), np.sin(angles)
    return np.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], axis=1)


def to_htm(rot, pos):
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = rot.as_matrix()
    T[:, :3, 3] = pos
    return T


### 1. Trajectories

def load_segments(dataset_dir):
    b, a = butter(4, SMOOTH_CUTOFF_HZ / (FS_POSE / 2.0))
    segments = []
    for scene in sorted(os.listdir(dataset_dir)):
        with open(os.path.join(dataset_dir, scene), "rb") as f:
            episodes = pickle.load(f)
        for ep_idx, ep in enumerate(episodes):
            for person in ("p1", "p2"):
                head = ep[person]["head"]
                if len(head["pos"]) < MIN_SEGMENT_S * FS_POSE:
                    continue
                floor = min(ep[person][foot]["pos"][:, 2].min() for foot in FEET)
                pos = head["pos"] - np.array([0.0, 0.0, floor])

                R_unity = Rotation.from_quat(head["pose"]).as_matrix()
                rot = Rotation.from_matrix(M_UNITY_TO_WORLD @ R_unity @ M_UNITY_TO_WORLD.T) * R_TO_FLU

                q = rot.as_quat()
                flip = np.cumprod(np.r_[1.0, np.sign(np.sum(q[1:] * q[:-1], axis=1))])
                q = filtfilt(b, a, q * flip[:, None], axis=0)
                segments.append({
                    "pos": filtfilt(b, a, pos, axis=0),
                    "rot": Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)),
                    "src": {"scene": scene, "episode": ep_idx, "person": person},
                })
    return segments


def is_valid(xy, start, room, others):
    r = USER_RADIUS
    if np.any(xy < r) or np.any(xy[:, 0] > room[0] - r) or np.any(xy[:, 1] > room[1] - r):
        return False
    for other in others:
        d = np.linalg.norm(other[start:start + len(xy), :2] - xy, axis=1)
        if np.any(d < 2.0 * r):
            return False
    return True


def place_first(rng, seg, room, others):
    theta = rng.uniform(-np.pi, np.pi)
    xy = rotate_xy(np.full(len(seg["pos"]), theta), seg["pos"][:, :2])
    lo = USER_RADIUS - xy.min(axis=0)
    hi = np.asarray(room) - USER_RADIUS - xy.max(axis=0)
    if np.any(lo > hi):
        return None
    xy = xy + rng.uniform(lo, hi)
    if not is_valid(xy, 0, room, others):
        return None
    return np.c_[xy, seg["pos"][:, 2]], Rotation.from_euler("z", theta) * seg["rot"]


def continue_from(seg, end_pos, end_rot, end_step, delta):
    """
    Attach seg to the tail (end_pos, end_rot) so its heading continues the previous one, then ramp in an
    extra yaw `delta` (|delta| <= MAX_STITCH_YAW_DEG) over at least BLEND_S and no faster than MAX_TURN_DEG
    per MAX_TURN_POSES. Velocity, tilt and height are blended from the previous tail over BLEND_S.
    Returns the new poses, excluding the shared junction pose.
    """
    pos, rot = seg["pos"], seg["rot"]
    k = np.arange(len(pos))

    max_step = np.deg2rad(MAX_TURN_DEG) / MAX_TURN_POSES
    # peak slope of a cosine ramp is pi/2 * delta/n
    n_turn = max(math.ceil(0.5 * np.pi * abs(delta) / max_step), int(BLEND_S * FS_POSE))
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.minimum(k / n_turn, 1.0)))
    phi = yaw_of(end_rot) - yaw_of(rot[0]) + delta * ramp

    w = 0.5 * (1.0 + np.cos(np.pi * np.minimum(k / (BLEND_S * FS_POSE), 1.0)))  # 1 -> 0

    steps = rotate_xy(phi[1:], np.diff(pos[:, :2], axis=0))
    steps = w[1:, None] * end_step + (1.0 - w[1:, None]) * steps
    xy = end_pos[:2] + np.cumsum(steps, axis=0)
    z = pos[1:, 2] + (end_pos[2] - pos[0, 2]) * w[1:]

    rot = Rotation.from_euler("z", phi) * rot
    tilt = (end_rot * rot[0].inv()).as_rotvec()
    rot = Rotation.from_rotvec(w[:, None] * tilt) * rot
    return np.c_[xy, z], rot[1:]


def build_user(rng, segments, room, others, n_poses):
    # Prefer the smallest extra yaw that keeps the user valid
    deltas = [0.0]
    for mag in np.arange(DELTA_STEP_DEG, MAX_STITCH_YAW_DEG + 1e-9, DELTA_STEP_DEG):
        sign = rng.choice([-1.0, 1.0])
        deltas += [sign * mag, -sign * mag]
    deltas = np.deg2rad(deltas)

    for _ in range(SEGMENT_TRIES):
        seg = segments[rng.integers(len(segments))]
        placed = place_first(rng, seg, room, others)
        if placed is not None:
            break
    else:
        return None
    positions, rotations = [placed[0]], [placed[1]]
    sources = [dict(seg["src"], t_start=0.0, delta_deg=None)]
    n = len(placed[0])

    while n < n_poses:
        end_pos, end_rot = positions[-1][-1], rotations[-1][-1]
        end_step = (positions[-1][-1] - positions[-1][-2])[:2]
        for _ in range(SEGMENT_TRIES):
            seg = segments[rng.integers(len(segments))]
            for delta in deltas:
                pos, rot = continue_from(seg, end_pos, end_rot, end_step, delta)
                pos, rot = pos[:n_poses - n], rot[:n_poses - n]
                if is_valid(pos[:, :2], n, room, others):
                    break
            else:
                continue
            break
        else:
            return None
        positions.append(pos)
        rotations.append(rot)
        sources.append(dict(seg["src"], t_start=n / FS_POSE, delta_deg=float(np.rad2deg(delta))))
        n += len(pos)

    return np.concatenate(positions), Rotation.concatenate(rotations), sources


def generate_trajectories(rng, segments, n_mobile, room, duration):
    n_poses = int(round(duration * FS_POSE)) + 1
    users = []
    for uid in range(n_mobile):
        others = [u["pos"] for u in users]
        for _ in range(USER_TRIES):
            built = build_user(rng, segments, room, others, n_poses)
            if built is not None:
                break
        else:
            raise RuntimeError(f"Could not place user {uid} in room {room}; try a larger room or fewer users.")
        pos, rot, sources = built
        users.append({"id": uid, "t": np.arange(n_poses) / FS_POSE, "pos": pos, "rot": rot, "sources": sources})
        print(f"user {uid}: {len(sources)} segments")
    return users


### 2. Anchors

def generate_anchors(rng, n_mobile, n_static, room):
    xy = rng.uniform([0.0, 0.0], room, size=(n_static, 2))
    return [{"ID": n_mobile + i, "position": [float(x), float(y), 0.0]} for i, (x, y) in enumerate(xy)]


### 3. IMU

def make_imu():
    # Units expected by imu_model.IMU: deg/hr, deg/rt-hr, m/s^2, m/s/rt-hr.
    # gnss-ins-sim treats b_corr = inf as white bias noise (not a random walk), so bias instability is
    # zeroed here and the random walk is added in synthesize_imu.
    return imu_model.IMU(accuracy={
        "gyro_b": np.zeros(3),
        "gyro_arw": per_axis(NOISE_GYRO) * (180.0 / np.pi) * 60.0,
        "gyro_b_stability": np.zeros(3),
        "gyro_b_corr": np.full(3, np.inf),
        "accel_b": np.zeros(3),
        "accel_vrw": per_axis(NOISE_ACC) * 60.0,
        "accel_b_stability": np.zeros(3),
        "accel_b_corr": np.full(3, np.inf),
    }, axis=6, gps=False)


def random_walk(rng, sigma, n, fs):
    steps = rng.standard_normal((n, 3)) * sigma * math.sqrt(1.0 / fs)
    steps[0] = 0.0
    return np.cumsum(steps, axis=0)


def synthesize_imu(rng, user, imu):
    T_imu_to_world = to_htm(user["rot"], user["pos"]) @ np.linalg.inv(T_HEAD_TO_IMU)
    t = user["t"]
    pos_spline = CubicSpline(t, T_imu_to_world[:, :3, 3])
    rot_spline = RotationSpline(t, Rotation.from_matrix(T_imu_to_world[:, :3, :3]))

    t_imu = np.arange(0.0, t[-1] + 1e-9, 1.0 / IMU_FS)
    R_wb = rot_spline(t_imu)
    f_b = R_wb.inv().apply(pos_spline(t_imu, 2) - GRAVITY)

    h = 1e-4
    t_lo, t_hi = np.clip(t_imu - h, t[0], t[-1]), np.clip(t_imu + h, t[0], t[-1])
    w_b = (rot_spline(t_lo).inv() * rot_spline(t_hi)).as_rotvec() / (t_hi - t_lo)[:, None]

    n = len(t_imu)
    accel = pathgen.acc_gen(IMU_FS, f_b, imu.accel_err) + random_walk(rng, per_axis(ACC_WALK), n, IMU_FS)
    gyro = pathgen.gyro_gen(IMU_FS, w_b, imu.gyro_err) + random_walk(rng, per_axis(GYRO_WALK), n, IMU_FS)

    return [{
        "t": float(ti), "type": "imu",
        "ax": float(a[0]), "ay": float(a[1]), "az": float(a[2]),
        "gx": float(g[0]), "gy": float(g[1]), "gz": float(g[2]),
    } for ti, a, g in zip(t_imu, accel, gyro)]


### 4. UWB

def synthesize_uwb(rng, users, anchors, duration, rate, airtime):
    """
    Pure ALOHA: every node starts ranging exchanges (airtime seconds) to a uniformly random peer as a
    Poisson process. An exchange succeeds iff no other exchange starts within +-airtime of it. Only the
    initiator logs the range, so anchor-initiated exchanges occupy the channel but produce no output.
    """
    n_nodes = len(users) + len(anchors)
    if n_nodes < 2:
        return {u["id"]: [] for u in users}
    starts, srcs = [], []
    for node in range(n_nodes):
        k = rng.poisson(rate * duration)
        starts.append(rng.uniform(0.0, duration, k))
        srcs.append(np.full(k, node))
    starts, srcs = np.concatenate(starts), np.concatenate(srcs)
    order = np.argsort(starts)
    starts, srcs = starts[order], srcs[order]
    dsts = (srcs + rng.integers(1, n_nodes, len(srcs))) % n_nodes

    gaps = np.diff(starts)
    ok = np.r_[True, gaps >= airtime] & np.r_[gaps >= airtime, True]
    print(f"UWB: {len(starts)} exchanges, {ok.mean():.1%} collision-free")

    def decawave_positions(node, t):
        if node >= len(users):
            return np.tile(anchors[node - len(users)]["position"], (len(t), 1))
        u = users[node]
        rot = RotationSpline(u["t"], u["rot"])(t)
        T_head_to_world = to_htm(rot, CubicSpline(u["t"], u["pos"])(t))
        return (T_head_to_world @ np.linalg.inv(T_HEAD_TO_DECAWAVE))[:, :3, 3]

    out = {u["id"]: [] for u in users}
    for u in users:
        mine = ok & (srcs == u["id"])
        t, dst = starts[mine], dsts[mine]
        src_pos = decawave_positions(u["id"], t)
        dst_pos = np.empty_like(src_pos)
        for node in np.unique(dst):
            sel = dst == node
            dst_pos[sel] = decawave_positions(node, t[sel])
        gt_range = np.linalg.norm(src_pos - dst_pos, axis=1)
        ranges = rng.normal(gt_range, UWB_STD)
        out[u["id"]] = [{"t": float(ti), "type": "uwb", "id": int(d), "range": float(r)}
                        for ti, d, r in zip(t, dst, ranges)]
    return out


### 5. Output

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def vive_json(user):
    step = int(round(FS_POSE / VIVE_FS))
    T_world_to_head = np.linalg.inv(to_htm(user["rot"][::step], user["pos"][::step]))
    return [{"t": float(t), "type": "vive_pose", "T_body_world": T.tolist()}
            for t, T in zip(user["t"][::step], T_world_to_head)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("N_mobile", type=int, help="number of mobile user trajectories")
    parser.add_argument("N_static", type=int, help="number of static anchors")
    parser.add_argument("--room_dim", type=float, nargs=2, default=[20.0, 20.0], metavar=("W", "H"))
    parser.add_argument("--duration", type=float, default=70.0, help="length of each user trajectory, s")
    parser.add_argument("--uwb_rate", type=float, default=6, help="ranging attempts per node per second")
    parser.add_argument("--uwb_airtime", type=float, default=0.002, help="duration of one ranging exchange, s")
    parser.add_argument("--dataset", default=LOCOVR_PATH)
    parser.add_argument("--out_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.N_mobile < 1 or args.N_static < 0:
        parser.error("need N_mobile >= 1 and N_static >= 0")

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # gnss-ins-sim draws from the global numpy RNG
    room = tuple(args.room_dim)

    segments = load_segments(args.dataset)
    print(f"Loaded {len(segments)} LocoVR head segments")

    users = generate_trajectories(rng, segments, args.N_mobile, room, args.duration)
    anchors = generate_anchors(rng, args.N_mobile, args.N_static, room)
    imu = make_imu()
    uwb = synthesize_uwb(rng, users, anchors, users[0]["t"][-1], args.uwb_rate, args.uwb_airtime)

    out_dir = os.path.join(args.out_root, f"mobile_{args.N_mobile}_static_{args.N_static}")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "anchors.json"), anchors)
    write_json(os.path.join(out_dir, "config.json"), {
        "args": vars(args),
        "imu": {"frequency": IMU_FS, "noise_gyro": NOISE_GYRO, "noise_acc": NOISE_ACC,
                "gyro_walk": GYRO_WALK, "acc_walk": ACC_WALK},
        "uwb_std": UWB_STD,
        "user_radius": USER_RADIUS,
        "sources": {u["id"]: u["sources"] for u in users},
    })
    for u in users:
        user_dir = os.path.join(out_dir, str(u["id"]))
        os.makedirs(user_dir, exist_ok=True)
        write_json(os.path.join(user_dir, "vive_synth.json"), vive_json(u))
        write_json(os.path.join(user_dir, "imu_synth.json"), synthesize_imu(rng, u, imu))
        write_json(os.path.join(user_dir, "uwb_synth.json"), uwb[u["id"]])
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
