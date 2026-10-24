

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace
import yaml

# Imports do work, even if not detected
from evo.core import trajectory
from evo.core import sync
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import plot
from evo.core.geometry import umeyama_alignment
import matplotlib.pyplot as plt

# Note; By default SLAM tracked body frame is the left camera
# A SLAM trajectory is T_leftcam_in_slamframe
def define_transforms(in_kalibr):
    T = SimpleNamespace()

    # Transform from optitrack UWB anchor markers to UWB antenna.
    # TODO: Verify by hand
    T.T_optiuwb_to_uwbtx = np.eye(4) # Probably better to express as a vector in the vUWB frame
    T.T_optiuwb_to_uwbtx[:3, 3] = [0.045, 0.04, 0] # 3cm down along x-axis.

    # 'Head' refers to the vicon tracked head pose
    T.T_imu_to_body = np.eye(4)
    T.T_body_to_imu = np.linalg.inv(T.T_imu_to_body)

    # Transform from head tracking optitrack markers to left camera
    # TODO: Verify by hand
    T.T_head_to_cam1 = np.array(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0.0175],
         [0, 0, -1, -0.08],
         [0, 0, 0, 1]]
    )

    with open(in_kalibr, 'r') as fs: calibration = yaml.safe_load(fs)
    T.T_imu_to_cam1 = np.array(calibration['cam0']['T_cam_imu'])
    T.T_cam1_to_body = T.T_imu_to_body @ np.linalg.inv(T.T_imu_to_cam1)
    T.T_head_to_body = T.T_cam1_to_body @ T.T_head_to_cam1

    # Would only change if you're using vicon2gt
    T.T_inertial_to_world = np.eye(4)

    # Transform from head tracking optitrack markers to decawave antenna
    # TODO: Verify by hand

    T.T_head_to_decawave = np.array(
        [
         [1, 0, 0, 0.01],
         [0, 1, 0, 0.0525],
         [0, 0, 1, 0.0175],
         [0, 0, 0, 1]
        ]
    )

    # From old Vicon code
    # T_decawave_to_head = np.eye(4)
    # T_decawave_to_head[:3,3] = np.array([-0.01, -0.0175, 0.0525])
    # T.T_head_to_decawave = np.linalg.inv(T_decawave_to_head)
    
    # GTSAM expects the "pose of object in body frame"
    # i.e. T_decawave_to_body, but I can do that invert in C++
    T.T_body_to_decawave = T.T_head_to_decawave @ np.linalg.inv(T.T_head_to_body)

    return T


def slam_quat_to_HTM(nparr): # Doesnt timestamp
    translation = nparr[1:4]
    quat = nparr[4:8]
    
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()

    # Assemble homogeneous transformation matrix (4x4)
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = translation

    return H

def HTM_to_TUM(T): # 2D pose matrix to TUM format
    # Extract translation and rotation
    t = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

    # Non timestamped
    return [ t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]]

def slam_HTM_to_TUM(nparr): # Same as HTM to TUM but it handles timestamped
    if len(nparr) != 17:
        print(nparr)
        raise ValueError("Expected 17 elements: [timestamp, 16 HTM elements]")

    timestamp = nparr[0]
    T_flat = nparr[1:]
    T = np.array(T_flat).reshape((4, 4))

    # Extract translation and rotation
    t = T[:3, 3]
    R_mat = T[:3, :3]
    quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

    return [timestamp, t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]]

# Expects data to be input as an HTM
# Pose can be passed in any frame, but be mindful of the SLERP left hand coordinate system problem
# Pass a target_timestamp, first < target < second
def interpolate_pose(first_pose, first_timestamp, second_pose, second_timestamp, target_timestamp, n_points):

    # Now interpolate between these two poses
    interp_interval = [first_timestamp, second_timestamp]
    interp_timestamps = np.linspace(first_timestamp, second_timestamp, n_points)

    # Use Slerp to interpolate on SO(3) rotations
    interp_rots = R.from_matrix([first_pose[:3, :3], second_pose[:3, :3]])
    slurpy = Slerp(interp_interval, interp_rots)
    interpolated_rotations = slurpy(interp_timestamps)

    # Use linspace to interpolate on R3 positions
    interpolated_positions = np.linspace(first_pose[:3, 3], second_pose[:3, 3], n_points)

    # Fetch the closest interpolation timestamp to the uwb measurement, and map that interpolated pose to the measurement
    idx_match = np.argmin(np.abs(interp_timestamps - target_timestamp))

    interp_pose = np.eye(4)
    interp_pose[:3,:3] = interpolated_rotations[idx_match].as_matrix()
    interp_pose[:3, 3] = interpolated_positions[idx_match]

    return interp_pose

def dump_stats(traj_ref_sync, traj_est_sync):

    # Translation APE
    ape_metric_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric_trans.process_data((traj_ref_sync, traj_est_sync))
    ape_stats = ape_metric_trans.get_all_statistics()
    print(f"    Translation APE,\n\t{ape_stats["mean"]=},\n\t{ape_stats["rmse"]=}")

    # Rotation APE
    ape_metric_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    ape_metric_rot.process_data((traj_ref_sync, traj_est_sync))
    ape_stats = ape_metric_rot.get_all_statistics()
    # print(f" Rotational APE {json.dumps(ape_stats, indent=1)}")
    print(f"    Rotation APE,\n\t{ape_stats["mean"]=},\n\t{ape_stats["rmse"]=}")

def yaw_umeyama_align(traj_ref, traj_est, correct_scale=False):
    """
    Yaw-only Umeyama alignment for evo trajectories.

    Returns a NEW aligned PoseTrajectory3D.
    """

    P = traj_est.positions_xyz.copy()   # estimated
    Q = traj_ref.positions_xyz.copy()   # reference

    # centroids
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)

    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    # XY covariance only
    H = P_centered[:, :2].T @ Q_centered[:, :2]

    U, S, Vt = np.linalg.svd(H)

    R2 = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T

    # optional scale
    scale = 1.0
    if correct_scale:
        var_P = np.sum(P_centered[:, :2] ** 2)
        scale = np.sum(S) / var_P

    # build full 3D yaw rotation
    R_align = np.eye(3)
    R_align[:2, :2] = R2

    # translation
    t_align = mu_Q - scale * (R_align @ mu_P)

    # apply to positions
    aligned_positions = (
        scale * (R_align @ P.T)
    ).T + t_align

    # orientations
    est_rots = R.from_quat(
        traj_est.orientations_quat_wxyz[:, [1,2,3,0]]
    )

    yaw_rot = R.from_matrix(R_align)

    aligned_rots = yaw_rot * est_rots

    quat_xyzw = aligned_rots.as_quat()

    # convert xyzw -> wxyz
    aligned_quats_wxyz = np.column_stack([
        quat_xyzw[:, 3],
        quat_xyzw[:, 0],
        quat_xyzw[:, 1],
        quat_xyzw[:, 2]
    ])

    return PoseTrajectory3D(
        positions_xyz=aligned_positions,
        orientations_quat_wxyz=aligned_quats_wxyz,
        timestamps=traj_est.timestamps.copy()
    )

# These get passed in as lists of 1 x 17 numpy arrays
def umeyama_alignment1(body_opti_HTMs, body_slam_HTMs):

    # Per examples here: https://github.com/MichaelGrupp/evo/blob/master/examples/alignment_demo.py

    body_opti_HTMs = np.array(body_opti_HTMs)
    body_slam_HTMs = np.array(body_slam_HTMs)

    opti_ts = body_opti_HTMs[:, 0]
    slam_ts = body_slam_HTMs[:, 0]

    opti_poses = np.linalg.inv(body_opti_HTMs[:, 1:17].reshape((-1, 4,4)))
    slam_poses = np.linalg.inv(body_slam_HTMs[:, 1:17].reshape((-1, 4,4)))

    # Convert positions to evo format
    opti_positions = opti_poses[:, :3, 3]
    slam_positions = slam_poses[:, :3, 3]

    # Convert rotations to evo format
    opti_rot_mats = opti_poses[:, :3, :3]
    slam_rot_mats = slam_poses[:, :3, :3]
    opti_quats_xyzw = R.from_matrix(opti_rot_mats).as_quat()
    slam_quats_xyzw = R.from_matrix(slam_rot_mats).as_quat()
    def xyzw_to_wxyz(q):
        return np.column_stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]])
    opti_quats = xyzw_to_wxyz(opti_quats_xyzw)
    slam_quats = xyzw_to_wxyz(slam_quats_xyzw)

    opti_traj = PoseTrajectory3D(
        positions_xyz=opti_positions,
        orientations_quat_wxyz=opti_quats,
        timestamps=opti_ts
    )

    slam_traj = PoseTrajectory3D(
        positions_xyz=slam_positions,
        orientations_quat_wxyz=slam_quats,
        timestamps=slam_ts
    )


    # Time synchronization
    traj_ref, traj_est = sync.associate_trajectories(opti_traj, slam_traj, max_diff = 0.01)

    # Align (SE3)
    # traj_est.align(traj_ref, correct_scale=True)
    traj_est = yaw_umeyama_align(traj_ref, traj_est, correct_scale=True)

    print(f"Umeyama Alignment Results:")
    dump_stats(traj_ref, traj_est)

    # traj_est is now SLAM trajectory aligned to optitrack world frame

    traj_est_out = []

    rots = R.from_quat(traj_est.orientations_quat_wxyz[:, [1,2,3,0]])  # wxyz → xyzw

    for t, p, r in zip(
        traj_est.timestamps,
        traj_est.positions_xyz,
        rots
    ):
        # inverse rotation
        r_inv = r.inv()

        # inverse translation: -R^T t
        p_inv = -r_inv.apply(p)

        # quaternion back to xyzw
        qx, qy, qz, qw = r_inv.as_quat()

        # append
        traj_est_out.append([
            t,
            p_inv[0], p_inv[1], p_inv[2],
            qx, qy, qz, qw
        ])

    # SLAM timestamps that survived association with optitrack
    return traj_est_out, traj_est.timestamps.copy()


import copy
# These get passed in as lists of 1 x 17 numpy arrays
def bonus_umeyama_alignment(
    body_opti_HTMs,
    body_slam_HTMs,
    body_live_slam_HTMs,
    fail_end_ts
):

    # ------------------------------------------------------------
    # Convert input arrays
    # ------------------------------------------------------------

    body_opti_HTMs = np.array(body_opti_HTMs)
    body_slam_HTMs = np.array(body_slam_HTMs)
    body_live_slam_HTMs = np.array(body_live_slam_HTMs)

    opti_ts = body_opti_HTMs[:, 0]
    slam_ts = body_slam_HTMs[:, 0]
    live_slam_ts = body_live_slam_HTMs[:, 0]

    opti_poses = np.linalg.inv(
        body_opti_HTMs[:, 1:17].reshape((-1, 4, 4))
    )

    slam_poses = np.linalg.inv(
        body_slam_HTMs[:, 1:17].reshape((-1, 4, 4))
    )

    live_slam_poses = np.linalg.inv(
        body_live_slam_HTMs[:, 1:17].reshape((-1, 4, 4))
    )

    # ------------------------------------------------------------
    # Convert to EVO trajectories
    # ------------------------------------------------------------

    def poses_to_traj(poses, timestamps):

        positions = poses[:, :3, 3]

        rot_mats = poses[:, :3, :3]
        quats_xyzw = R.from_matrix(rot_mats).as_quat()

        quats_wxyz = np.column_stack([
            quats_xyzw[:, 3],
            quats_xyzw[:, 0],
            quats_xyzw[:, 1],
            quats_xyzw[:, 2]
        ])

        return PoseTrajectory3D(
            positions_xyz=positions,
            orientations_quat_wxyz=quats_wxyz,
            timestamps=timestamps
        )

    opti_traj = poses_to_traj(opti_poses, opti_ts)
    slam_traj = poses_to_traj(slam_poses, slam_ts)
    live_slam_traj = poses_to_traj(live_slam_poses, live_slam_ts)

    #TODO: Undo later
    # slam_traj = live_slam_traj

    # ------------------------------------------------------------
    # Crop trajectories AFTER fail_end_ts for alignment computation
    # ------------------------------------------------------------

    print(fail_end_ts)
    opti_mask = opti_traj.timestamps >= fail_end_ts
    slam_mask = slam_traj.timestamps >= fail_end_ts

    opti_traj_crop = PoseTrajectory3D(
        positions_xyz=opti_traj.positions_xyz[opti_mask],
        orientations_quat_wxyz=opti_traj.orientations_quat_wxyz[opti_mask],
        timestamps=opti_traj.timestamps[opti_mask]
    )

    slam_traj_crop = PoseTrajectory3D(
        positions_xyz=slam_traj.positions_xyz[slam_mask],
        orientations_quat_wxyz=slam_traj.orientations_quat_wxyz[slam_mask],
        timestamps=slam_traj.timestamps[slam_mask]
    )

    # ------------------------------------------------------------
    # Synchronize cropped trajectories
    # ------------------------------------------------------------

    traj_ref, traj_est = sync.associate_trajectories(
        opti_traj_crop,
        slam_traj_crop,
        max_diff=0.01
    )
    # ------------------------------------------------------------
    # Compute alignment ONCE
    # ------------------------------------------------------------

    aligned_sync = yaw_umeyama_align(
        traj_ref,
        copy.deepcopy(traj_est),
        correct_scale=True
    )

    print("Umeyama Alignment Results:")
    dump_stats(traj_ref, aligned_sync)

    # ------------------------------------------------------------
    # Recover yaw-only alignment transform
    # ------------------------------------------------------------

    p_before = traj_est.positions_xyz
    p_after = aligned_sync.positions_xyz

    # centroids
    mu_before = p_before.mean(axis=0)
    mu_after = p_after.mean(axis=0)

    # centered
    P = p_before - mu_before
    Q = p_after - mu_after

    # yaw-only covariance
    H = P[:, :2].T @ Q[:, :2]

    U, Svals, Vt = np.linalg.svd(H)

    R2 = Vt.T @ U.T

    if np.linalg.det(R2) < 0:
        Vt[-1, :] *= -1
        R2 = Vt.T @ U.T

    # scale
    var_P = np.sum(P[:, :2] ** 2)
    s = np.sum(Svals) / var_P

    # full 3D yaw rotation
    R_align = np.eye(3)
    R_align[:2, :2] = R2

    # translation
    t_align = mu_after - s * (R_align @ mu_before)

    # ------------------------------------------------------------
    # Apply alignment to arbitrary trajectory
    # ------------------------------------------------------------

    def apply_alignment(traj):

        traj = copy.deepcopy(traj)

        # positions
        aligned_positions = (
            s * (R_align @ traj.positions_xyz.T)
        ).T + t_align

        # rotations
        rot_align = R.from_matrix(R_align)

        rots = R.from_quat(
            traj.orientations_quat_wxyz[:, [1,2,3,0]]
        )

        aligned_rots = rot_align * rots

        quats_xyzw = aligned_rots.as_quat()

        quats_wxyz = np.column_stack([
            quats_xyzw[:, 3],
            quats_xyzw[:, 0],
            quats_xyzw[:, 1],
            quats_xyzw[:, 2]
        ])

        return PoseTrajectory3D(
            positions_xyz=aligned_positions,
            orientations_quat_wxyz=quats_wxyz,
            timestamps=traj.timestamps
        )

    aligned_slam = apply_alignment(slam_traj)
    aligned_live_slam = apply_alignment(live_slam_traj)

    ### debug plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(
    #     opti_traj.positions_xyz[:, 0],
    #     opti_traj.positions_xyz[:, 1],
    #     opti_traj.positions_xyz[:, 2],
    #     label="Opti"
    # )

    # ax.plot(
    #     aligned_slam.positions_xyz[:, 0],
    #     aligned_slam.positions_xyz[:, 1],
    #     aligned_slam.positions_xyz[:, 2],
    #     label="Aligned SLAM"
    # )

    # ax.plot(
    #     aligned_live_slam.positions_xyz[:, 0],
    #     aligned_live_slam.positions_xyz[:, 1],
    #     aligned_live_slam.positions_xyz[:, 2],
    #     label="Aligned Live SLAM"
    # )

    # ax.legend()
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # plt.show()
     ##

    # ------------------------------------------------------------
    # Convert back to TUM format
    # ------------------------------------------------------------

    def traj_to_tum(traj):

        out = []

        rots = R.from_quat(
            traj.orientations_quat_wxyz[:, [1,2,3,0]]
        )

        for t, p, r in zip(
            traj.timestamps,
            traj.positions_xyz,
            rots
        ):

            r_inv = r.inv()

            p_inv = -r_inv.apply(p)

            qx, qy, qz, qw = r_inv.as_quat()

            out.append([
                t,
                p_inv[0],
                p_inv[1],
                p_inv[2],
                qx,
                qy,
                qz,
                qw
            ])

        return out



    # Live SLAM timestamps present in the aligned live trajectory
    return (
        traj_to_tum(aligned_slam),
        traj_to_tum(aligned_live_slam),
        aligned_live_slam.timestamps.copy()
    )