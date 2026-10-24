import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

# Optitrack world frame is Y-up, so global yaw is a rotation about Y.
# ORB-SLAM's frame (post / localframe) has gravity roughly along -Y too, so the same axis is used there.
UP_AXIS = np.array([0, 0, 1])


# def perturb_traj(traj_json, t_vec, R_yaw):
#     """
#     Translate every pose by t_vec, then rotate the whole trajectory by R_yaw
#     about the (translated) position of its first pose.
#     Equivalent to left-multiplying every T_body_world by
#         G = Trans(c) @ Rot(R_yaw) @ Trans(-c) @ Trans(t_vec),   c = p_first + t_vec
#     """
#     traj_json = copy.deepcopy(traj_json) # Lists may share dicts (e.g. deepcopy'd siblings), don't mutate in place
#     if len(traj_json) == 0: return traj_json

#     first = min(traj_json, key=lambda j: j["t"])
#     traj = [np.linalg.inv(np.asarray(j["T_body_world"])) for j in traj_json] # First invert all poses
#     # This trajectory is T_body_to_world poses

#     for pose in traj: pose[:3, 3] += t_vec # Translate all poses by vector in world frame.

#     # Represent all poses in trajectory relative to 0
#     first_pose = traj[0]
#     traj_deltas = [ pose @ np.linalg.inv(first_pose) for pose in traj]

#     rotated_first_pose = copy.deepcopy(first_pose)
#     rotated_first_pose[:3,:3] = R_yaw @ first_pose[:3,:3]
#     print(rotated_first_pose)

#     rotated_traj = [ delta @ rotated_first_pose for delta in traj_deltas]
#     # Final trajectory that has been first translated, then rotated about the first pose
#     # Still in T_body_to_world

#     for j, pose in zip(traj_json, rotated_traj):
#         j["T_body_world"] = np.linalg.inv(pose)

#     return traj_json

def perturb_traj(traj_json, t_vec, R_yaw):
    """
    Translate every pose by t_vec, then rotate the whole trajectory by R_yaw
    about the (translated) position of its first pose.
    Equivalent to left-multiplying every T_body_world by
        G = Trans(c) @ Rot(R_yaw) @ Trans(-c) @ Trans(t_vec),   c = p_first + t_vec
    """
    traj_json = copy.deepcopy(traj_json) # Lists may share dicts (e.g. deepcopy'd siblings), don't mutate in place
    if len(traj_json) == 0: return traj_json

    first = min(traj_json, key=lambda j: j["t"])
    traj = [np.linalg.inv(np.asarray(j["T_body_world"])) for j in traj_json] # First invert all poses
    # This trajectory is T_body_to_world poses

    for pose in traj: pose[:3, 3] += t_vec # Translate all poses by vector in world frame.

    # Represent all poses in trajectory relative to 0
    first_pose = traj[0]
    traj_deltas = [  np.linalg.inv(first_pose) @ pose for pose in traj]

    rotated_first_pose = copy.deepcopy(first_pose)
    rotated_first_pose[:3,:3] = np.linalg.inv(R_yaw) @ first_pose[:3,:3]
    print(rotated_first_pose)

    rotated_traj = [ rotated_first_pose @ delta for delta in traj_deltas]
    # Final trajectory that has been first translated, then rotated about the first pose
    # Still in T_body_to_world

    for j, pose in zip(traj_json, rotated_traj):
        j["T_body_world"] = np.linalg.inv(pose)

    # traj = [np.linalg.inv(np.asarray(j["T_body_world"])).copy() for j in traj_json]  # already body->world, no inverse
    # for pose in traj: pose[:3, 3] += t_vec                             # translate in world frame

    # first_pose = np.linalg.inv(min(zip(traj_json, traj), key=lambda x: x[0]["t"])[1]) # translated first pose
    # rotated_first_pose = first_pose.copy()
    # rotated_first_pose[:3, :3] = R_yaw @ first_pose[:3, :3]

    # G = rotated_first_pose @ np.linalg.inv(first_pose)                 # = Trans(c) Rot Trans(-c)
    # for j, pose in zip(traj_json, traj):
    #     j["T_body_world"] = np.linalg.inv(G @ pose)


    return traj_json

def perturb(perturb_t, perturb_r, post_slam_json, aligned_post_slam_json, aligned_live_slam_json, localframe_live_slam_json, opti_json):
    """
    Perturb each SLAM trajectory by exactly perturb_t meters in a random 3D direction,
    then by exactly perturb_r degrees of yaw (random sign) about its own first pose.
    One random draw is shared by all four trajectories.
    """
    perturb_t = perturb_t or 0.0
    perturb_r = perturb_r or 0.0

    direction = np.random.normal(size=3)
    direction /= np.linalg.norm(direction)
    t_vec = perturb_t * direction

    yaw_deg = np.random.choice([-1, 1]) * perturb_r
    R_yaw = R.from_rotvec(np.deg2rad(yaw_deg) * UP_AXIS).as_matrix()

    print(f" Perturbing SLAM trajectories by t={t_vec} ({perturb_t}m), yaw={yaw_deg}deg")

    return (
        perturb_traj(post_slam_json, t_vec, R_yaw),
        perturb_traj(aligned_post_slam_json, t_vec, R_yaw),
        perturb_traj(aligned_live_slam_json, t_vec, R_yaw),
        perturb_traj(localframe_live_slam_json, t_vec, R_yaw),
        perturb_traj(opti_json, t_vec, R_yaw),
    )
