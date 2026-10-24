import cv2
import numpy as np
from utils.math_utils import *
import copy
import os


def aggregate_uwb(topic_to_processing, uwb_csv, uwb_range_distribution):
    for j in topic_to_processing['/uwb_ranges'][1]:
        csv_row = []
        for k, v in j.items(): csv_row.append(v) # This should iterate in the order of how keys are originally defined in the json
        uwb_csv.append(csv_row)
        uwb_range_distribution.append(j['range'])

    return topic_to_processing['/uwb_ranges'][1]

def aggregate_imu(topic_to_processing, imu_csv):
    for j in topic_to_processing['/camera/camera/imu'][1]:
        csv_row = []
        for k, v in j.items(): csv_row.append(v)
        imu_csv.append(csv_row)
    
    return topic_to_processing['/camera/camera/imu'][1]

def aggregate_infra1(topic_to_processing, out_infra1):
    data = []
    for j in topic_to_processing['/camera/camera/infra1/image_rect_raw'][1]:
        cv2.imwrite(out_infra1+"/"+j["name"], j["raw"])
        j_no_image = { k:v for k,v in j.items() if not (k == "raw") }
        data.append(j_no_image)
    return data

def aggregate_infra2(topic_to_processing, out_infra1):
    data = []
    for j in topic_to_processing['/camera/camera/infra2/image_rect_raw'][1]:
        cv2.imwrite(out_infra1+"/"+j["name"], j["raw"])
        j_no_image = { k:v for k,v in j.items() if not (k == "raw") }
        data.append(j_no_image)
    return data

def aggregate_tracker(trackbody_to_mybody_func, tracker_data_tum):
    tracker_body_poses = []  # All T_world_to_body as flattened HTM

    for i in range(tracker_data_tum.shape[0]-1):
        tracker_pose = slam_quat_to_HTM(tracker_data_tum[i, :])   # tracker pose as HTM
        T_world_to_body = trackbody_to_mybody_func(tracker_pose)     # convert tracker to body frame
        tracker_body_poses.append([tracker_data_tum[i, 0]] + list(T_world_to_body.flatten()))
    
    tracker_body_poses = np.array(tracker_body_poses)

    # Compute velocity for body (using translation columns of HTM)
    dt = np.diff(tracker_body_poses[:, 0])
    dx = np.diff(tracker_body_poses[:, 4]) / dt
    dy = np.diff(tracker_body_poses[:, 7]) / dt
    dz = np.diff(tracker_body_poses[:, 10]) / dt
    tracker_body_velocities = np.vstack((
        tracker_body_poses[:-1, 0],  # timestamps
        dx, dy, dz
    )).T

    return tracker_body_poses, tracker_body_velocities

def aggregate_assisted_uwb(uwb_json, trackbody_to_mybody_func, tracker_data_tum, N_POINTS):
    assisted_uwb_json = []

    # Still using 'vicon' syntax but thats ok who cares
    # Must interpolate in original frame, then apply your transform: hence trackbody_to_mybody_func
    
    vwf = tracker_data_tum # 'vicon world frame' instead of 'slam world frame'
    
    for u in uwb_json:
        # Find the closest timestamp vicon measurements to our UWB range
        tdiffs = np.abs(vwf[:,0] - u["t"])
        vicon_idx1 = np.argmin(tdiffs)
        tdiffs[vicon_idx1] = np.inf
        vicon_idx2 = np.argmin(tdiffs)
        istart, iend = sorted([vicon_idx1, vicon_idx2]) # Make sure indices are ascending

        start_pose = slam_quat_to_HTM(vwf[istart, :]) # Convert quat -> HTM
        end_pose = slam_quat_to_HTM(vwf[iend, :])

        interp_pose = interpolate_pose(
            start_pose, vwf[istart, 0],
            end_pose, vwf[iend, 0],
            u["t"], N_POINTS
        ) # Interpolated pose is a T_world_to_cam1

        interp_pose_world = trackbody_to_mybody_func(interp_pose)

        u2 = copy.deepcopy(u)
        u2["type"] = "assisted_uwb"
        u2["T_body_world"] = interp_pose_world
        assisted_uwb_json.append(u2)

    return assisted_uwb_json