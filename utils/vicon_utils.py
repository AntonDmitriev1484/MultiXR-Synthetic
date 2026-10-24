
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from utils.math_utils import * 

def euler_to_tum(arr, degrees=True):
    """
    Convert Euler + translation to TUM format [t tx ty tz qx qy qz qw].
    """
    rot = R.from_euler('xyz', arr[1:4], degrees=degrees)
    qx, qy, qz, qw = rot.as_quat()  # [x, y, z, w]
    return np.array([arr[0], arr[4], arr[5], arr[6], qx, qy, qz, qw])


def parse_vicon(file):
    """
    Old Vicon parser for Euler-angle style exports.
    """
    rows = []
    frame_counter = 1
    line_counter = 0
    with open(file, 'r') as f:
        for line in f:
            if line_counter >= 5:
                row = [x for x in line.strip().split(',')]
                if '' in row:  # Missing pose → repeat last
                    row = rows[len(rows)-1]
                    row[0] = frame_counter
                row = [float(x) for x in row]
                rows.append(row)
                frame_counter += 1
            line_counter += 1

    data = np.array(rows)

    # Example indexing (custom per dataset)
    headset_arr_rpy = data[:, [0, 2, 3, 4, 5, 6, 7]]
    anchor_arr_rpy = data[:, [0, 8, 9, 10, 11, 12, 13]]

    headset_arr = np.array([euler_to_tum(row) for row in headset_arr_rpy])
    anchor_arr = np.array([euler_to_tum(row) for row in anchor_arr_rpy])

    # Scale to meters
    headset_arr[:, 1:4] /= 1000
    anchor_arr[:, 1:4] /= 1000

    return headset_arr, anchor_arr


def parse_vicon_csv(file, subject_filter=None):
    """
    Parse Vicon CSV with columns:
    timestamp,frame,subject,segment,x,y,z,qx,qy,qz,qw

    Args:
        file (str): path to CSV
        subject_filter (str or None): if given, only return this subject’s poses

    Returns:
        dict[str, np.ndarray]: {subject: array[N,8]} with rows [t tx ty tz qx qy qz qw]
    """
    subjects = {}

    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["timestamp"]) / 1e3 # Convert from ms to s
            subj = row["subject"]

            tx = float(row["x"]) / 1e3  # mm → m
            ty = float(row["y"]) / 1e3
            tz = float(row["z"]) / 1e3
            qx = float(row["qx"])
            qy = float(row["qy"])
            qz = float(row["qz"])
            qw = float(row["qw"])

            pose = np.array([t, tx, ty, tz, qx, qy, qz, qw])

            if subject_filter is None or subj == subject_filter:
                if subj not in subjects:
                    subjects[subj] = []
                subjects[subj].append(pose)

    # convert to numpy arrays
    for subj in subjects:
        subjects[subj] = np.vstack(subjects[subj])

    return subjects

# Crop all Vicon data to be within the ROS timestamps
# Assumption is that Vicon data timestamps are clock synced with NUC.
def crop_vicon(vicon_data, start, end):

    for tracked_name, data in vicon_data.items():
        data = [ d for d in data if start < d[0] and d[0] < end ] # Doesn't mutate vicon data
        vicon_data[tracked_name] = data # this does
    return vicon_data

def clean_vicon(vicon_data):

    # If you're mobile and translation suddenly drop to 0, that means tracking was lost. interpolate that thang

    for tracked_name, data in vicon_data.items():

        # In case we start off at a 0 pose, find the first non-zero pose
        # and set that to be our start pose

        def is_outlier(tum_pose):
            norm = np.linalg.norm(np.array(tum_pose)[1:])
            return norm <= 1e-5

        # If our starting pose is an outlier and we have nothing to interpolate between
        start_pose = None
        for i in range(0, len(data)):
            if not is_outlier(data[i]): 
                start_pose = np.array(data[i]) # Next valid TUM timestamped pose
                break

        for p in range(0,i):
            data[p] = start_pose

        # Now clean
        for i in range(1, len(data)):
            if is_outlier(data[i]):

                last_pose = np.array(data[i-1]) # Last valid TUM timestamped pose
                next_pose = None
                interp_pose = None
                for j in range(i+1, len(data)): # Find next valid TUM timestamped pose
                    # print(data)
                    if not is_outlier(data[j]): 
                        next_pose = np.array(data[j]) # Next valid TUM timestamped pose
                        current_timestamp = data[i][0]
                        interp_pose = interpolate_pose(
                            slam_quat_to_HTM(last_pose), last_pose[0],
                            slam_quat_to_HTM(next_pose), next_pose[0],
                            current_timestamp, 100
                        )
                        break

                if interp_pose is not None:
                    interp_pose = HTM_to_TUM(interp_pose) # Returns a non timestamped HTM
                    data[i] = np.insert(interp_pose, 0, current_timestamp) #I'm pretty sure this mutates the original array?

    vicon_data[tracked_name] = data
    return vicon_data

def get_tx_position(T_vuwb_to_uwbtx, data):
    positions = []
    for pose in data: # Loop through until you find a pose that is not an outlier
        if np.linalg.norm(np.array(pose)[1:]) > 1e-5:
            T_vuwb_to_world = slam_quat_to_HTM(pose)
            T_world_to_tx = T_vuwb_to_uwbtx @ np.linalg.inv(T_vuwb_to_world)
            position = np.linalg.inv(T_world_to_tx)[:3,3]
            return position
    # for pose in data:
    #     if np.linalg.norm(np.array(pose)[1:4]) != 0: # Filter out lost tracking outliers
    #         T_vuwb_to_world = slam_quat_to_HTM(pose)
    #         T_world_to_tx = T_vuwb_to_uwbtx @ np.linalg.inv(T_vuwb_to_world)
    #         positions.append(np.linalg.inv(T_world_to_tx)[:3,3])
    # return np.average(np.array(positions), axis=0)