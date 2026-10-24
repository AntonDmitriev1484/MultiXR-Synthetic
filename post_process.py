
from pathlib import Path
import pkgutil
import importlib
import inspect
import os
import json
import csv
import yaml
import argparse

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from types import SimpleNamespace

import shutil
import math
import copy

from utils.load_rostypes import *
from utils.ros_msg_handlers import *
from utils.math_utils import *
from utils.vicon_utils import *
from utils.optitrack_utils import *
from utils.uwb_utils import *
from utils.ros1_slam_utils import *
from utils.perturb import perturb


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, '__dict__'):
            return vars(obj)
        return super().default(obj)

### Process optitrack data before alignment
def prep_optitrack_output(args, START, END, T, opti_data):
    opti_json = []
    body_opti_tum_traj = [] # Explain: Used later in UWB error code
    body_opti_HTMs = [] # Explain: Used later in Umeyama alignment
    if args.opti is not None:
        def opti_tracked_body_to_my_body(T_head_to_world):
            return T.T_head_to_body @ np.linalg.inv(T_head_to_world)
        
        opti_body_poses, opti_body_velocities = aggregate_tracker(opti_tracked_body_to_my_body, np.array(opti_data))
        body_opti_HTMs = opti_body_poses
        # Each pose is the IMU pose in the optitrack world frame.

        # aggregate_tracker returns [timestamp, flattened HTM]
        # for opti_tum_traj we need to convert back to tum
        for pose in opti_body_poses: body_opti_tum_traj.append(slam_HTM_to_TUM(pose))

        # If we passed a valid frequency to subsample to
        opti_freq = len(opti_data) / (END-START)
        if opti_freq > args.opti > 0:
            skip = math.ceil(opti_freq / args.opti) # Number of poses to skip in subsampling to synth slam frequency
            opti_body_poses = np.array(opti_body_poses)
            opti_body_poses = opti_body_poses[::skip] # Finally, subsample to required frequency

        opti_json = [ {
                "t": float(body_pose[0]),
                "type": "opti_pose",
                "T_body_world" : body_pose[1:].reshape((4,4)),
            } for body_pose, body_v in zip( list(opti_body_poses), list(opti_body_velocities))]
        
    return opti_json, body_opti_HTMs, body_opti_tum_traj

### Process SLAM data before alignment
def prep_orbslam3_output(args, START, END, T, post_slam_data, live_slam_data=[], real_failures=False):
    post_slam_json = []
    live_slam_json = []
    body_post_slam_HTMs = [] # Used later in Umeyama alignment
    body_live_slam_HTMs = [] # Used later in Umeyama alginment, if real failures.

    print(f"{real_failures=}")

    if args.slam is not None:
        post_slam_data[:,0] *= 1e-9 # Adjust timestamps to be in 's'

        def slam_tracked_body_to_my_body(T_cam1_to_sorigin): # SLAM quat gives you the transform from cam1 frame to slam origin
            return T.T_cam1_to_body @ np.linalg.inv(T_cam1_to_sorigin)

        slam_body_poses, slam_body_velocities = aggregate_tracker(slam_tracked_body_to_my_body, post_slam_data)
        body_post_slam_HTMs = slam_body_poses
        # Each pose is the IMU pose in the slam frame. i.e. T_slamworld_to_imu. This is what plotting assumes.

        # If we passed a valid frequency to subsample to
        slam_freq = len(post_slam_data) / (END-START)
        if slam_freq > args.slam > 0:
            skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
            slam_body_poses = np.array(slam_body_poses)
            slam_body_poses = slam_body_poses[::skip] # Finally, subsample to required frequency

        # Convert from nparray to json format
        post_slam_json = [ {
                "t": float(body_pose[0]),
                "type": "slam_pose",
                "T_body_world" : body_pose[1:].reshape((4,4)),

            } for body_pose, body_v in zip( list(slam_body_poses), list(slam_body_velocities))]
        
        if real_failures:
            live_slam_data[:,0] *= 1e-9 # Adjust timestamps to be in 's'
            live_slam_body_poses, live_slam_body_velocities = aggregate_tracker(slam_tracked_body_to_my_body, live_slam_data)
            body_live_slam_HTMs = live_slam_body_poses
            # IF THERES A REAL FAILURE, OVERWRITE unaligned save_traj SLAM with live SLAM
            live_slam_json = [ {
                    "t": float(body_pose[0]),
                    "type": "slam_pose",
                    "T_body_world" : body_pose[1:].reshape((4,4)),

                } for body_pose, body_v in zip( list(live_slam_body_poses), list(live_slam_body_velocities))]

    return post_slam_json, live_slam_json, body_post_slam_HTMs, body_live_slam_HTMs

### Just annotate
def prep_annotate_slam_output(args, START, END, T, aligned_slam_data, mes_type, real_failures=None):


    def identity(T_body_to_world): # Already in body frame.
        return T_body_to_world
    # Aggregate is basically just for format conversion
    aligned_slam_body_poses, aligned_slam_body_velocities = aggregate_tracker(identity, np.array(aligned_slam_data))
    # Each pose is the IMU pose in the optitrack world frame.

    # Also do all of the labeling in here of tracking vs no tracking
            # If we passed a valid frequency to subsample to
    slam_freq = len(aligned_slam_data) / (END-START)
    if slam_freq > args.slam > 0:
        skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
        aligned_slam_body_poses = np.array(aligned_slam_body_poses)
        aligned_slam_body_poses = aligned_slam_body_poses[::skip] # Finally, subsample to required frequency
    

    aligned_live_slam_json = [ {
            "t": float(body_pose[0]),
            "type": mes_type,
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": 0,
                    "vy": 0,
                    "vz": 0
            }
        } for body_pose, body_v in zip( list(aligned_slam_body_poses), list(aligned_slam_body_velocities))]
    

    
    if real_failures is not None and len(real_failures) > 0: # Annotate failures if they're present
        real_failures = real_failures[0]
        fail_start_t = START + real_failures["start"] # Timestamp failure starts and we rely on IMU
        new_map_start_t = START + real_failures["init_newmap"] # Timestamp that we start tracking a new map
        relocalized_t = START + real_failures["end"] # Timestamp we end failure
        imu_only_idxs = np.where( 
            (new_map_start_t >= aligned_slam_body_poses[:,0]) & 
            (aligned_slam_body_poses[:,0]>= fail_start_t)
            )[0]
        newmap_idxs = np.where( 
            (relocalized_t >= aligned_slam_body_poses[:,0]) &
            (aligned_slam_body_poses[:,0] >= new_map_start_t)
            )[0]
        # Apply tracking status labels
        for i in range(len(aligned_live_slam_json)):
            if i in imu_only_idxs: aligned_live_slam_json[i]["status"] = "imu"
            elif i in newmap_idxs: aligned_live_slam_json[i]["status"] = "newmap"
            else: aligned_live_slam_json[i]["status"] = "tracking"
    else: 
        for i in range(len(aligned_live_slam_json)): aligned_live_slam_json[i]["status"] = "tracking"
        
    return aligned_live_slam_json

### Annotate and transform
def prep_synth_fail_slam_output(args, START, END, T, aligned_slam_data, mes_type, synth_failures=None):

    def identity(T_body_to_world): # Already in body frame.
        return T_body_to_world
    # Aggregate is basically just for format conversion
    aligned_slam_body_poses, aligned_slam_body_velocities = aggregate_tracker(identity, np.array(aligned_slam_data))

    imu_only_idxs = []
    newmap_idxs = []
    if synth_failures is not None and len(synth_failures) > 0:

        synth_failures = synth_failures[0] # For now just assuming 1 failure
        fail_start_t = START + synth_failures["start"] # Timestamp failure starts and we rely on IMU
        new_map_start_t = START + synth_failures["init_newmap"] # Timestamp that we start tracking a new map
        relocalized_t = START + synth_failures["end"] # Timestamp we end failure

        print(f" {fail_start_t} {new_map_start_t} {relocalized_t}")

        imu_only_idxs = np.where( (new_map_start_t >= aligned_slam_body_poses[:,0]) & (aligned_slam_body_poses[:,0]>= fail_start_t))[0]
        newmap_idxs = np.where( (relocalized_t >= aligned_slam_body_poses[:,0]) & (aligned_slam_body_poses[:,0] >= new_map_start_t))[0]
        # Create the newmap segment that starts at the origin
        # newmap_idxs = np.where( relocalized_t >= aligned_slam_body_poses[:,0] >= new_map_start_t)
        
        traj_origin = aligned_slam_body_poses[0,1:].reshape((4,4)).copy()
        new_map_origin = aligned_slam_body_poses[newmap_idxs[0],1:].reshape((4,4)).copy() # This is world to body
        for i in newmap_idxs:
            body_pose = aligned_slam_body_poses[i,1:].reshape((4,4)).copy() # This is world to body
            new_map_pose =  body_pose @ np.linalg.inv(new_map_origin) @ traj_origin
            aligned_slam_body_poses[i,1:] = new_map_pose.flatten()

    # Also do all of the labeling in here of tracking vs no tracking

    aligned_post_slam_json = [ {
            "t": float(body_pose[0]),
            "type": mes_type,
            "T_body_world" : body_pose[1:].reshape((4,4)),
            "v_world": {
                    "vx": 0,
                    "vy": 0,
                    "vz": 0
            }
        } for body_pose, body_v in zip( list(aligned_slam_body_poses), list(aligned_slam_body_velocities))]
    
    # Apply tracking status labels
    for i in range(len(aligned_post_slam_json)):
        if i in imu_only_idxs: aligned_post_slam_json[i]["status"] = "imu"
        elif i in newmap_idxs: aligned_post_slam_json[i]["status"] = "newmap"
        else: aligned_post_slam_json[i]["status"] = "tracking"

    body_slam_aligned_tum_traj = [slam_HTM_to_TUM(p) for p in aligned_slam_body_poses]

    return aligned_post_slam_json, body_slam_aligned_tum_traj

### Synthetic data layer
# Converts the generate.py output in generated/{trial_name}/ into the inputs the original pipeline expects.

ROOT = os.path.dirname(os.path.abspath(__file__))

def define_synthetic_transforms():
    # generate.py assumes the head, IMU, camera and decawave frames all coincide with the body.
    T = SimpleNamespace()
    for name in ["T_optiuwb_to_uwbtx", "T_imu_to_body", "T_body_to_imu", "T_head_to_cam1", "T_imu_to_cam1",
                 "T_cam1_to_body", "T_head_to_body", "T_inertial_to_world", "T_head_to_decawave",
                 "T_body_to_decawave"]:
        setattr(T, name, np.eye(4))
    return T

def load_vive(inpath):
    # vive_synth.json stores T_body_world, i.e. world -> head. Invert to get head -> world.
    vive_json = json.load(open(f"{inpath}/vive_synth.json", "r"))
    t = np.array([j["t"] for j in vive_json])
    T_head_to_world = np.array([np.linalg.inv(np.array(j["T_body_world"])) for j in vive_json])
    return t, T_head_to_world

def load_synthetic_anchors(dataset_path):
    anchors = json.load(open(f"{dataset_path}/anchors.json", "r"))
    for a in anchors: a["position"] = np.array(a["position"])
    return anchors

def synthetic_user_ids(dataset_path):
    return sorted(int(d) for d in os.listdir(dataset_path) if d.isdigit())

def post_process(args):

    ID = args.id

    outpath = f"{ROOT}/post/{ID}/{args.trial_name}_post"
    out_world = f"{outpath}/world/"  # Vicon can define apriltags and anchors set up in world frame
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(out_world, exist_ok=True)

    dataset_path = f"{args.generated_dir}/{args.trial_name}"
    inpath = f"{dataset_path}/{ID}"

    # Synthetic data has no real SLAM failures
    real_failures = False

    imu_json = json.load(open(f"{inpath}/imu_synth.json", "r"))
    uwb_json = json.load(open(f"{inpath}/uwb_synth.json", "r"))

    vive_t, T_head_to_world = load_vive(inpath)
    metadata = {"start_ns": vive_t[0] * 1e9, "end_ns": vive_t[-1] * 1e9}

    # Emulate single user visual-inertial SLAM: the SLAM origin is the first head position and heading,
    # but gravity aligned (pitch and roll dropped), like ORB-SLAM3 in inertial mode.
    # Rows are [t_ns, TUM of T_cam1_to_sorigin], matching the ORB-SLAM3 'save_traj' text file.
    first_pose = T_head_to_world[0].copy()
    yaw = np.arctan2(first_pose[1, 0], first_pose[0, 0])
    first_pose[:3, :3] = [[np.cos(yaw), -np.sin(yaw), 0.0],
                          [np.sin(yaw), np.cos(yaw), 0.0],
                          [0.0, 0.0, 1.0]]
    traj_deltas = [np.linalg.inv(first_pose) @ pose for pose in T_head_to_world]
    post_slam_data = np.array([[t * 1e9] + HTM_to_TUM(pose) for t, pose in zip(vive_t, traj_deltas)])
    live_slam_data = post_slam_data.copy()

    # Ground truth head poses in the world frame stand in for optitrack: rows are [t_s, TUM of T_head_to_world]
    opti_helmet_traj = [np.array([t] + HTM_to_TUM(pose)) for t, pose in zip(vive_t, T_head_to_world)]


    # # Filter for messages within bag timestamp range.
    # START = metadata["start_ns"] * 1e-9

    START = post_slam_data[0,0] * 1e-9 # Adjust timestamps to be in 's'

    END = metadata["end_ns"] * 1e-9
    print(f"Data duration {START} - {END}")


    # if args.opti:
    opti_data = crop_opti(opti_helmet_traj, START, END)
    opti_data = clean_opti(opti_data)



    def filtt(arr): # For filtering a json output
        return list(filter(lambda x: (START <= x["t"] <= END), arr)) # Then filter by ros timestamps
    def filtt2(arr): # For filtering a CSV output
        return list(filter(lambda x: (START <= x[0] <= END), arr))


    ### Define all coordinate frames in T
    T = define_synthetic_transforms()

    # Position of each UWB anchor transmitter in the world frame
    anchor_positions = {}
    if args.opti is not None:
        anchor_positions = load_synthetic_anchors(dataset_path)

    ### Process SLAM data
    # All trajectories have a post SLAM, and if they have real failures, they will have a live SLAM as well.
    # Convert to body frame, subsample, 
    # output body in SLAM world frame trajectory as jsons, and HTMs (for later alignment)
    post_slam_json, live_slam_json, body_post_slam_HTMs, body_live_slam_HTMs = (
        prep_orbslam3_output(args, START, END, T, post_slam_data, live_slam_data, real_failures)
        if args.slam is not None
        else ([], [], [], [])
    )

    ### Process Optitrack data
    # Convert to body frame, subsample, 
    # output body in optitrack world frame trajectory as jsons, and HTMs (for later alignment)
    # ouptut as a TUM trajectory, for later UWB error calculation code
    opti_json, body_opti_HTMs, body_opti_tum_traj = (
        prep_optitrack_output(args, START, END, T, opti_data)
        if args.opti is not None
        else ([], [], [])
    )

    ### Align SLAM trajectories to optitrack shared frame
    aligned_post_slam_json = []
    aligned_live_slam_json = []

    localframe_live_slam_json = []

    body_post_slam_aligned_tum_traj = []
    body_live_slam_aligned_tum_traj = []

    
    if args.align:


        if real_failures:
            # Compute alignment using the SLAM 'save_traj' trajectory to the Optitrack trajectory
            # apply alignment to SLAM 'ros1 bag' trajectory

            real_failures = json.load(open(f'./real_fails/{args.trial_name}_nuc{args.id}_fail.json','r'))
            all_data_start_ts = metadata["start_ns"] * 1e-9
            fail_end_ts = real_failures[0]["end"] + all_data_start_ts      
            # output body live slam trajectory - for use in evaluation
            body_post_slam_aligned_tum_traj, body_live_slam_aligned_tum_traj, aligned_ts = bonus_umeyama_alignment(body_opti_HTMs, body_post_slam_HTMs, body_live_slam_HTMs, fail_end_ts)

            # annotate the live slam trajectory with real failures file - for use in plotting
            aligned_live_slam_json = prep_annotate_slam_output(args, all_data_start_ts, END, T, np.array(body_live_slam_aligned_tum_traj), "aligned_live_slam_pose", real_failures)

            # annotate the post slam trajectory in the same way - for use in graph
            aligned_post_slam_json = prep_annotate_slam_output(args, all_data_start_ts, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_slam_pose", real_failures)

            # Crop to the same timestamps as the aligned trajectory, so both share the same origin pose
            A = [ slam_HTM_to_TUM(h) for h in body_live_slam_HTMs if np.isin(h[0], aligned_ts) ]
            localframe_live_slam_json = prep_annotate_slam_output(args, all_data_start_ts, END, T, np.array(A), "localframe_live_slam_pose", real_failures)

        elif args.synth_failures: 
            # For now lets just assume that I will never add a synthetic failure to a real failure trajectory
            # otherwise the synthetic will overwrite the real failure annotations and it'll make a mess.
            # Since we're creating synthetic failures, we're using post_slam

            synth_failures = []
            try:
                synth_failures = json.load(open(f'{ROOT}/post/{ID}/synth_failures/{args.trial_name}.json','r'))
            except Exception as err:
                print("No synth failures detected, setting synth_failures = []")

            all_data_start_ts = metadata["start_ns"] * 1e-9
            # fail_end_ts = synth_failures[0]["end"] + all_data_start_ts   

            ### Add a synthetic failure to the optitrack-aligned trajectory 

            body_post_slam_aligned_tum_traj, aligned_ts = umeyama_alignment1(body_opti_HTMs, body_post_slam_HTMs)

            # annotate the post slam trajectory, for use in the graph
            aligned_post_slam_json = prep_annotate_slam_output(
                args, all_data_start_ts, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_slam_pose", synth_failures)

            # add newmap deformation and IMU segment, 
            # and annotate live slam trajectory for use in plotting (aligned_live_slam_json) and evaluation (body_live_slam_aligned_tum_traj)
            aligned_live_slam_json, body_live_slam_aligned_tum_traj = prep_synth_fail_slam_output(
                args, all_data_start_ts, END, T, np.array(body_post_slam_aligned_tum_traj), "aligned_live_slam_pose", synth_failures)


            ### Add a synthetic failure to the local frame trajectory

            # Crop to the same timestamps as the aligned trajectory, so both share the same origin pose
            body_post_slam_tum_traj = [ slam_HTM_to_TUM(h) for h in body_post_slam_HTMs if np.isin(h[0], aligned_ts) ]

            # add newmap deformation and IMU segment, 
            # and annotate live slam trajectory for use in plotting (aligned_live_slam_json) and evaluation (body_live_slam_aligned_tum_traj)
            localframe_live_slam_json, _ = prep_synth_fail_slam_output(
                args, all_data_start_ts, END, T, np.array(body_post_slam_tum_traj), "localframe_live_slam_pose", synth_failures)


        else:
            print("No real or synthetic failures! Aligning regular-style")
            body_post_slam_aligned_tum_traj, aligned_ts = umeyama_alignment1(body_opti_HTMs, body_post_slam_HTMs)

            def identity(T_body_to_world): # Already in body frame.
                return T_body_to_world
            # Aggregate is basically just for format conversion
            aligned_post_slam_body_poses, aligned_post_slam_body_velocities = aggregate_tracker(identity, np.array(body_post_slam_aligned_tum_traj))
            
                        # If we passed a valid frequency to subsample to
            slam_freq = len(aligned_post_slam_body_poses) / (END-START)
            if slam_freq > args.slam > 0:
                skip = math.ceil(slam_freq / args.slam) # Number of poses to skip in subsampling to synth slam frequency
                aligned_post_slam_body_poses = np.array(aligned_post_slam_body_poses)
                aligned_post_slam_body_poses = aligned_post_slam_body_poses[::skip] # Finally, subsample to required frequency
            
            aligned_post_slam_json = [ {
                    "t": float(body_pose[0]),
                    "type": "aligned_slam_pose",
                    "T_body_world" : body_pose[1:].reshape((4,4)),
                    "v_world": {
                        "vx": 0,
                        "vy": 0,
                        "vz": 0
                    },
                    "status": "tracking"
                } for body_pose, body_v in zip( list(aligned_post_slam_body_poses), list(aligned_post_slam_body_velocities))]
    
            # Just make live slam a copy of post SLAM
            body_live_slam_aligned_tum_traj = body_post_slam_aligned_tum_traj
            aligned_live_slam_json = copy.deepcopy(aligned_post_slam_json)
            for j in aligned_live_slam_json: j["type"] = "aligned_live_slam_pose"

    ### Annoying extra code we need to add to run cappella with real trials
    real_trials = [
    'multi2_follow_loss2',
    'multi2_trip_loss',
    'multi2_board_loss3'
    ]
    if args.trial_name in real_trials and not real_failures:
        print(f"Cappella for {args.id}")
        localframe_live_slam_json = copy.deepcopy(post_slam_json)
        for u in localframe_live_slam_json: u["type"] = "localframe_live_slam_pose"

    ### Apply sensitivity analysis operations

    # For testing a synthetic network delay before using each UWB range.
    if args.synth_delay is not None:
        for mes in uwb_json:
            mes["t"] += args.synth_delay * 1e-3

    if args.median_filter:
        """
        HOW IT WORKS:
        Each output sample is the median of a symmetric window of 2w+1 raw samples
        centered on the current sample. The median is immune to up to w outliers
        within the window. No detection step — the median itself is both the
        detector and the replacement.

        REPLACEMENT:
        The median of the local window replaces every sample,
        including clean ones (slight smoothing of the true signal).

        PARAMETERS:
        w : half-window size. w=5 → 11-sample window.
            Larger w → more robust to long NLoS bursts, more lag.
        """
        def moving_median(x, w=5):
            x = np.asarray(x, float)
            n = len(x)
            y = x.copy()
            h = w // 2

            for i in range(n):
                y[i] = np.median(x[max(0, i-h):min(n, i+h+1)])

            return y
        
        for id in [1,2,3,4,5]:
            all_mes = [j for j in uwb_json if j["id"]==id]
            x = np.array([j["range"] for j in all_mes])
            x_new = moving_median(x) # Apply filter
            for i, mes in enumerate(all_mes): mes["range"] = x_new[i] # Update references

    if args.cfar_filter:

        import cfar
        from collections import defaultdict

        def apply_cfar_to_uwb(raw_json, k=2.0, method="CA-CFAR"):
            """
            Runs CFAR per (src, id) group and overwrites range in-place.
            """

            # pick CFAR function
            fn = cfar.CFAR_VARIANTS[method]

            # group indices into sequences
            groups = defaultdict(list)
            for i, r in enumerate(raw_json):
                if r.get("type") != "uwb":
                    continue
                groups[(args.id, r["id"])].append(i)

            for key, idxs in groups.items():
                idxs = sorted(idxs, key=lambda i: raw_json[i]["t"])

                t = np.array([raw_json[i]["t"] for i in idxs])
                x = np.array([raw_json[i]["range"] for i in idxs])

                # run CFAR
                result = fn(t, x, k=k)

                # CFAR returns (reject_mask, threshold)
                if isinstance(result, tuple):
                    reject = result[0]
                else:
                    reject = result

                # interpolate cleaned signal
                x_new = cfar.interp_accepted(t, x, reject)

                # overwrite original JSON entries
                for j, i in enumerate(idxs):
                    raw_json[i]["range"] = float(x_new[j])

            return raw_json
        
        uwb_json = apply_cfar_to_uwb(uwb_json)

    if args.dw_cfar_filter:
        import dw_cfar
        from collections import defaultdict


        def apply_dw_cfar_to_uwb(raw_json, src_id, k=2.0):
            """
            Runs the DW+CFAR hybrid filter and overwrites ranges in-place.
            """

            groups = defaultdict(list)

            for i, r in enumerate(raw_json):
                if r.get("type") != "uwb":
                    continue

                groups[(src_id, r["id"])].append(i)

            for _, idxs in groups.items():

                idxs = sorted(idxs, key=lambda i: raw_json[i]["t"])

                t = np.array([raw_json[i]["t"] for i in idxs])
                x = np.array([raw_json[i]["range"] for i in idxs])

                recs = [raw_json[i] for i in idxs]

                reject, _, _, _, _, _, _ = dw_cfar.dw_cfar_detector(
                    t,
                    x,
                    recs,
                    k
                )

                x_new = dw_cfar.interp_accepted(t, x, reject)

                for j, i in enumerate(idxs):
                    raw_json[i]["range"] = float(x_new[j])

            return raw_json

        uwb_json = apply_dw_cfar_to_uwb(
            uwb_json,
            src_id=args.id,
            k=1.5
        )

    synth_uwb_json = []
    # if args.id == 4:
        # synth_uwb_json = range_synthesizer2(START, END, body_opti_tum_traj, T, f"/home/antond2/Desktop/Research/MultiXR-Post/merged/{args.trial_name}_merged")
    

    # Perturb every SLAM trajectory by a random translation + yaw (about each trajectory's first pose)
    if args.perturb_t is not None or args.perturb_r is not None:
        post_slam_json, aligned_post_slam_json, aligned_live_slam_json, localframe_live_slam_json, opti_json = perturb(
            args.perturb_t, args.perturb_r, post_slam_json, aligned_post_slam_json, aligned_live_slam_json, localframe_live_slam_json, opti_json)

    # Compose the final factor graph dataset
    all_data =uwb_json + imu_json + opti_json + synth_uwb_json \
            + post_slam_json + aligned_post_slam_json + aligned_live_slam_json \
                  + localframe_live_slam_json # Add on for Cappella
    
    for mes in all_data: mes["src"] = ID


    ### Hard coding things for evaluation
    # # Remove Node 2 from being used in multi2_board_loss3, it has un-controlled tracking loss.
    if args.trial_name == "multi2_board_loss3":
        for j in all_data:
            if j["src"] == 2:
                if j["type"] == "aligned_slam_pose": j["status"] = "lost"
                if j["type"] == "aligned_live_slam_pose": j["status"] = "lost"
                if j["type"] == "slam_pose": j["status"] = "lost" 

    # Remove Node 5 from being used in multi2_.._nlos trials. Optitrack didn't track it properly.
    if args.trial_name == "multi2_human_nlos" or args.trial_name == "multi2_object_nlos":
        copy_all_data = []
        for j in all_data:
            if (j["type"] == "uwb"): 
                if not (j["id"] == 5): copy_all_data.append(j)
            else: copy_all_data.append(j)
        all_data = copy_all_data

    ### Copy all world information: T, anchors, apriltags, to output

    if args.opti is not None:
        out_anchors = open(f'{outpath}/anchors.json', 'w')
        json.dump(anchor_positions, out_anchors, cls=NumpyEncoder, indent=1)


    with open(f'{outpath}/transforms.json', 'w') as fs: json.dump(vars(T), fs, cls=NumpyEncoder, indent=1)

    # Run sanity check to make sure measurements are at the frequency we expect them to be before testing in the graph

    print("Checking frequency of input data")
    print(f" Measured Synth UWB frequency {len(synth_uwb_json) / (END-START)}")
    print(f" Measured UWB frequency {len(uwb_json) / (END-START)}")
    if args.opti: print(f" Measured optitrack frequency {len(opti_data) / (END-START)}")
    print(f" Measured SLAM frequency {len(post_slam_data) / (END-START)}")

    print("Checking frequency of subsampled output")
    if args.opti: print(f" Measured subsampled optitrack frequency {len(opti_json) / (END-START)}")
    print(f" Measured subsampled SLAM frequency {len(post_slam_json) / (END-START)}")


    # Filter to make sure all messages ( and data jsons ) fall within the ROS recording time interval, (because some of them don't apparently)
    all_data = filtt(all_data)
    all_data = sorted(all_data, key=lambda x: x["t"])

    json.dump(all_data, open(outpath+"/all.json", 'w'), cls=NumpyEncoder, indent=1)

    # Write body in optitrack trajectory  to a csv file for loading into EVO
    np.savetxt(f"{outpath}/opti.txt", np.array(body_opti_tum_traj), fmt="%.8f")
    np.savetxt(f"{outpath}/aligned_slam.txt", np.array(body_post_slam_aligned_tum_traj), fmt="%.8f")
    np.savetxt(f"{outpath}/aligned_live_slam.txt", np.array(body_live_slam_aligned_tum_traj), fmt="%.8f")  
    # np.savetxt(f"{outpath}/opti_inv.txt", np.array(body_opti_inv_tum_traj), fmt="%.8f")
    # np.savetxt(f"{outpath}/slam_inv.txt", np.array(body_slam_inv_tum_traj), fmt="%.8f")

    if args.opti is None: 
        return all_data, {}, T, body_opti_tum_traj
    else:
        return all_data, anchor_positions, T, body_opti_tum_traj
    

if __name__ == "__main__":
    # Example usage:
    # Single user
        # python3 post_process.py 2 opti_circle_test1 --opti --slam 20
        # keep opti at default frequency, subsample SLAM to 20Hz.
    # Multi user
        # python3 post_process.py 0 opti_multi_circle --opti --slam -a -f --multi_merge
    # Synthetic data: trial_name is a dataset in generated/, e.g.
        # python3 post_process.py 0 mobile_5_static_3 --opti --slam -a --multi_merge

    NO_SUBSAMPLE = -10
    parser = argparse.ArgumentParser(description="Stream collector")
    parser.add_argument("id", type=int)
    parser.add_argument("trial_name" , type=str)
    parser.add_argument("--opti", nargs="?", const=0, type=float)
    parser.add_argument("--slam", nargs="?", const=0, type=float)
    # Replace SLAM tracked body in SLAM frame trajectory, with SLAM tracked body in optitrack frame trajectory
    parser.add_argument("--align", "-a", action="store_true")
    # Load a json file containing tracking lost segments
    parser.add_argument("--synth_failures", "-f", action="store_true")
    parser.add_argument("--multi_merge", action="store_true")
    parser.add_argument("--check_uwb_error", action="store_true")
    parser.add_argument("--synth_delay", type=int)
    parser.add_argument("--median_filter", action="store_true")
    parser.add_argument("--cfar_filter", action="store_true")
    parser.add_argument("--dw_cfar_filter", action="store_true")
    parser.add_argument("--anchor_prior_error", type=float)
    # Perturb SLAM trajectories by perturb_t meters (random 3D direction) then perturb_r degrees of yaw
    parser.add_argument("--perturb_t", type=float)
    parser.add_argument("--perturb_r", type=float)
    # Directory containing the generate.py datasets
    parser.add_argument("--generated_dir", default=f"{ROOT}/generated")

    args = parser.parse_args()

    if args.multi_merge:

        outpath = f'{ROOT}/merged/{args.trial_name}_merged/'
        os.makedirs(outpath, exist_ok=True)

        merged_all = []
        gt_trajectories = {}
        users = synthetic_user_ids(f"{args.generated_dir}/{args.trial_name}")

        for user in users:
            print(f"\nUSER{user}\n")

            args.id = user # Override whatever ID gets passed in
            all, anchor_positions, T, body_opti_tum_traj = post_process(args)
            merged_all = merged_all + all
            gt_trajectories[user] = body_opti_tum_traj
            # Record anchors once
            if user==users[0]:
                if args.anchor_prior_error is None: # Then file gtsam uses and ground truth are identical
                    json.dump(anchor_positions, open(outpath+"/anchors.json", 'w'), cls=NumpyEncoder, indent=1)
                    json.dump(anchor_positions, open(outpath+"/anchors_gt.json", 'w'), cls=NumpyEncoder, indent=1)
                elif args.anchor_prior_error > 0: # If we add synthetic error to the priors GTSAM uses.
                    print()
                    # Positive anchor_prior_error means you preturb the existing location by that amount for GTSAM
                    json.dump(anchor_positions, open(outpath+"/anchors_gt.json", 'w'), cls=NumpyEncoder, indent=1)
                    for a in anchor_positions:
                        # Generate a random error vector
                        # Random direction
                        R = args.anchor_prior_error
                        v = np.random.normal(size=3)
                        v /= np.linalg.norm(v)
                        # Radius distributed uniformly in volume
                        r = R * np.random.random() ** (1/3)
                        perturbation = r * v
                        a["position"] += perturbation
                        print(f" Shifted {a['ID']} by {perturbation}")
                    json.dump(anchor_positions, open(outpath+"/anchors.json", 'w'), cls=NumpyEncoder, indent=1) 
                    # Now gtsam will use this errored copy for prior
                elif args.anchor_prior_error < 0:
                    # Negative anchor_prior_error indicates putting anchors at origin for GTSAM
                    json.dump(anchor_positions, open(outpath+"/anchors_gt.json", 'w'), cls=NumpyEncoder, indent=1)
                    for a in anchor_positions:
                        a["position"] = [0,0,0]
                        print(f" Set {a['ID']} to origin")
                    json.dump(anchor_positions, open(outpath+"/anchors.json", 'w'), cls=NumpyEncoder, indent=1) 

            # Record transforms
            json.dump(vars(T), open(f'{outpath}/transforms{user}.json', 'w'), cls=NumpyEncoder, indent=1)
            
        
        # Mirror UWB ranges. This is something the Beluga firmware could report, but doesn't.
        # It only logs range on the report, not on the final, this means the responder compute the range, but never logs it.
        # This will effectively double our ranges.
        mirrored_uwb = []
        for j in merged_all:
            if j["type"] == "uwb":
                j_ = copy.deepcopy(j)
                temp_src = j["src"]
                j_["src"] = j["id"]
                j_["id"] = temp_src
                mirrored_uwb.append(j_)
        merged_all += mirrored_uwb 

        merged_all = sorted(merged_all, key=lambda x: x["t"])

        ### Flock Gossip
        # Synthesize direct ranges for starting to test gossip.
        # synth_anchor_selfloc_ranges = range_synthesizer4(merged_all, anchor_positions, gt_trajectories, T, std=0.2)          
        # merged_all += synth_anchor_selfloc_ranges
        # Append closest SLAM pose to each range.
        merged_all = embed_poses_in_ranges(merged_all, anchor_positions, gt_trajectories, mobile_nodes=users)

        # Adjust for ?anchor coordinate frame error? on 1 and 5
        # In multi2, multi3, and opti_multi1
        anchor_err_trial = False
        for t in ["multi2", "multi3", "opti_multi1"]: 
            if t in args.trial_name: 
                anchor_err_trial = True
                break
        if anchor_err_trial:
            # Remove constant positive offset off nodes
            for u in merged_all:
                if u["type"] == "uwb" and u["id"] == 1: u["range"] -= 0.16
                if u["type"] == "uwb" and u["id"] == 5: u["range"] -= 0.17

        merged_all = sorted(merged_all, key=lambda x: x["t"])
        json.dump(merged_all, open(outpath+"/all.json", 'w'), cls=NumpyEncoder, indent=1)

        # UWB error analysis
        if args.check_uwb_error:
            # error_analysis needs Decawave diagnostics (firstpathamp*, rxpreamcount, maxgrowthcir) that synthetic ranges don't have
            print("--check_uwb_error is not supported for synthetic data, skipping")


    else:
        post_process(args)