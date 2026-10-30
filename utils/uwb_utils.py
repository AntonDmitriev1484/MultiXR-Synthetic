import numpy as np
from utils.math_utils import *
import matplotlib.pyplot as plt
import json
import copy

# Body (IMU) trajectory in world frame
# Default UWB frequency is 5

# def range_synthesizer(START, END, body_traj, T, outpath, f_uwb=5/4.0, std=0.2):
#     anchors = {
#             1: [2, 0.5, 3],
#             3: [-6, 2, 2],
#             4: [-3, -3, 1],
#             5: [0, -4, 2]
#     }
#     # anchors = {
#     #         1: [2, 0.5, 3],
#     #         3: [-6, 2, 2],
#     #         4: [-3, -3, 1],
#     #         5: [0, -4, 2],
#     #         6: [-2, 2, 1.5]
#     # }

#     write_anchors = []
#     synth_ranges = []

#     # First transform body_traj to uwb_rx_traj

#     dt_uwb = 1/f_uwb
#     timestamps = np.arange(START, END, dt_uwb)

#     body_traj = np.array(body_traj)

#     for anchor_id, position in anchors.items():
#         N_ranges = timestamps.shape[0]
        
#         write_anchors.append({"ID":anchor_id, "position":position})

#         for i in range(0, timestamps.shape[0]):
#             t = timestamps[i]
            
#             vicon_timestamps = body_traj[:,0]
#             tdiffs = np.abs(vicon_timestamps - t)
#             idx = np.argmin(tdiffs) # Get closest vicon pose to this timestamp

#             T_world_to_body_tum = body_traj[idx]
#             T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
#             T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
#             T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

#             ## NOTE: Something is getting inverted here, either in pose reporting or in range calculation

#             dest_position = position
#             source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
#             # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

#             dist = np.linalg.norm(dest_position -  source_position)
#             synth_range =  np.random.normal(loc=dist, scale=std)

#             j = {
#                 "t":t,
#                 "type": "synth_uwb",
#                 "tag": "synth_for_user",
#                 "src": 2,
#                 "id": anchor_id,
#                 "pose": T_world_to_decawave, 
#                 "range": synth_range
#             }
#             synth_ranges.append(j)
        
#     out_anchors = open(f'{outpath}/anchors.json', 'w')
#     class NumpyEncoder(json.JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             if hasattr(obj, '__dict__'):
#                 return vars(obj)
#             return super().default(obj)
        
#     print(write_anchors)
#     json.dump(write_anchors, out_anchors, cls=NumpyEncoder, indent=1)

#     return synth_ranges



# Returns evenly spaced ranges
def range_synthesizer2(START, END, body_traj, T, outpath, f_uwb=5, std=0.01):
    anchors = {
            1: [2, 0.5, 3],
            3: [-6, 2, 2],
            2: [-3, -3, 1],
            5: [0, -4, 2]
    }
    # anchors = {
    #         1: [2, 0.5, 3],
    #         3: [-6, 2, 2],
    #         4: [-3, -3, 1],
    #         5: [0, -4, 2],
    #         6: [-2, 2, 1.5]
    # }

    write_anchors = []
    synth_ranges = []

    # First transform body_traj to uwb_rx_traj

    dt_uwb = 1/f_uwb
    timestamps = np.arange(START, END, dt_uwb)

    body_traj = np.array(body_traj)

    for n, (anchor_id, position) in enumerate(anchors.items()):
        N_ranges = timestamps.shape[0]

        shift = (n/len(anchors.items()))
        print(f"{anchor_id} shift = {shift}")
        this_timestamps = timestamps + shift # Each anchors timestamps are shifted over by
        
        write_anchors.append({"ID":anchor_id, "position":position})

        for i in range(0, this_timestamps.shape[0]):
            t = this_timestamps[i]
            
            vicon_timestamps = body_traj[:,0]
            tdiffs = np.abs(vicon_timestamps - t)
            idx = np.argmin(tdiffs) # Get closest vicon pose to this timestamp

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            ## NOTE: Something is getting inverted here, either in pose reporting or in range calculation

            dest_position = position
            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
            # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

            dist = np.linalg.norm(dest_position -  source_position)
            synth_range =  np.random.normal(loc=dist, scale=std)

            j = {
                "t":t,
                "type": "uwb",
                "tag": "synth_for_user",
                "src": 2,
                "id": anchor_id,
                "pose": T_world_to_decawave, 
                "range": synth_range
            }
            synth_ranges.append(j)
        
    out_anchors = open(f'{outpath}/anchors.json', 'w')
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if hasattr(obj, '__dict__'):
                return vars(obj)
            return super().default(obj)
        
    print(write_anchors)
    json.dump(write_anchors, out_anchors, cls=NumpyEncoder, indent=1)

    return synth_ranges


# Compute range error for inter-user and user-anchor ranges.
def error_analysis( id, multi_all, anchors, gt_trajectories, T, users):

    mobile_nodes = users
    # First transform body_traj to uwb_rx_traj

    range_errors = {}
    nlos_metric = {}
    timestamps = {}

    all_range_errors = []
    all_range_ts = []

    uwb = [j for j in multi_all if (j["src"] == id and j["type"] == "uwb")]

    for j in uwb:
        if j["id"] in mobile_nodes:

            other_body_traj = np.array(gt_trajectories[j["id"]-2])
            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            tx_position = T_decawave_to_world[:3,3]

            idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = other_body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            other_tx_position = T_decawave_to_world[:3,3]

            # Metrics recorded here
            synth_range = np.linalg.norm(tx_position -  other_tx_position)
            mes_range = j["range"]

            err = np.abs(synth_range - mes_range)
            range_errors.setdefault(j["id"], []).append(err)
            all_range_errors.append(err)
            all_range_ts.append(j["t"])

            timestamps.setdefault(j["id"], []).append(j["t"])

            A = 121.74

            fp_power_ = 10 * np.log10( ((j["firstpathamp1"] ** 2) + (j["firstpathamp2"]**2) + (j["firstpathamp3"]**2)) 
                                    / (j["rxpreamcount"]**2) ) - A
            rx_power_ = 10 * np.log10( j["maxgrowthcir"] * (2**17) / (j["rxpreamcount"] ** 2)) - A
            
            nlos_metric.setdefault(j["id"], []).append(rx_power_ - fp_power_)

        else:

            dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame

            # Metrics recorded here
            synth_range = np.linalg.norm(dest_position -  source_position)
            mes_range = j["range"]
            timestamps.setdefault(j["id"], []).append(j["t"])

            err = np.abs(synth_range - mes_range)
            range_errors.setdefault(j["id"], []).append(err)
            all_range_errors.append(err)
            all_range_ts.append(j["t"])

            A = 121.74

            fp_power_ = 10 * np.log10( ((j["firstpathamp1"] ** 2) + (j["firstpathamp2"]**2) + (j["firstpathamp3"]**2)) 
                                    / (j["rxpreamcount"]**2) ) - A
            rx_power_ = 10 * np.log10( j["maxgrowthcir"] * (2**17) / (j["rxpreamcount"] ** 2)) - A

            nlos_metric.setdefault(j["id"], []).append(rx_power_ - fp_power_)


    print()
    for node, err in range_errors.items():

        if node == id:
            continue

        nlos_score = nlos_metric[node]
        t = timestamps[node]
        # print(timestamps)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # print(err)
        axs[0].plot(t, err)
        axs[0].set_title(f"Range Errors from {id} to {node}")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Error (m)")
        axs[0].grid(True)


        axs[1].plot(t, nlos_score)

        axs[1].set_title("NLOS Metric")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("NLOS score")
        axs[1].grid(True)

        plt.tight_layout()

    plt.show()


# Compute range error for inter-user and user-anchor ranges.
def generate_parham_data( id, multi_all, anchors, gt_trajectories, T, trial_name, users):

    
    mobile_nodes = users
    # First transform body_traj to uwb_rx_traj

    range_errors = {}
    nlos_metric = {}
    timestamps = {}

    all_range_errors = []
    all_range_ts = []


    uwb = [j for j in multi_all if (j["src"] == id and j["type"] == "uwb")]
    annotated_uwb = []

    for j in uwb:
        if j["id"] in mobile_nodes:

            # print(len(gt_trajectories))
            print(f"{j['id']-2}=")
            other_body_traj = np.array(gt_trajectories[j["id"]-2])
            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            tx_position = T_decawave_to_world[:3,3]

            idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
            T_world_to_body_tum = other_body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)
            other_tx_position = T_decawave_to_world[:3,3]

            annotated_range = copy.deepcopy(j)
            annotated_range["optitrack_src_tx_position"] = tx_position
            annotated_range["optitrack_dst_tx_position"] = other_tx_position
            annotated_uwb.append(annotated_range)

        else:

            dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

            body_traj = np.array(gt_trajectories[j["src"]-2])

            idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))

            T_world_to_body_tum = body_traj[idx]
            T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
            
            T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
            T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

            source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame

            # Metrics recorded here
            annotated_range = copy.deepcopy(j)
            annotated_range["optitrack_src_tx_position"] = source_position
            annotated_range["optitrack_dst_tx_position"] = dest_position
            annotated_uwb.append(annotated_range)

    return annotated_uwb

# # Input trajectories are T_world_to_body we need T_body_world
def range_synthesizer3( multi_all, anchors, gt_trajectories, T, std):

    # Replace all real ranges with synthetic.

    mobile_nodes = [2,3,4]
    # First transform body_traj to uwb_rx_traj

    rs = [j["range"] for j in multi_all if j["type"] == "uwb"]
    print(f"{rs[:5]=}")



    for id in mobile_nodes: # For each user
        for j in multi_all:
            if j["src"] == id and j["type"] == "uwb": # Find all ranges that belong to that user, and convert them to synthetic.

                if j["id"] in mobile_nodes: # If we are ranging to another user

                    other_body_traj = np.array(gt_trajectories[j["id"]-2])
                    body_traj = np.array(gt_trajectories[j["src"]-2])

                    idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
                    T_world_to_body_tum = body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    tx_position = T_decawave_to_world[:3,3]

                    idx = np.argmin(np.abs(other_body_traj[:, 0] - j["t"]))
                    T_world_to_body_tum = other_body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    other_tx_position = T_decawave_to_world[:3,3]

                    # synth_range = np.random.normal(loc=np.linalg.norm(tx_position -  other_tx_position), scale=std)
                    synth_range = np.linalg.norm(tx_position -  other_tx_position)

                    j["range"] = synth_range


                else: # If we are ranging to an anchor

                    dest_position = [x['position'] for x in anchors if x['ID']== j['id']][0]

                    t = j["t"]
                    body_traj = np.array(gt_trajectories[j["src"]-2])

                    idx = np.argmin(np.abs(body_traj[:, 0] - t))

                    T_world_to_body_tum = body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    T_world_to_decawave = np.linalg.inv(T_decawave_to_world)

                    source_position = T_decawave_to_world[:3,3] # Physical antenna position in world frame
                    # source_position = np.linalg.inv(T_world_to_body)[:3,3]# body position in world frame (if just using RangeFactor)

                    # synth_range = np.random.normal(loc=np.linalg.norm(source_position -  dest_position), scale=std)
                    synth_range = np.linalg.norm(source_position -  dest_position)

                    j["range"] = synth_range

    rs = [j["range"] for j in multi_all if j["type"] == "uwb"]
    print(f"{rs[:5]=}")

    return multi_all


# For each range timestamp to a mobile node, take the most recent optitrack pose, and add a synthetic direct range to anchor 1 or 5
# This roughly models gossip giving us a range constraint to an anchor, on ranges to non-anchor nodes.

# # Input trajectories are T_world_to_body we need T_body_world
def range_synthesizer4( multi_all, anchors, gt_trajectories, T, std):

    print(anchors)

    # Replace all real ranges with synthetic.

    mobile_nodes = [2,3,4]
    stationary_nodes = [1,5]
    # First transform body_traj to uwb_rx_traj

    range_counter = 0

    synth_ranges = []

    for id in mobile_nodes: # For each user
        for j in multi_all:
            if j["src"] == id and j["type"] == "uwb": # Find all ranges that belong to that user, and convert them to synthetic.
                if j["id"] in mobile_nodes: # For all user - > user ranges
                    # Keep the original range
                    # And create an extra synthetic range to an anchor
                    
                    body_traj = np.array(gt_trajectories[j["src"]-2])

                    idx = np.argmin(np.abs(body_traj[:, 0] - j["t"]))
                    T_world_to_body_tum = body_traj[idx]
                    T_world_to_body = slam_quat_to_HTM(T_world_to_body_tum)
                    T_decawave_to_world = np.linalg.inv(T_world_to_body) @ np.linalg.inv(T.T_body_to_decawave) # compute tag decawave_to_world from body_to_world pose
                    tx_position = T_decawave_to_world[:3,3]


                    dest_position = anchors[range_counter % len(anchors)]['position']
                    dest_id = anchors[range_counter % len(anchors)]['ID']

                    synth_range = np.random.normal(loc=np.linalg.norm(tx_position -  dest_position), scale=std)
                    # synth_range = np.linalg.norm(tx_position -  dest_position)

                    j_synth = copy.copy(j)
                    j_synth["range"] = synth_range
                    j_synth["id"] = dest_id
                    j_synth["tag"] = "synth_for_anchor_selfloc"
                    synth_ranges.append(j_synth)

                    range_counter += 1

    return synth_ranges

def embed_poses_in_ranges( multi_all, anchors, gt_trajectories, mobile_nodes=[2,3,4]):

    # All aligned slam poses from each device, gathered once rather than once per range
    slam_poses = {}
    for m in multi_all:
        if m["type"] == "aligned_slam_pose": slam_poses.setdefault(m["src"], []).append(m)
    slam_ts = {src: np.array([m["t"] for m in poses]) for src, poses in slam_poses.items()} # Reduce to timestamps

    # For each range, find the closest SLAM pose, and embed that into the range.
    for mes in multi_all:
        if mes["type"] == "uwb":
            this_id = mes["src"]
            if this_id in mobile_nodes:
                idx = np.argmin(np.abs(slam_ts[this_id] - mes["t"])) # Get the closest slam pose timestamp to this UWB measurement
                mes["embedded_slam_pose"] = slam_poses[this_id][idx] # Embed it into the range.

    return multi_all