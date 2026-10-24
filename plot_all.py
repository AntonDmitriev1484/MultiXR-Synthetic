import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# post_process.py writes each user's output to post/{id}/{trial_name}_post/
ROOT = os.path.dirname(os.path.abspath(__file__))

def post_dir(id, trial_name):
    return f"{ROOT}/post/{id}/{trial_name}_post"

ROOM_HEIGHT = 2.5   # m, config.json only records the floor dimensions

def room_limits(trial_name):
    """Axis limits of the room generate.py placed this trial's trajectories in."""
    with open(f"{ROOT}/generated/{trial_name}/config.json", "r") as f:
        W, H = json.load(f)["args"]["room_dim"]
    return (0, W), (0, H), (0, ROOM_HEIGHT)

# Okabe-Ito colorblind-safe palette
COLORS = {
    "opti": "#0072B2",       # blue
    "est": "#D55E00",  # vermillion
    "recovered": "#009E73",       # bluish green
    "anchor": "#CC79A7",        # reddish purple
    "lost": "#CC79A7",     # orange
}

# COLORS = {
#     "opti": "#0072B2",       # blue
#     "est": "#D55E00",  # vermillion
#     "recovered": "#E69F00",       # bluish green
#     "anchor": "#CC79A7",        # reddish purple
#     "lost": "#E69F00",     # orange
# }

def draw_axes(ax, T, length=0.1):
    """Draw coordinate axes from transformation matrix T."""
    H = np.linalg.inv(T)
    origin = (H @ np.array([0,0,0,1]))[:3]
    x_axis = (H @ np.array([1,0,0,1]))[:3]
    y_axis = (H @ np.array([0,1,0,1]))[:3]
    z_axis = (H @ np.array([0,0,1,1]))[:3]

    ax.quiver(*origin, *(x_axis-origin) * length, color='r')
    ax.quiver(*origin, *(y_axis-origin) * length, color='g')
    ax.quiver(*origin, *(z_axis-origin) * length, color='b')

import json
import numpy as np
import matplotlib.pyplot as plt

# -1 plot trajectory but not coordinate frames
# -2 don't plot trajectory

DONT_PLOT=-2

def plot_trial(
    id,
    trial_name,
    slam_stride=-2,
    opti_stride=-2,
    est_stride=-2,
    show_live_slam=False,
    run_config="",
    label_text="",
    anchors=False,
    transforms_json=None,
    calibration=False,
    paths=None,
    show=True,
    ax=None
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------
    # File paths
    # ------------------------------

    all_json_path = {}
    anchors_path = {}
    transforms_path = {}

    all_json_path = f"{post_dir(id, trial_name)}/all.json"

    anchors_path = f"{post_dir(id, trial_name)}/anchors.json"

    transforms_path = f"{post_dir(id, trial_name)}/transforms.json"

    live_slam_path = all_json_path
    post_slam_path = all_json_path
    opti_path = all_json_path
    slam_path = all_json_path
    est_path = ""
    anchor_optimization_path = ""
    final_anchor_estimate_path = ""

    # Preserve function argument unless overridden
    # est_path already came from the function parameter
    if paths is not None:
        live_slam_path = paths.live_slam_path
        post_slam_path = paths.post_slam_path
        opti_path = paths.opti_path
        slam_path = paths.slam_path

        if hasattr(paths, "est_path") and paths.est_path is not None:
            est_path = paths.est_path
        
        if hasattr(paths, "anchor_optimization_path") and paths.anchor_optimization_path is not None:
            anchor_optimization_path = paths.anchor_optimization_path
        
        if hasattr(paths, "final_anchor_estimate_path") and paths.final_anchor_estimate_path is not None:
            final_anchor_estimate_path = paths.final_anchor_estimate_path

    # ------------------------------
    # Load required data
    # ------------------------------
    with open(all_json_path, "r") as f:
        all_data = json.load(f)

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    T_imu_to_body = np.array(transforms["T_imu_to_body"])

    # ------------------------------
    # Initialize pose containers
    # ------------------------------

    # Load in slam poses
    slam_poses = []
    lost_slam_poses = []
    with open(slam_path, "r") as f:
        for item in json.load(f):
                if (
                    item.get("type") in ("slam_pose")
                    and "T_body_world" in item
                ):
                    pose = np.array(item["T_body_world"])
                    if item.get("status") == "imu" or item.get("status") == "newmap":
                        slam_poses.append(None)
                        lost_slam_poses.append(pose)
                    else:
                        slam_poses.append(pose)
                        lost_slam_poses.append(None)

    # Load in optitrack poses
    opti_poses = []
    with open(opti_path, "r") as f:
        for item in json.load(f):
            if (
                item.get("type") == "opti_pose"
                and "T_body_world" in item
            ):
                opti_poses.append(np.array(item["T_body_world"]))


    aligned_slam_poses = []
    lost_aligned_slam_poses = []

    tracking = True
    if show_live_slam:
        with open(live_slam_path, "r") as f:
            for item in json.load(f):
                if ( item.get("type") == "aligned_live_slam_pose"):
                    pose = np.array(item["T_body_world"])
                    if item.get("status") == "tracking":
                        aligned_slam_poses.append(pose)
                        if not tracking: lost_aligned_slam_poses.append(pose) # Ensure continuity between red lost line and the visual recovered trajectory
                        lost_aligned_slam_poses.append(None)
                    else:
                        aligned_slam_poses.append(None)
                        lost_aligned_slam_poses.append(pose)
                        tracking = False
    else:
        with open(post_slam_path, "r") as f:
            for item in json.load(f):
                if ( item.get("type") == "aligned_slam_pose"):
                    pose = np.array(item["T_body_world"])
                    if item.get("status") == "tracking":
                        aligned_slam_poses.append(pose)
                        if not tracking: lost_aligned_slam_poses.append(pose) # Ensure continuity between red lost line and the visual recovered trajectory
                        lost_aligned_slam_poses.append(None)
                    else:
                        aligned_slam_poses.append(None)
                        lost_aligned_slam_poses.append(pose)
                        tracking = False

    # Aligned SLAM will automatically overrule regular SLAM in plotting
    if len(aligned_slam_poses) > 0:
        slam_poses = aligned_slam_poses
        lost_slam_poses = lost_aligned_slam_poses

    # ------------------------------
    # Create figure if needed
    # ------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # ------------------------------
    # Plot SLAM
    # ------------------------------
    if len(slam_poses) > 0 and slam_stride != DONT_PLOT:
        positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in slam_poses
        ])

        ax.plot(
            positions_world[:, 0],
            positions_world[:, 1],
            positions_world[:, 2],
            label="SLAM",
            color="green"
        )

        lost_positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in lost_slam_poses
        ])

        ax.plot(
            lost_positions_world[:, 0],
            lost_positions_world[:, 1],
            lost_positions_world[:, 2],
            label="LOST SLAM",
            color="red"
        )

        valid_slam = [p for p in slam_poses if p is not None]

        if slam_stride > 0:
            for i in range(0, len(valid_slam), slam_stride):
                draw_axes(ax, valid_slam[i], length=0.4)

    # ------------------------------
    # Plot OptiTrack
    # ------------------------------
    if len(opti_poses) > 0 and opti_stride != DONT_PLOT:
        positions_world = np.array([
            np.linalg.inv(p)[:3, 3]
            for p in opti_poses
        ])

        ax.plot(
            positions_world[:, 0],
            positions_world[:, 1],
            positions_world[:, 2],
            label="Optitrack",
            color=COLORS["opti"]
        )

        if opti_stride > 0:
            for i in range(0, len(opti_poses), opti_stride):
                draw_axes(ax, opti_poses[i], length=0.4)

                

    # ------------------------------
    # Plot estimates (optional)
    # ------------------------------
    if est_path is not None:
        try:
            with open(est_path, "r") as f:
                anchor_traj_data = json.load(f)

            anchor_poses = []

            for item in anchor_traj_data:
                if (
                    item.get("type") == "est_pose"
                    and "T_body_world" in item
                ):
                    anchor_poses.append(
                        np.array(item["T_body_world"])
                    )

            if len(anchor_poses) > 0 and est_stride != DONT_PLOT:

                positions_world = np.array([
                    np.linalg.inv(p)[:3, 3]
                    for p in anchor_poses
                ])

                ax.plot(
                    positions_world[:, 0],
                    positions_world[:, 1],
                    positions_world[:, 2],
                    label=label_text,
                    color=COLORS["est"]
                )

                if est_stride > 0:
                    for i in range(0, len(anchor_poses), est_stride):

                        draw_axes(ax, anchor_poses[i], length=0.4)

                        # Add pose index text
                        pos = np.linalg.inv(anchor_poses[i])[:3, 3]

                        # ax.text(
                        #     pos[0],
                        #     pos[1],
                        #     pos[2],
                        #     str(i),
                        #     fontsize=8
                        # )
        except Exception as e:
            print(f"Failed to load estimates: {e}")

    # ------------------------------
    # Anchors
    # ------------------------------
    if anchors:
        try:
            with open(anchors_path, "r") as f:
                anchor_data = json.load(f)

            for d in anchor_data:
                pos = d["position"]

                ax.scatter(pos[0], pos[1], pos[2], color=COLORS["recovered"])
                ax.text(pos[0], pos[1], pos[2], d["ID"])

        except Exception as e:
            print(f"Anchor plotting failed: {e}")


    # ------------------------------
    # Plot estimated anchor trajectory
    # ------------------------------
    if anchor_optimization_path is not None:
        try:
            with open(anchor_optimization_path, "r") as f:
                anchor_traj_data = json.load(f)

            anchor_poses = []

            for item in anchor_traj_data:
                if (
                    item.get("type") == "est_pose"
                    and "T_body_world" in item
                ):
                    anchor_poses.append(
                        np.array(item["T_body_world"])
                    )
            # TODO: Need to invert here.
            
            if len(anchor_poses) > 0 and est_stride != DONT_PLOT:

                positions_world = np.array([
                    np.linalg.inv(p)[:3, 3]
                    for p in anchor_poses
                ])

                ax.plot(
                    positions_world[:, 0],
                    positions_world[:, 1],
                    positions_world[:, 2],
                    label=label_text,
                    color=COLORS["est"]
                )

                if est_stride > 0:
                    for i in range(0, len(anchor_poses), est_stride):

                        draw_axes(ax, anchor_poses[i], length=0.4)

                        # Add pose index text
                        pos = np.linalg.inv(anchor_poses[i])[:3, 3]

        except Exception as e:
            print(f"Failed to load estimates: {e}")

    # ------------------------------
    # Plot final estimated positions
    # ------------------------------
    if not (final_anchor_estimate_path == ""):
        try:
            with open(final_anchor_estimate_path, "r") as f:
                anchor_data = json.load(f)
            for d in anchor_data:
                pos = d["position"]
                ax.scatter(pos[0], pos[1], pos[2], color=COLORS["est"])
                ax.text(pos[0], pos[1], pos[2], d["ID"])
        except Exception as e:
            print(f"Anchor plotting failed: {e}")

    # ------------------------------
    # Plot formatting
    # ------------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    xlim, ylim, zlim = room_limits(trial_name)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_title(f"NUC{id} {trial_name}")
    # Filter ablation study
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Enabled, UWB Filter Disabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Enabled, UWB Filter Enabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Disabled, UWB Filter Enabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Disabled, UWB Filter Disabled")

    # Example trials
    # ax.set_title(f"Following Another User")
    # ax.set_title(f"Tripping")
    
    ax.view_init(elev=45, azim=45)
    ax.legend()

    if show:
        plt.show()

    return fig, ax



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("id", type=int)
    parser.add_argument("trial_name")

    parser.add_argument(
        "--slam",
        type=int,
        nargs="?",
        const=-1,
        default=-2,
        help="Optional stride. If passed without value, uses 1. -1 disables."
    )
    parser.add_argument(
        "--opti",
        type=int,
        nargs="?",
        const=-1,
        default=-2,
        help="Optional stride. If passed without value, uses 1. -1 disables."
    )
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--transforms_json", default=None)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--live_slam", action="store_true")

    args = parser.parse_args()

    # -1 means don't plot coordinate axes on trajectory

    plot_trial(
        id=args.id,
        trial_name=args.trial_name,
        slam_stride=args.slam,
        opti_stride=args.opti,
        anchors=args.anchors,
        show_live_slam=args.live_slam,
        transforms_json=args.transforms_json,
        calibration=args.calibration,
    )


if __name__ == "__main__":
    main()

def plot_trial_paper(
    id,
    trial_name,
    slam_stride=-2,
    opti_stride=-2,
    est_stride=-2,
    show_live_slam=False,
    run_config="",
    label_text="",
    anchors=False,
    transforms_json=None,
    calibration=False,
    paths=None,
    show=True,
    ax=None
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------
    # File paths
    # ------------------------------

    all_json_path = {}
    anchors_path = {}
    transforms_path = {}

    all_json_path = f"{post_dir(id, trial_name)}/all.json"

    anchors_path = f"{post_dir(id, trial_name)}/anchors.json"

    transforms_path = f"{post_dir(id, trial_name)}/transforms.json"

        # ------------------------------
    # Create figure if needed
    # ------------------------------
    if ax is None:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection="3d")

    live_slam_path = all_json_path
    post_slam_path = all_json_path
    opti_path = all_json_path
    slam_path = all_json_path
    est_path = ""

    # Preserve function argument unless overridden
    # est_path already came from the function parameter
    if paths is not None:
        live_slam_path = paths.live_slam_path
        post_slam_path = paths.post_slam_path
        opti_path = paths.opti_path
        slam_path = paths.slam_path

        if hasattr(paths, "est_path") and paths.est_path is not None:
            est_path = paths.est_path

    # ------------------------------
    # Load required data
    # ------------------------------
    with open(all_json_path, "r") as f:
        all_data = json.load(f)

    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    T_imu_to_body = np.array(transforms["T_imu_to_body"])

    # ------------------------------
    # Initialize pose containers
    # ------------------------------

    # Load in slam poses
    slam_poses = []
    lost_slam_poses = []
    with open(slam_path, "r") as f:
        for item in json.load(f):
                if (
                    item.get("type") in ("slam_pose")
                    and "T_body_world" in item
                ):
                    pose = np.array(item["T_body_world"])
                    if item.get("status") == "imu" or item.get("status") == "newmap":
                        slam_poses.append(None)
                        lost_slam_poses.append(pose)
                    else:
                        slam_poses.append(pose)
                        lost_slam_poses.append(None)

    aligned_slam_poses = []
    lost_aligned_slam_poses = []

    failure_starts = []
    failure_ends = []

    tracking = True
    prev_tracking = True

    with open(post_slam_path, "r") as f:
        for item in json.load(f):

            if item.get("type") != "aligned_live_slam_pose":
                continue

            pose = np.array(item["T_body_world"])
            timestamp = item["t"]

            current_tracking = item.get("status") == "tracking"

            # tracking -> lost
            if prev_tracking and not current_tracking:
                failure_starts.append(timestamp)

            # lost -> tracking
            if not prev_tracking and current_tracking:
                failure_ends.append(timestamp)

            if current_tracking:
                aligned_slam_poses.append(pose)

                if not tracking:
                    lost_aligned_slam_poses.append(pose)

                lost_aligned_slam_poses.append(None)

            else:
                aligned_slam_poses.append(None)
                lost_aligned_slam_poses.append(pose)

            tracking = current_tracking
            prev_tracking = current_tracking

    # Aligned SLAM will automatically overrule regular SLAM in plotting
    if len(aligned_slam_poses) > 0:
        slam_poses = aligned_slam_poses
        lost_slam_poses = lost_aligned_slam_poses

    
    # Load in optitrack poses
    opti_poses = []
    opti_timestamps = []

    def nearest_opti_index(target_time, timestamps):
        timestamps = np.asarray(timestamps)
        return np.argmin(np.abs(timestamps - target_time))

    with open(opti_path, "r") as f:
        for item in json.load(f):

            if (
                item.get("type") == "opti_pose"
                and "T_body_world" in item
            ):
                opti_poses.append(np.array(item["T_body_world"]))
                opti_timestamps.append(item["t"])

    for t in failure_starts:

        idx = nearest_opti_index(t, opti_timestamps)

        pos = np.linalg.inv(opti_poses[idx])[:3, 3]

        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            marker='o',
            s=100,
            linewidths=1,
            facecolors=COLORS["lost"],
            edgecolors='black',
            label='Tracking Lost'
        )

    for t in failure_ends:

        idx = nearest_opti_index(t, opti_timestamps)

        pos = np.linalg.inv(opti_poses[idx])[:3, 3]

        ax.scatter(
            pos[0],
            pos[1],
            pos[2],
            marker='o',
            s=100,
            facecolors=COLORS["recovered"],
            edgecolors='black',
            linewidths=1,
            label='Tracking Recovered'
        )


    # ------------------------------
    # Plot SLAM
    # ------------------------------
    if len(slam_poses) > 0 and slam_stride != DONT_PLOT:
        positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in slam_poses
        ])

        ax.plot(
            positions_world[:, 0],
            positions_world[:, 1],
            positions_world[:, 2],
            label="SLAM",
            color=COLORS["est"]
        )

        lost_positions_world = np.array([
            np.linalg.inv(p)[:3, 3] if p is not None
            else [np.nan, np.nan, np.nan]
            for p in lost_slam_poses
        ])

        ax.plot(
            lost_positions_world[:, 0],
            lost_positions_world[:, 1],
            lost_positions_world[:, 2],
            label="Lost SLAM",
            color="#CC79A7"
        )

        valid_slam = [p for p in slam_poses if p is not None]

        if slam_stride > 0:
            for i in range(0, len(valid_slam), slam_stride):
                draw_axes(ax, valid_slam[i], length=0.4)
                

    # ------------------------------
    # Plot estimates (optional)
    # ------------------------------
    if est_path is not None:
        try:
            with open(est_path, "r") as f:
                est_data = json.load(f)

            est_poses = []

            for item in est_data:
                if (
                    item.get("type") == "est_pose"
                    and "T_body_world" in item
                ):
                    est_poses.append(
                        np.array(item["T_body_world"])
                    )

            if len(est_poses) > 0 and est_stride != DONT_PLOT:

                positions_world = np.array([
                    np.linalg.inv(p)[:3, 3]
                    for p in est_poses
                ])

                ax.plot(
                    positions_world[:, 0],
                    positions_world[:, 1],
                    positions_world[:, 2],
                    label=label_text,
                    color=COLORS["est"]
                )

                if est_stride > 0:
                    for i in range(0, len(est_poses), est_stride):

                        draw_axes(ax, est_poses[i], length=0.4)

                        # Add pose index text
                        pos = np.linalg.inv(est_poses[i])[:3, 3]

                        # ax.text(
                        #     pos[0],
                        #     pos[1],
                        #     pos[2],
                        #     str(i),
                        #     fontsize=8
                        # )
        except Exception as e:
            print(f"Failed to load estimates: {e}")

    
    # ------------------------------
    # Plot OptiTrack
    # ------------------------------
    positions_world = np.array([
        np.linalg.inv(p)[:3, 3]
        for p in opti_poses
    ])

    ax.plot(
        positions_world[:, 0],
        positions_world[:, 1],
        positions_world[:, 2],
        label="Optitrack",
        color=COLORS["opti"]
    )

    if opti_stride > 0:
        for i in range(0, len(opti_poses), opti_stride):
            draw_axes(ax, opti_poses[i], length=0.4)


    # ------------------------------
    # Anchors
    # ------------------------------
    if anchors:
        try:
            with open(anchors_path, "r") as f:
                anchor_data = json.load(f)

            for d in anchor_data:
                pos = d["position"]

                ax.scatter(pos[0], pos[1], pos[2], color=COLORS["est"])
                ax.text(pos[0], pos[1], pos[2], d["ID"])

        except Exception as e:
            print(f"Anchor plotting failed: {e}")

    # ------------------------------
    # Plot formatting
    # ------------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    xlim, ylim, zlim = room_limits(trial_name)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    # ax.set_title(f"NUC{id} {trial_name}")
    # Filter ablation study
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Enabled, UWB Filter Disabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Enabled, UWB Filter Enabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Disabled, UWB Filter Enabled")
    # ax.set_title(f"User Trajectory\nLow-Pass Filter Disabled, UWB Filter Disabled")

    # Example trials
    ax.set_title(f"Following Another User")
    # ax.set_title(f"Tripping")
    
    ax.view_init(elev=45, azim=45)
    ax.legend()

    if show:
        plt.show()

    return fig, ax
