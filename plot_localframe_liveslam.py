import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from plot_all import post_dir, room_limits

# Okabe-Ito colorblind-safe palette
COLORS = {
    "opti": "#0072B2",       # blue
    "est": "#D55E00",  # vermillion
    "recovered": "#009E73",       # bluish green
    "anchor": "#CC79A7",        # reddish purple
    "lost": "#CC79A7",     # orange
}

# Root of the gtsam_test post-integration results. The per-trial file lives at
# <root>/<id>/<trial_name>/localframe_live_slam.json
POST_INTEGRATION_ROOT = "/home/antond2/Desktop/Research/gtsam_test/results/out/multi"


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


# -1 plot trajectory but not coordinate frames
# -2 don't plot trajectory

DONT_PLOT=-2


def post_integration_path(id, trial_name, root=POST_INTEGRATION_ROOT):
    """Resolve --post_integration into a json file path.

    `root` may be the results root, or a path to a json file (used verbatim).
    """
    if root.endswith(".json"):
        return root
    return f"{root}/{id}/{trial_name}/localframe_live_slam.json"


def load_localframe_slam(path):
    """Read localframe_live_slam_pose entries into tracking / lost pose lists.

    Both lists are the same length as the pose stream, with None wherever the
    other list holds the pose, so plotting them yields two gapped trajectories.
    """
    slam_poses = []
    lost_slam_poses = []

    tracking = True
    with open(path, "r") as f:
        for item in json.load(f):
            if (
                item.get("type") == "localframe_live_slam_pose"
                and "T_body_world" in item
            ):
                pose = np.array(item["T_body_world"])
                if item.get("status") == "tracking":
                    slam_poses.append(pose)
                    if not tracking: lost_slam_poses.append(pose) # Ensure continuity between red lost line and the visual recovered trajectory
                    lost_slam_poses.append(None)
                    tracking = True
                else:
                    slam_poses.append(None)
                    lost_slam_poses.append(pose)
                    tracking = False

    return slam_poses, lost_slam_poses


def positions_from_poses(poses):
    """World-frame positions, with NaN gaps where a pose is None."""
    return np.array([
        np.linalg.inv(p)[:3, 3] if p is not None
        else [np.nan, np.nan, np.nan]
        for p in poses
    ])


def plot_slam_traj(
    ax,
    slam_poses,
    lost_slam_poses,
    stride,
    label,
    color,
    lost_color,
    linestyle="-",
):
    """Plot one tracking/lost trajectory pair, optionally with body axes."""
    positions_world = positions_from_poses(slam_poses)
    ax.plot(
        positions_world[:, 0],
        positions_world[:, 1],
        positions_world[:, 2],
        label=label,
        color=color,
        linestyle=linestyle,
    )

    lost_positions_world = positions_from_poses(lost_slam_poses)
    ax.plot(
        lost_positions_world[:, 0],
        lost_positions_world[:, 1],
        lost_positions_world[:, 2],
        label=f"LOST {label}",
        color=lost_color,
        linestyle=linestyle,
    )

    if stride > 0:
        valid_slam = [p for p in slam_poses if p is not None]
        for i in range(0, len(valid_slam), stride):
            draw_axes(ax, valid_slam[i], length=0.4)


def plot_trial(
    id,
    trial_name,
    slam_stride=-2,
    paths=None,
    fixed_lims=False,
    show=True,
    ax=None,
    post_integration=None,
    stride=None,
):
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    # ------------------------------
    # File paths
    # ------------------------------

    all_json_path = f"{post_dir(id, trial_name)}/all.json"

    live_slam_path = all_json_path

    # Preserve function argument unless overridden
    if paths is not None:
        live_slam_path = paths.live_slam_path
        if post_integration is None:
            post_integration = getattr(paths, "post_integration_path", None)

    # Coordinate frames are drawn every `stride` poses. Fall back to a positive
    # --slam value, which has always doubled as a stride.
    frame_stride = stride if stride else (slam_stride if slam_stride > 0 else 0)

    # ------------------------------
    # Load in local frame live slam poses
    # ------------------------------

    slam_poses, lost_slam_poses = load_localframe_slam(live_slam_path)

    # ------------------------------
    # Create figure if needed
    # ------------------------------
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # ------------------------------
    # Plot local frame live SLAM
    # ------------------------------
    if len(slam_poses) > 0 and slam_stride != DONT_PLOT:
        plot_slam_traj(
            ax,
            slam_poses,
            lost_slam_poses,
            frame_stride,
            label="Local Frame Live SLAM",
            color="green",
            lost_color="red",
        )
    else:
        print("No localframe_live_slam_pose entries found")

    # ------------------------------
    # Plot post-integration result
    # ------------------------------
    title = f"NUC{id} {trial_name}\nLocal Frame Live SLAM"

    if post_integration is not None:
        pi_path = post_integration_path(id, trial_name, post_integration)
        pi_poses, pi_lost_poses = load_localframe_slam(pi_path)

        if len(pi_poses) > 0:
            plot_slam_traj(
                ax,
                pi_poses,
                pi_lost_poses,
                frame_stride,
                label="Post Integration",
                color=COLORS["opti"],
                lost_color=COLORS["anchor"],
                linestyle="--",
            )
            title += " + Post Integration"
        else:
            print(f"No localframe_live_slam_pose entries found in {pi_path}")

    # ------------------------------
    # Plot formatting
    # ------------------------------
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # The local frame is the SLAM world frame, whose origin is the first head pose rather than the room
    # corner. A room-sized box centered on that origin contains any trajectory that stays in the room.
    if fixed_lims:
        xlim, ylim, zlim = room_limits(trial_name)
        ax.set_xlim(-xlim[1], xlim[1])
        ax.set_ylim(-ylim[1], ylim[1])
        ax.set_zlim(-zlim[1], zlim[1])

    ax.set_title(title)

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
        default=-1,
        help="Optional stride. If passed without value, plots trajectory only. -2 disables."
    )
    parser.add_argument(
        "--fixed_lims",
        action="store_true",
        help="Use a room-sized box (from generated/<trial_name>/config.json) centered on the SLAM origin."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Draw a coordinate frame on every Xth tracking pose, for both the "
            "live SLAM and post-integration trajectories. Overrides --slam's stride."
        ),
    )
    parser.add_argument(
        "--post_integration",
        nargs="?",
        const=POST_INTEGRATION_ROOT,
        default=None,
        help=(
            "Also plot the post-integration trajectory from "
            f"{POST_INTEGRATION_ROOT}/<id>/<trial_name>/localframe_live_slam.json. "
            "Optionally pass a different results root, or a json file directly."
        ),
    )

    args = parser.parse_args()

    # -1 means don't plot coordinate axes on trajectory

    plot_trial(
        id=args.id,
        trial_name=args.trial_name,
        slam_stride=args.slam,
        fixed_lims=args.fixed_lims,
        post_integration=args.post_integration,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
