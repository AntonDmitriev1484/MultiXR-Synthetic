"""
Interactive 3D plot of one user's trajectory from a generated dataset.

Example:
  python3 plot_single.py mobile_5_static_4 0 --anchors --stride 30
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
ROOM_HEIGHT = 2.5   # m, height of the drawn room box
ARROW_LENGTH = 0.3  # m


def resolve_dataset(name):
    """Datasets are looked up in generated/ unless an existing path is given."""
    return name if os.path.isdir(name) else os.path.join(GENERATED_DIR, name)


def load_trajectory(dataset_dir, uid):
    """Returns timestamps (N,) and T_body_to_world (N, 4, 4) for a user."""
    with open(os.path.join(dataset_dir, str(uid), "vive_synth.json")) as f:
        poses = json.load(f)
    t = np.array([p["t"] for p in poses])
    T_body_to_world = np.linalg.inv(np.array([p["T_body_world"] for p in poses]))
    return t, T_body_to_world


def load_config(dataset_dir):
    with open(os.path.join(dataset_dir, "config.json")) as f:
        return json.load(f)


def user_ids(dataset_dir):
    return sorted(int(d) for d in os.listdir(dataset_dir) if d.isdigit())


def make_axes():
    fig = plt.figure(figsize=(9, 8))
    return fig, fig.add_subplot(projection="3d")


def draw_room(ax, dataset_dir):
    W, H = load_config(dataset_dir)["args"]["room_dim"]
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], 0.0, color="k", linewidth=1.5)
    ax.plot(corners[:, 0], corners[:, 1], ROOM_HEIGHT, color="k", linewidth=0.5, alpha=0.3)
    for x, y in corners[:4]:
        ax.plot([x, x], [y, y], [0.0, ROOM_HEIGHT], color="k", linewidth=0.5, alpha=0.3)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, ROOM_HEIGHT)
    ax.set_box_aspect((W, H, ROOM_HEIGHT))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")


def draw_anchors(ax, dataset_dir):
    with open(os.path.join(dataset_dir, "anchors.json")) as f:
        anchors = json.load(f)
    if not anchors:
        return
    p = np.array([a["position"] for a in anchors])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="^", s=80, c="k", label="UWB anchors", depthshade=False)
    for a in anchors:
        ax.text(*a["position"], f" {a['ID']}")


def draw_trajectory(ax, T_body_to_world, stride, color, label):
    p = T_body_to_world[:, :3, 3]
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=1.2, label=label)
    ax.scatter(*p[0], color=color, marker="o", s=40, depthshade=False)
    ax.scatter(*p[-1], color=color, marker="s", s=40, depthshade=False)
    if stride > 0:
        # Head forward (body x) axis in the world frame
        fwd = T_body_to_world[::stride, :3, 0]
        q = p[::stride]
        ax.quiver(q[:, 0], q[:, 1], q[:, 2], fwd[:, 0], fwd[:, 1], fwd[:, 2], color=color,
                  length=ARROW_LENGTH, linewidth=0.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset name in generated/, e.g. mobile_5_static_4")
    parser.add_argument("id", type=int, help="user id")
    parser.add_argument("--anchors", action="store_true", help="show UWB anchors")
    parser.add_argument("--stride", type=int, default=30, help="draw a heading arrow every N poses (0 = none)")
    parser.add_argument("--save", help="write the figure to this path instead of showing it")
    args = parser.parse_args()

    dataset_dir = resolve_dataset(args.dataset)
    t, T_body_to_world = load_trajectory(dataset_dir, args.id)
    fig, ax = make_axes()
    draw_room(ax, dataset_dir)
    draw_trajectory(ax, T_body_to_world, args.stride, "C0", f"user {args.id}")
    if args.anchors:
        draw_anchors(ax, dataset_dir)
    ax.set_title(f"{os.path.basename(os.path.normpath(dataset_dir))} - user {args.id} "
                 f"({t[-1] - t[0]:.1f} s, o start, □ end)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
