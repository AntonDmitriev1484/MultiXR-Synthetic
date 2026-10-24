"""
Interactive 3D plot of every user trajectory in a generated dataset, superimposed.

Example:
  python3 plot_all.py mobile_5_static_4 --stride 60 --anchors
"""

import argparse
import os

import matplotlib.pyplot as plt

from gen_plot_single import (draw_anchors, draw_room, draw_trajectory, load_trajectory, make_axes,
                         resolve_dataset, user_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset name in generated/, e.g. mobile_5_static_4")
    parser.add_argument("--anchors", action="store_true", help="show UWB anchors")
    parser.add_argument("--stride", type=int, default=30, help="draw a heading arrow every N poses (0 = none)")
    parser.add_argument("--save", help="write the figure to this path instead of showing it")
    args = parser.parse_args()

    dataset_dir = resolve_dataset(args.dataset)
    ids = user_ids(dataset_dir)
    cmap = plt.get_cmap("tab10" if len(ids) <= 10 else "tab20")
    fig, ax = make_axes()
    draw_room(ax, dataset_dir)
    for i, uid in enumerate(ids):
        _, T_body_to_world = load_trajectory(dataset_dir, uid)
        draw_trajectory(ax, T_body_to_world, args.stride, cmap(i % cmap.N), f"user {uid}")
    if args.anchors:
        draw_anchors(ax, dataset_dir)
    ax.set_title(f"{os.path.basename(os.path.normpath(dataset_dir))} (o start, □ end)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
