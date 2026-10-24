import os
from pathlib import Path
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

def load_rostypes():
    add_types = {}

    # --- (Optional) Explicit Jazzy tf support ---
    jazzy_share = "/opt/ros/jazzy/share"

    def load_pkg(pkg):
        pkg_dir = Path(jazzy_share) / pkg / "msg"
        if pkg_dir.exists():
            for f in pkg_dir.glob("*.msg"):
                msg_def = f.read_text()
                msg_name = f"{pkg}/msg/{f.stem}"
                add_types.update(get_types_from_msg(msg_def, msg_name))

    # Only needed if types are missing in bag
    load_pkg("tf2_msgs")
    load_pkg("geometry_msgs")

    # --- Use Jazzy typestore ---
    typestore = get_typestore(Stores.ROS2_JAZZY)
    typestore.register(add_types)

    return typestore