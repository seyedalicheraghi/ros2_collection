"""Constants and small helpers shared across the so101_ros2 package.

Edit this file (and only this file) to adapt the package to your machine:
  • FOLLOWER_PORT / LEADER_PORT  — your USB serial paths
  • DATASET_REPO_ID / DATASET_ROOT — where recorded datasets land
  • FPS / IMAGE_W / IMAGE_H       — recording rate / image size

Changing FPS, IMAGE_W, IMAGE_H, or MOTOR_NAMES after a dataset has been
started makes new episodes incompatible with old ones — delete and restart.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Hardware ports ────────────────────────────────────────────────────────────
# `/dev/ttyACM*` numbering is assigned in plug-detect order, so it swaps
# leader/follower across reboots and replugs. Using the udev-managed
# `/dev/serial/by-id/` symlinks keys on each board's USB serial number,
# which is stable for the lifetime of the boards.
#
# Mapping for this rig (verified 2026-05-15):
#   leader   → serial 5B41531912
#   follower → serial 5B41532892
#
# If you swap boards or get new ones, list them with `ls /dev/serial/by-id/`
# and update the two paths below.
LEADER_PORT   = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41531912-if00"
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B41532892-if00"
FOLLOWER_ID   = "my_follower_arm"
LEADER_ID     = "my_leader_arm"

# ── Recording / dataset config ────────────────────────────────────────────────
DATASET_REPO_ID = "local/so101_arducam"
# Stored on the NTFS data drive (193 GB free) instead of the root partition
# (which was at 99% capacity). If the NTFS drive is unmounted (e.g. booted
# into Windows recently with hibernate on), recording will fail at write time
# — flip back to `Path.home() / "lerobot_datasets"` to use the root drive.
DATASET_ROOT    = Path("/media/ali/AE0043120042E147/lerobot_datasets")
FPS             = 30
IMAGE_H         = 480
IMAGE_W         = 640

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
# `.pos` suffix matches the keys returned by SOFollower.get_observation()
# and by SOLeader.get_action(). Used for observation.state and action.
MOTOR_NAMES = [f"{j}.pos" for j in JOINT_NAMES]

# ── Internal paths ────────────────────────────────────────────────────────────
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT    = PACKAGE_ROOT.parent
CONFIG_FILE  = PACKAGE_ROOT / "camera_config.json"
SNAPSHOT_DIR = Path.home() / "lerobot_datasets" / "snapshots"

# Module path used by dashboard.py to spawn camera publishers as subprocesses.
PUBLISHER_MODULE = "so101_ros2.camera_publisher"


def realsense_launch_cmd() -> list[str]:
    """`ros2 launch` argv for the Intel RealSense D415 colour stream.

    ROS2 Jazzy's launch parser rejects empty values (`camera_namespace:=`),
    so we set explicit names. Resulting topic: `/realsense/top/color/image_raw`.
    Keep `camera_config.json` in sync with this.

    Install once:  sudo apt install ros-$ROS_DISTRO-realsense2-camera
    """
    return [
        "ros2", "launch", "realsense2_camera", "rs_launch.py",
        "camera_namespace:=realsense",
        "camera_name:=top",
        "enable_color:=true",
        "enable_depth:=false",
        "enable_infra1:=false",
        "enable_infra2:=false",
        "pointcloud.enable:=false",
        "publish_tf:=false",
        # Power-cycle the camera at launch. Fixes "depth_module.enable_auto_exposure:
        # Device or resource busy" XU errors that hit when librealsense queries
        # depth-module options on a camera left in a stale state by a prior failed
        # init or by hot-swapping the sensor (we just replaced ours mid-session).
        "initial_reset:=true",
        f"rgb_camera.color_profile:={IMAGE_W}x{IMAGE_H}x{FPS}",
    ]


# ── Camera roles for the interactive labeller ─────────────────────────────────
# Three cameras around the SO-ARM101:
#   top   — Intel RealSense D415, served by realsense2_camera ROS2 driver
#   front — UVC ArduCam, workspace-front view
#   wrist — UVC ArduCam, gripper-mounted view
CAMERA_ROLES = {
    "0": {
        "name": "top",
        "label": "observation.images.top",
        "topic": "/realsense/top/color/image_raw",
        "description": "Intel RealSense D415 — top / overhead view",
    },
    "1": {
        "name": "front",
        "label": "observation.images.front",
        "topic": "/arducam/front/image_raw",
        "description": "ArduCam — front view (workspace level)",
    },
    "2": {
        "name": "wrist",
        "label": "observation.images.wrist",
        "topic": "/arducam/wrist/image_raw",
        "description": "ArduCam — wrist / gripper-mounted",
    },
}
REQUIRED_ROLES = {"top", "front", "wrist"}


def load_camera_config(*, require_all: bool = True) -> dict:
    """Read camera_config.json.

    Args:
        require_all: when True (default — used by data_collector, since dataset
            schemas need every camera), exits with an error if any role is
            missing. When False (used by the dashboard), prints a note and
            returns whatever roles are configured.
    """
    if not CONFIG_FILE.exists():
        sys.stderr.write(f"ERROR: {CONFIG_FILE} not found.\n")
        sys.stderr.write("Run  poetry run so101-configure  first.\n")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    if not cfg:
        sys.stderr.write("ERROR: camera_config.json is empty.\n")
        sys.stderr.write("Run  poetry run so101-configure\n")
        sys.exit(1)
    missing = REQUIRED_ROLES - cfg.keys()
    if missing:
        msg = f"camera_config.json missing roles: {sorted(missing)}\n"
        if require_all:
            sys.stderr.write(f"ERROR: {msg}")
            sys.stderr.write("Re-run  poetry run so101-configure\n")
            sys.exit(1)
        else:
            sys.stderr.write(f"Note: {msg}")
            sys.stderr.write("(Only the configured cameras will appear in the dashboard.)\n")
    return cfg
