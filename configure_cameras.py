#!/usr/bin/env python3
"""
Interactive camera configuration tool for SO101 data collection.

Opens a snapshot from each detected USB camera in Preview so you can
see what it shows, then asks you to label it. Saves the result to
camera_config.json which is used by collect_data_ros2.py and the
publisher launch commands.

Usage:
  python configure_cameras.py
"""

import cv2
import json
import subprocess
import os
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "camera_config.json"
SNAPSHOT_DIR = Path("/tmp/cam_config")

ROLES = {
    "0": {
        "name": "realsense",
        "label": "observation.images.realsense",
        "topic": "/camera/color/image_raw",
        "description": "RealSense — world scene (top-down overview)",
    },
    "1": {
        "name": "base",
        "label": "observation.images.shoulder",
        "topic": "/arducam/shoulder/image_raw",
        "description": "ArduCam — base camera (lowest position on arm)",
    },
    "2": {
        "name": "wrist",
        "label": "observation.images.wrist",
        "topic": "/arducam/wrist/image_raw",
        "description": "ArduCam — wrist/grip camera (next to gripper)",
    },
}


def capture_frame(index: int) -> tuple:
    """Open camera, flush buffer, return (frame, width, height) or None."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ret or frame is None:
        return None
    return frame, w, h


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    config = {}

    print()
    print("━" * 55)
    print("  SO101 Camera Configuration")
    print("━" * 55)
    print()
    print("For each camera a snapshot will open in Preview.")
    print("Type the role number and press Enter:\n")
    print("  0  →  RealSense (world scene / top-down)")
    print("  1  →  ArduCam base (lowest position on arm)")
    print("  2  →  ArduCam wrist / grip (next to gripper)")
    print("  s  →  skip (built-in webcam or unknown)")
    print()

    found = False
    for idx in range(7):
        result = capture_frame(idx)
        if result is None:
            continue
        frame, w, h = result
        found = True

        # Burn index label into the saved image so it's visible in Preview.
        labeled = frame.copy()
        cv2.putText(
            labeled, f"INDEX {idx}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4,
        )
        snap_path = SNAPSHOT_DIR / f"camera_{idx}.jpg"
        cv2.imwrite(str(snap_path), labeled)

        # Open snapshot in macOS Preview.
        subprocess.Popen(["open", str(snap_path)])

        print(f"Camera {idx}  ({w}×{h})")
        print(f"  Snapshot: {snap_path}")
        choice = input("  Role [0/1/2/s]: ").strip().lower()

        if choice in ROLES:
            role = ROLES[choice]
            config[role["name"]] = {
                "index": idx,
                "topic": role["topic"],
                "label": role["label"],
                "description": role["description"],
            }
            print(f"  ✓ Saved as: {role['description']}\n")
        else:
            print(f"  – Skipped\n")

    if not found:
        print("No cameras detected. Make sure cameras are plugged in.")
        return

    # Check all three roles were assigned.
    missing = [r["name"] for r in ROLES.values() if r["name"] not in config]
    if missing:
        print(f"Warning: roles not assigned: {', '.join(missing)}")
        print("Re-run configure_cameras.py to complete the configuration.\n")

    # Write config file.
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print("━" * 55)
    print(f"  Config saved → {CONFIG_FILE}")
    print("━" * 55)
    print()
    print("Camera mapping:")
    for name, info in config.items():
        print(f"  index {info['index']}  {name:10s}  {info['topic']}")

    print()
    print("Next — open 3 terminals and run one command in each:")
    print()
    for name, info in config.items():
        print(f"  # {info['description']}")
        print(f"  python opencv_camera_publisher.py --index {info['index']} --topic {info['topic']}")
        print()

    print("Then in a 4th terminal:")
    print("  python collect_data_ros2.py")
    print()


if __name__ == "__main__":
    main()
