"""Headless camera publisher launcher.

Starts every camera publisher listed in camera_config.json (one subprocess
each) and stays in the foreground so Ctrl+C cleanly tears them all down.
Equivalent to clicking "Start all" in the dashboard, but without Qt.

Use this when you want to record without keeping the dashboard window open:

    Terminal 1:  poetry run so101-publish      # this script
    Terminal 2:  poetry run so101-record

Each child subprocess inherits the dashboard's launch logic from
`PublisherProcess`:
  * "top"  → `ros2 launch realsense2_camera ...`
  * others → `python -m so101_ros2.camera_publisher --index ... --topic ...`
"""

from __future__ import annotations

import signal
import sys
import time

from so101_ros2.dashboard import PublisherProcess
from so101_ros2.settings import load_camera_config


def main() -> None:
    cam_cfg = load_camera_config(require_all=False)
    if not cam_cfg:
        sys.stderr.write("ERROR: no cameras in camera_config.json — run so101-configure first.\n")
        sys.exit(1)

    publishers: list[PublisherProcess] = []
    for name, info in cam_cfg.items():
        publishers.append(PublisherProcess(
            name=name,
            index=int(info["index"]) if info.get("index") is not None else None,
            topic=info["topic"],
        ))

    print(f"Starting {len(publishers)} camera publisher(s):")
    for p in publishers:
        source = "ROS2 driver (realsense2_camera)" if p.is_realsense else f"index {p.index}"
        print(f"  {p.name:10s}  {source}  →  {p.topic}")
        p.start()

    print("\nPublishers running. Press Ctrl+C to stop.\n", flush=True)

    stop_requested = False

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_requested:
            # Surface any publisher that died (e.g. realsense node crashed).
            for p in publishers:
                if not p.is_running():
                    print(f"  ⚠  publisher '{p.name}' exited — restart so101-publish to retry", flush=True)
                    stop_requested = True
                    break
            time.sleep(0.5)
    finally:
        print("\nStopping publishers ...", flush=True)
        for p in publishers:
            p.stop()
        print("Done.")


if __name__ == "__main__":
    main()
