"""SO101 + ROS2 data-collection package.

A thin layer on top of LeRobot for recording teleoperation episodes with
USB cameras + an SO-ARM101 leader/follower pair. Self-contained — does not
modify or extend any module under `src/lerobot/`.

After `poetry install`, the package exposes four console scripts:

    poetry run so101-configure
    poetry run so101-publisher --index 0 --topic /camera/color/image_raw
    poetry run so101-dashboard
    poetry run so101-record

The verbose `python -m so101_ros2.<module>` form also works.
"""

__version__ = "0.1.0"
