"""Teleop + dataset recording loop.

Reads ROS2 image topics via the shared CameraStreams node, runs leader→follower
teleoperation through LeRobot's SO-ARM driver, and writes a LeRobotDataset to
disk. Run after the camera publishers are up (either through the dashboard or
in dedicated terminals).

    poetry run so101-record

Single-keypress controls (no Enter needed):
    e  →  save current episode, automatically start next
    c  →  cancel & retry the same episode number
    q  →  save current episode and quit
"""

from __future__ import annotations

import queue
import shutil
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

from so101_ros2.camera_streams import CameraStreams, spin_in_background
from so101_ros2.settings import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    FPS,
    IMAGE_H,
    IMAGE_W,
    JOINT_NAMES,
    LEADER_ID,
    LEADER_PORT,
    MOTOR_NAMES,
    load_camera_config,
)


class KeyReader:
    """Reads single keypresses from stdin in a daemon thread (no Enter needed)."""

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                ch = sys.stdin.read(1)
                self._q.put(ch.lower())
        except Exception:
            pass

    def poll(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def restore(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _dict_to_array(d: dict, keys: list[str]) -> np.ndarray:
    return np.array([d[k] for k in keys], dtype=np.float32)


def _probe_extra_reads(follower: SOFollower) -> tuple[bool, bool]:
    """Check whether Present_Velocity / Present_Load are readable on this bus.

    SO-ARM101 uses Feetech STS3215 motors which expose both, but probing once
    at startup keeps the recording loop from raising on a misconfigured bus.
    Returns (has_velocity, has_effort).
    """
    has_v = has_e = False
    try:
        follower.bus.sync_read("Present_Velocity")
        has_v = True
    except Exception as e:
        print(f"  warning: Present_Velocity not readable ({e}) — skipping observation.velocity")
    try:
        follower.bus.sync_read("Present_Load")
        has_e = True
    except Exception as e:
        print(f"  warning: Present_Load not readable ({e}) — skipping observation.effort")
    return has_v, has_e


def record_episode(
    episode_num: int,
    task: str,
    dataset: LeRobotDataset,
    streams: CameraStreams,
    follower: SOFollower,
    leader: SOLeader,
    keys: KeyReader,
    record_velocity: bool,
    record_effort: bool,
) -> tuple[int, str]:
    """Record one episode. Returns (frame_count, control_key)."""
    print(f"\n  Episode {episode_num}  |  task: '{task}'")
    print("  [e] save & next   [c] cancel & retry   [q] save & quit\n")

    frame_count = 0
    dt = 1.0 / FPS

    while True:
        t0 = time.perf_counter()

        key = keys.poll()
        if key in ("e", "c", "q"):
            print()
            return frame_count, key

        frames = streams.get_frames()
        if any(v is None for v in frames.values()):
            time.sleep(0.005)
            continue

        leader_action = leader.get_action()
        follower.send_action(leader_action)
        follower_obs = follower.get_observation()
        # Each sync_read is one bus round-trip (~5–10 ms). Skipping these
        # entirely on machines where they're not supported keeps the loop
        # from stalling when the underlying bus throws.
        velocity = follower.bus.sync_read("Present_Velocity") if record_velocity else None
        effort   = follower.bus.sync_read("Present_Load")     if record_effort   else None

        frame = {
            **frames,
            "observation.state": _dict_to_array(follower_obs, MOTOR_NAMES),
            "action":            _dict_to_array(leader_action, MOTOR_NAMES),
            "task":              task,
        }
        if velocity is not None:
            frame["observation.velocity"] = _dict_to_array(velocity, JOINT_NAMES)
        if effort is not None:
            frame["observation.effort"]   = _dict_to_array(effort,   JOINT_NAMES)

        dataset.add_frame(frame)
        frame_count += 1
        print(f"\r  {frame_count} frames", end="", flush=True)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))


def main() -> None:
    cam_cfg = load_camera_config()
    print("\nCamera config loaded:")
    for name, info in cam_cfg.items():
        print(f"  {name:10s} → index {info['index']}  topic: {info['topic']}")

    cameras = [{"label": info["label"], "topic": info["topic"]} for info in cam_cfg.values()]

    # Build dataset feature schema from camera roles + arm joint names.
    dataset_features: dict = {
        info["label"]: {
            "dtype": "video",
            "shape": (IMAGE_H, IMAGE_W, 3),
            "names": ["height", "width", "channels"],
        }
        for info in cam_cfg.values()
    }
    dataset_features["observation.state"] = {
        "dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES,
    }
    dataset_features["action"] = {
        "dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES,
    }
    # Velocity and effort are added to the schema only if the bus actually
    # supports them — see _probe_extra_reads below. Both are added here as a
    # placeholder so the conditional schema build is in one place.
    extra_velocity_feature = {
        "dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES,
    }
    extra_effort_feature = {
        "dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES,
    }

    # ROS2 setup.
    rclpy.init()
    streams = CameraStreams(cameras, resize=True)
    executor, _ = spin_in_background(streams)

    # Wait for cameras BEFORE arms — keeps the serial bus from idle-timing-out.
    print("\nWaiting for camera streams", end="", flush=True)
    while not streams.ready:
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" ready.")

    # Connect arms.
    print("\nConnecting follower arm ...")
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    follower.connect(calibrate=False)
    print("Connecting leader arm ...")
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect(calibrate=False)

    # Probe extra follower registers (Present_Velocity / Present_Load) so we
    # can expand the dataset schema BEFORE creating it on disk.
    record_velocity, record_effort = _probe_extra_reads(follower)
    if record_velocity:
        dataset_features["observation.velocity"] = extra_velocity_feature
        print("  ✓ recording observation.velocity (Present_Velocity)")
    if record_effort:
        dataset_features["observation.effort"] = extra_effort_feature
        print("  ✓ recording observation.effort (Present_Load — signed torque proxy)")

    # Open or create the dataset on disk. When resuming, the schema is
    # whatever the existing dataset declares — align our recording flags
    # to that so we don't write fields the dataset doesn't have, or skip
    # fields it expects.
    dataset_path = DATASET_ROOT / DATASET_REPO_ID
    dataset_ready = (dataset_path / "meta" / "info.json").exists()
    if dataset_ready:
        print(f"\nResuming existing dataset at {dataset_path}")
        dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, root=dataset_path)
        existing = set(dataset.meta.features.keys())
        # Override the runtime probe to match what's already on disk.
        record_velocity = "observation.velocity" in existing
        record_effort   = "observation.effort"   in existing
        print(f"  observation.velocity in dataset: {record_velocity}")
        print(f"  observation.effort   in dataset: {record_effort}")
    else:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        print(f"\nCreating new dataset at {dataset_path}")
        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=FPS,
            root=dataset_path,
            robot_type="so101",
            features=dataset_features,
            use_videos=True,
            vcodec="h264",
        )

    # Episode loop.
    print()
    task = input("Task description: ").strip() or "teleoperation"
    start_ep_input = input(f"Start from episode [{dataset.meta.total_episodes}]: ").strip()
    episode_num = int(start_ep_input) if start_ep_input.isdigit() else dataset.meta.total_episodes

    print()
    print("━" * 50)
    print(f"  Task : {task}")
    print(f"  Start: episode {episode_num}")
    print("  Keys : [e] save & next  [c] cancel & retry  [q] save & quit")
    print("━" * 50)

    keys = KeyReader()
    episodes_saved = 0

    try:
        while True:
            n_frames, key = record_episode(
                episode_num, task, dataset, streams, follower, leader, keys,
                record_velocity=record_velocity,
                record_effort=record_effort,
            )
            if key in ("e", "q"):
                if n_frames > 0:
                    print(f"  saving episode {episode_num} ...", flush=True)
                    # parallel_encoding=False avoids a fork() deadlock: the parent
                    # process has multiple threads (rclpy executor, KeyReader),
                    # and ProcessPoolExecutor children inherit half-held locks
                    # which hang ffmpeg. Sequential encoding is ~1-2s/episode.
                    dataset.save_episode(parallel_encoding=False)
                    episodes_saved += 1
                    print(f"  ✓ Episode {episode_num} saved  ({n_frames} frames)")
                    episode_num += 1
                else:
                    print(f"  Episode {episode_num}: no frames — skipped.")
                if key == "q":
                    break
            elif key == "c":
                dataset.clear_episode_buffer()
                print(f"  ✗ Episode {episode_num} cancelled — retrying.")
    finally:
        print("\n  cleanup: restoring terminal ...", flush=True)
        keys.restore()
        print("  cleanup: disconnecting follower ...", flush=True)
        try:
            follower.disconnect()
        except Exception as e:
            print(f"    (follower disconnect raised: {e})")
        print("  cleanup: disconnecting leader ...", flush=True)
        try:
            leader.disconnect()
        except Exception as e:
            print(f"    (leader disconnect raised: {e})")
        print("  cleanup: shutting down rclpy ...", flush=True)
        try:
            streams.destroy_node()
            executor.shutdown()
            rclpy.shutdown()
        except Exception as e:
            print(f"    (rclpy shutdown raised: {e})")
        print(f"Done. Saved {episodes_saved} episode(s) to:\n  {dataset_path}")


if __name__ == "__main__":
    main()
