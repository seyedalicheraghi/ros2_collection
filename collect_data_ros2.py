#!/usr/bin/env python3
"""
ROS2 data collector for SO101 with ArduCam cameras.

Runs teleoperation (leader → follower) while recording synchronized
camera images and robot state in LeRobot format for pi0.5 fine-tuning.

Before running:
  1. python configure_cameras.py   ← identifies cameras, writes camera_config.json
  2. Start one publisher per camera (commands printed by configure_cameras.py)
  3. python collect_data_ros2.py

Episode controls (single keypress — no Enter needed):
  e  →  save current episode, automatically start next
  c  →  cancel / discard current episode, retry same number
  q  →  save current episode and quit
"""

import json
import queue
import shutil
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

# ── Hardware ports ─────────────────────────────────────────────────────────────
FOLLOWER_PORT = "/dev/cu.usbmodem5B415328921"
LEADER_PORT   = "/dev/cu.usbmodem5B415319121"
FOLLOWER_ID   = "my_follower_arm"
LEADER_ID     = "my_leader_arm"

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_REPO_ID = "local/so101_arducam"
DATASET_ROOT    = Path.home() / "lerobot_datasets"
FPS             = 30
IMAGE_H         = 480
IMAGE_W         = 640

MOTOR_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# ── Camera config ──────────────────────────────────────────────────────────────

CONFIG_FILE = Path(__file__).parent / "camera_config.json"

def load_camera_config() -> dict:
    if not CONFIG_FILE.exists():
        print(f"ERROR: {CONFIG_FILE} not found.")
        print("Run  python configure_cameras.py  first.")
        sys.exit(1)
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    missing = {"realsense", "base", "wrist"} - cfg.keys()
    if missing:
        print(f"ERROR: camera_config.json missing roles: {missing}")
        print("Re-run  python configure_cameras.py")
        sys.exit(1)
    return cfg


# ── Single-keypress reader ─────────────────────────────────────────────────────

class KeyReader:
    """Reads single keypresses from stdin in a background thread (no Enter needed)."""

    def __init__(self):
        self._q: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                ch = sys.stdin.read(1)
                self._q.put(ch.lower())
        except Exception:
            pass

    def poll(self) -> str | None:
        """Return the next key if one is waiting, else None."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


# ── ROS2 camera subscriber ─────────────────────────────────────────────────────

class CameraNode(Node):
    """Subscribes to 3 camera topics independently.

    Each camera updates its own slot as fast as it publishes — no
    synchronizer, so a slow camera (e.g. RealSense at 6 fps via AVFoundation)
    does not bottleneck the others. The recording loop always gets the latest
    available frame from each camera.
    """

    def __init__(self, cam_cfg: dict):
        super().__init__("so101_camera_node")
        self._bridge = CvBridge()
        self._lock   = threading.Lock()

        self._base_label      = cam_cfg["base"]["label"]
        self._wrist_label     = cam_cfg["wrist"]["label"]
        self._realsense_label = cam_cfg["realsense"]["label"]

        self._frames: dict[str, np.ndarray | None] = {
            self._base_label:      None,
            self._wrist_label:     None,
            self._realsense_label: None,
        }

        self.create_subscription(Image, cam_cfg["base"]["topic"],
                                 lambda m: self._store(m, self._base_label), 1)
        self.create_subscription(Image, cam_cfg["wrist"]["topic"],
                                 lambda m: self._store(m, self._wrist_label), 1)
        self.create_subscription(Image, cam_cfg["realsense"]["topic"],
                                 lambda m: self._store(m, self._realsense_label), 1)

        self.get_logger().info(
            f"Subscribed to:\n"
            f"  {cam_cfg['base']['topic']}  →  {self._base_label}\n"
            f"  {cam_cfg['wrist']['topic']}  →  {self._wrist_label}\n"
            f"  {cam_cfg['realsense']['topic']}  →  {self._realsense_label}"
        )

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (IMAGE_H, IMAGE_W):
            import cv2
            img = cv2.resize(img, (IMAGE_W, IMAGE_H))
        return img

    def _store(self, msg: Image, label: str) -> None:
        frame = self._resize(self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8"))
        with self._lock:
            self._frames[label] = frame

    def get_frames(self) -> dict[str, np.ndarray | None]:
        with self._lock:
            return dict(self._frames)

    @property
    def ready(self) -> bool:
        with self._lock:
            return all(v is not None for v in self._frames.values())


# ── Helpers ────────────────────────────────────────────────────────────────────

def dict_to_array(d: dict, keys: list[str]) -> np.ndarray:
    return np.array([d[k] for k in keys], dtype=np.float32)


# ── Episode recording ──────────────────────────────────────────────────────────

def record_episode(
    episode_num: int,
    task: str,
    dataset: LeRobotDataset,
    cam_node: CameraNode,
    follower: SOFollower,
    leader: SOLeader,
    keys: KeyReader,
) -> tuple[int, str]:
    """
    Record one episode. Returns (frame_count, key) where key is:
      'e'  →  user pressed End  (save, go to next episode)
      'c'  →  user pressed Cancel (discard, retry same episode)
      'q'  →  user pressed Quit  (save, exit)
    """
    print(f"\n  Episode {episode_num}  |  task: '{task}'")
    print("  [e] save & next   [c] cancel & retry   [q] save & quit\n")

    frame_count = 0
    dt = 1.0 / FPS

    while True:
        t0 = time.perf_counter()

        # Check for keypress first (non-blocking).
        key = keys.poll()
        if key in ("e", "c", "q"):
            print()
            return frame_count, key

        frames = cam_node.get_frames()
        if any(v is None for v in frames.values()):
            time.sleep(0.005)
            continue

        leader_action = leader.get_action()
        follower.send_action(leader_action)
        follower_obs = follower.get_observation()

        frame = {
            **frames,
            "observation.state": dict_to_array(follower_obs,  MOTOR_NAMES),
            "action":            dict_to_array(leader_action, MOTOR_NAMES),
            "task":              task,
        }
        dataset.add_frame(frame)
        frame_count += 1

        # Live frame counter on a single updating line.
        print(f"\r  {frame_count} frames", end="", flush=True)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load camera config.
    cam_cfg = load_camera_config()
    print("\nCamera config loaded:")
    for name, info in cam_cfg.items():
        print(f"  {name:10s} → index {info['index']}  topic: {info['topic']}")

    # 2. Build dataset features.
    dataset_features = {
        cam_cfg["base"]["label"]:      {"dtype": "video", "shape": (IMAGE_H, IMAGE_W, 3), "names": ["height", "width", "channels"]},
        cam_cfg["wrist"]["label"]:     {"dtype": "video", "shape": (IMAGE_H, IMAGE_W, 3), "names": ["height", "width", "channels"]},
        cam_cfg["realsense"]["label"]: {"dtype": "video", "shape": (IMAGE_H, IMAGE_W, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES},
        "action":            {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES},
    }

    # 3. Start ROS2.
    rclpy.init()
    cam_node = CameraNode(cam_cfg)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(cam_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    # 4. Create or resume dataset.
    dataset_path = DATASET_ROOT / DATASET_REPO_ID
    dataset_ready = (dataset_path / "meta" / "info.json").exists()
    if dataset_ready:
        print(f"\nResuming existing dataset at {dataset_path}")
        dataset = LeRobotDataset(repo_id=DATASET_REPO_ID, root=dataset_path)
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

    # 5. Wait for cameras FIRST — arms connect after so serial bus doesn't idle-timeout.
    print("\nWaiting for camera streams", end="", flush=True)
    while not cam_node.ready:
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" ready.")

    # 6. Connect arms (after cameras are ready so there is no idle wait).
    print("\nConnecting follower arm ...")
    follower = SOFollower(SOFollowerRobotConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID))
    follower.connect(calibrate=False)
    print("Connecting leader arm ...")
    leader = SOLeader(SOLeaderTeleopConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect(calibrate=False)

    # 7. Ask for task and starting episode.
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

    # 8. Episode loop with single-keypress control.
    keys = KeyReader()
    episodes_saved = 0

    try:
        while True:
            n_frames, key = record_episode(
                episode_num, task, dataset, cam_node, follower, leader, keys
            )

            if key in ("e", "q"):
                if n_frames > 0:
                    dataset.save_episode()
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
        keys.restore()
        print(f"\nDisconnecting arms ...")
        follower.disconnect()
        leader.disconnect()
        cam_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        print(f"Done. Saved {episodes_saved} episode(s) to:\n  {dataset_path}")


if __name__ == "__main__":
    main()
