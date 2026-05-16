"""Teleop + LeRobot dataset recording, on top of safe_teleop's bus path.

    poetry run so101-record

Workflow
--------
Terminal 1:  poetry run so101-dashboard
             → click "Publish" on top, front, wrist (verify FPS / sharpness)
Terminal 2:  poetry run so101-record
             → connects arms via safe_teleop.connect_arms(), records episodes

Single-keypress controls during recording (no Enter needed):
    e  →  save current episode, automatically start next
    c  →  cancel & retry the same episode number
    q  →  save current episode and quit

Per-frame schema (matches the reference dataset at
/media/ali/AE0043120042E147/lerobot_datasets_original) — suitable for
OpenPI Pi0.5 training.
---------------------------------------------------------------------
    observation.images.top      uint8 (H, W, 3)  RealSense color
    observation.images.front    uint8 (H, W, 3)  ArduCam workspace view
    observation.images.wrist    uint8 (H, W, 3)  ArduCam gripper-mounted
    observation.state           float32[6]       follower joints, LeRobot-normalized
                                                 (-100..100 for body, 0..100 for gripper)
                                                 — names suffixed with `.pos`
    action                      float32[6]       leader → follower target, normalized
                                                 same way as observation.state
    observation.velocity        float32[6]       raw signed counts/s from STS3215
                                                 Present_Velocity register
    observation.effort          float32[6]       raw signed value from Present_Load
                                                 (-1000..1000 ≈ -100%..+100% of motor torque)
    task                        str              natural-language description

Joint order: JOINT_NAMES from settings.py — i.e. shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper.

Boundaries
----------
This script does NOT modify safe_teleop, motor EEPROM, or calibration files.
The only motor write it performs is `safe_teleop._sync_write_goals(...)`,
which writes Goal_Position (RAM, same as safe_teleop itself). Per-motor
torque/accel caps come from `safe_teleop.connect_arms()` (the same call
the teleop CLI makes), so motor behavior here is identical to running
`so101-teleop` alone — we just also record frames.
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

from so101_ros2 import safe_teleop
from so101_ros2.camera_streams import CameraStreams, spin_in_background
from so101_ros2.settings import (
    DATASET_REPO_ID,
    DATASET_ROOT,
    FPS,
    IMAGE_H,
    IMAGE_W,
    JOINT_NAMES,
    load_camera_config,
)

# Gripper (id=6) uses LeRobot's RANGE_0_100 norm mode; the other 5 body
# joints use RANGE_M100_100. Matches src/lerobot/robots/so_follower/so_follower.py.
GRIPPER_ID = 6


def _normalize_follower(sid: int, raw: int, f_min: int, f_max: int) -> float:
    """Map raw motor counts → LeRobot-normalized range using follower calibration.

    Mirrors `MotorsBus._normalize` in src/lerobot/motors/motors_bus.py:841 so
    state/action match what `lerobot-record` would write for an SO101 follower:
    body joints span -100..+100, gripper spans 0..+100.
    """
    if f_max <= f_min:
        return 0.0
    bounded = max(f_min, min(f_max, raw))
    frac = (bounded - f_min) / (f_max - f_min)
    if sid == GRIPPER_ID:
        return float(frac * 100.0)
    return float(frac * 200.0 - 100.0)


class KeyReader:
    """Single-keypress reader from stdin (no Enter), via raw-mode termios."""

    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            while True:
                self._q.put(sys.stdin.read(1).lower())
        except Exception:
            pass

    def poll(self) -> str | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def restore(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def record_episode(
    episode_num: int,
    task: str,
    dataset: LeRobotDataset,
    streams: CameraStreams,
    calib: dict,
    follower_read,
    l_read,
    keys: KeyReader,
) -> tuple[int, str]:
    """Record one episode at FPS Hz. Returns (frame_count, control_key).

    Reads follower position+velocity+load in a single 6-byte sync_read so the
    extra signals don't add a second bus round-trip. Leader still uses the
    position-only read since velocity/load aren't needed to compute action.
    On a bus glitch (cascade failure after retries), we skip the frame and
    continue rather than aborting the episode.
    """
    print(f"\n  Episode {episode_num}  |  task: '{task}'")
    print("  [e] save & next   [c] cancel & retry   [q] save & quit\n")

    frame_count = 0
    bus_glitches = 0
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

        # ---- arm I/O: same pattern as safe_teleop.main() ----
        try:
            follower   = safe_teleop.sync_read_pos_vel_load_retry(follower_read)
            leader_pos = safe_teleop._sync_read_retry(safe_teleop._l_ph, l_read)
            goals: list[int] = []
            state    = np.empty(6, dtype=np.float32)
            action   = np.empty(6, dtype=np.float32)
            velocity = np.empty(6, dtype=np.float32)
            effort   = np.empty(6, dtype=np.float32)
            for sid in safe_teleop.IDS:
                l_min, l_max, f_min, f_max = calib[sid]
                pos, vel_raw, load_raw = follower[sid]
                target = safe_teleop._remap(leader_pos[sid], l_min, l_max, f_min, f_max)
                state[sid - 1]    = _normalize_follower(sid, pos,    f_min, f_max)
                action[sid - 1]   = _normalize_follower(sid, target, f_min, f_max)
                velocity[sid - 1] = float(vel_raw)
                effort[sid - 1]   = float(load_raw)
                delta = target - pos
                if delta >  safe_teleop.MAX_STEP: delta =  safe_teleop.MAX_STEP
                if delta < -safe_teleop.MAX_STEP: delta = -safe_teleop.MAX_STEP
                goals.append(pos + delta)
            safe_teleop._sync_write_goals(goals)
        except Exception as e:
            bus_glitches += 1
            if bus_glitches <= 3 or bus_glitches % 30 == 0:
                print(f"\n  bus glitch #{bus_glitches}: {e} — skipping frame", flush=True)
            time.sleep(0.005)
            continue
        # -------------------------------------------------------

        dataset.add_frame({
            **frames,
            "observation.state":    state,
            "action":               action,
            "observation.velocity": velocity,
            "observation.effort":   effort,
            "task":                 task,
        })
        frame_count += 1
        print(f"\r  {frame_count} frames  ({bus_glitches} bus skips)", end="", flush=True)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))


def _build_dataset_features(cam_cfg: dict) -> dict:
    feats = {
        info["label"]: {
            "dtype": "video",
            "shape": (IMAGE_H, IMAGE_W, 3),
            "names": ["height", "width", "channels"],
        }
        for info in cam_cfg.values()
    }
    pos_names = [f"{j}.pos" for j in JOINT_NAMES]
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES),),
        "names": pos_names,
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES),),
        "names": pos_names,
    }
    feats["observation.velocity"] = {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES),),
        "names": list(JOINT_NAMES),
    }
    feats["observation.effort"] = {
        "dtype": "float32",
        "shape": (len(JOINT_NAMES),),
        "names": list(JOINT_NAMES),
    }
    return feats


def _open_or_create_dataset(dataset_features: dict) -> LeRobotDataset:
    """Resume only if BOTH info.json AND tasks.parquet exist (a partial-write
    crash leaves info.json without tasks.parquet — the resume path would then
    fall through to the HF Hub fallback and 404). When partial, wipe + recreate.
    """
    dataset_path = DATASET_ROOT / DATASET_REPO_ID
    if dataset_path.exists():
        meta = dataset_path / "meta"
        if (meta / "info.json").exists() and (meta / "tasks.parquet").exists():
            print(f"\nResuming existing dataset at {dataset_path}")
            return LeRobotDataset(repo_id=DATASET_REPO_ID, root=dataset_path)
        print(f"\nFound partial dataset at {dataset_path} — wiping (no meta/tasks.parquet)")
        shutil.rmtree(dataset_path)

    print(f"\nCreating new dataset at {dataset_path}")
    return LeRobotDataset.create(
        repo_id=DATASET_REPO_ID,
        fps=FPS,
        root=dataset_path,
        robot_type="so101",
        features=dataset_features,
        use_videos=True,
        vcodec="h264",
    )


def main() -> None:
    cam_cfg = load_camera_config()
    print("\nCamera config loaded:")
    for name, info in cam_cfg.items():
        print(f"  {name:10s} → topic: {info['topic']}")

    cameras = [{"label": info["label"], "topic": info["topic"]} for info in cam_cfg.values()]
    dataset_features = _build_dataset_features(cam_cfg)

    rclpy.init()
    streams = CameraStreams(cameras, resize=True)
    executor, _ = spin_in_background(streams)

    print("\nWaiting for camera streams (run `so101-dashboard` and Publish all 3)", end="", flush=True)
    timeout_at = time.perf_counter() + 30.0
    while not streams.ready:
        if time.perf_counter() > timeout_at:
            print()
            sys.stderr.write(
                "\nERROR: cameras not publishing after 30s.\n"
                "Open the dashboard (`poetry run so101-dashboard`) and click "
                "Publish on top, front, and wrist before re-running.\n"
            )
            executor.shutdown()
            rclpy.shutdown()
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" ready.")

    # SAME bus path as safe_teleop. This sets safe_teleop._f_ph / _l_ph / _pkt
    # so the per-tick functions (_sync_read_retry, _sync_write_goals, _cleanup)
    # all work via the same module state the teleop CLI uses.
    print("\nConnecting arms via safe_teleop.connect_arms() ...")
    calib, _f_pos_only_read, l_read = safe_teleop.connect_arms()
    # Wider follower read: pos+vel+load in one bus round-trip. The
    # position-only group from connect_arms() is unused here — kept on
    # the safe_teleop side so safe_teleop.main() (the standalone teleop CLI)
    # still pays only the 2-byte cost.
    follower_read = safe_teleop.make_follower_pos_vel_load_read()

    dataset = _open_or_create_dataset(dataset_features)

    print()
    task = input("Task description: ").strip() or "teleoperation"
    start_ep_input = input(f"Start from episode [{dataset.meta.total_episodes}]: ").strip()
    episode_num = int(start_ep_input) if start_ep_input.isdigit() else dataset.meta.total_episodes

    print()
    print("━" * 50)
    print(f"  Task : {task}")
    print(f"  Start: episode {episode_num}")
    print(f"  Keys : [e] save & next  [c] cancel & retry  [q] save & quit")
    print("━" * 50)

    keys = KeyReader()
    episodes_saved = 0

    try:
        while True:
            n_frames, key = record_episode(
                episode_num, task, dataset, streams, calib, follower_read, l_read, keys
            )
            if key in ("e", "q"):
                if n_frames > 0:
                    print(f"  saving episode {episode_num} ...", flush=True)
                    # parallel_encoding=False avoids a fork() deadlock — parent
                    # holds rclpy + KeyReader threads with locks that ffmpeg
                    # children would inherit half-held.
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
        print("  cleanup: disconnecting arms (safe_teleop._cleanup) ...", flush=True)
        try:
            safe_teleop._cleanup()
        except Exception as e:
            print(f"    (cleanup raised: {e})")
        print("  cleanup: shutting down rclpy ...", flush=True)
        try:
            streams.destroy_node()
            executor.shutdown()
            rclpy.shutdown()
        except Exception as e:
            print(f"    (rclpy shutdown raised: {e})")
        print(f"\nDone. Saved {episodes_saved} episode(s) to:\n  {DATASET_ROOT / DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
