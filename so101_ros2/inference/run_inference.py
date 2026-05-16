"""Pi0.5 closed-loop inference on a connected SO-ARM101 follower.

    poetry run python -m so101_ros2.inference.run_inference \\
        --checkpoint /media/ali/AE0043120042E147/checkpoints/pi05_so101_lerobotv2/checkpoints/020000/pretrained_model \\
        --task "pick up the orange block"

Only the FOLLOWER arm needs to be connected (with 12V power on). The leader
arm is NOT required for inference.

Pipeline per tick (30 Hz default):
    follower joint counts ───┐
                             ├──► LeRobot-normalize (counts → -100..100 / 0..100)
    camera frames ───────────┤
                             │     ─► preprocessor (image resize, mean/std,
                             │         task tokenize, move to device)
                             │            ─► policy.select_action(...)
                             │                 ─► postprocessor (un-mean/std)
                             │                      ─► LeRobot-unnormalize → counts
                             │                           ─► MAX_STEP clip → goal
                             └──► applied to follower via sync_write_goals

Safety:
  • Per-motor torque/accel caps (safe_teleop._apply_safe_limits) applied at
    startup so policy actions can't draw bus brown-out current.
  • Each per-tick goal delta is clipped to safe_teleop.MAX_STEP (~9°) so
    a wild policy action can never slam the arm in one tick.
  • Ctrl-C disables torque and closes the port (motors go floppy).
  • --dry-run inferences without sending any motor commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
import torch
from scservo_sdk import GroupSyncRead, PacketHandler

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from so101_ros2 import safe_teleop
from so101_ros2.camera_streams import CameraStreams, spin_in_background
from so101_ros2.settings import FOLLOWER_PORT, FPS, JOINT_NAMES, load_camera_config


GRIPPER_ID = 6

# Fields written by newer lerobot (>=0.5) that older lerobot (≤0.4.x) doesn't
# know about. Two categories:
#   - "always safe to drop"  : descriptive metadata or only-used-when-other-field-set
#   - "behavior-critical"    : changes inference semantics; must equal expected value
_PI05_NEW_FIELDS_ALWAYS_DROP = (
    "relative_exclude_joints",  # only consulted when use_relative_actions=True
    "action_feature_names",      # purely descriptive joint-name list
)
_PI05_NEW_FIELDS_CRITICAL = {
    # If use_relative_actions=True, the model emits delta-actions and absolute
    # interpretation would crash the robot. Abort rather than silently change.
    "use_relative_actions": False,
}


# ── State / action normalization (LeRobot RANGE_M100_100 / RANGE_0_100) ─────
# Mirrors so101_ros2.data_collector._normalize_follower (forward direction)
# and lerobot.motors.motors_bus.MotorsBus._unnormalize (inverse).
def normalize_state(sid: int, raw: int, f_min: int, f_max: int) -> float:
    if f_max <= f_min:
        return 0.0
    bounded = max(f_min, min(f_max, raw))
    frac = (bounded - f_min) / (f_max - f_min)
    if sid == GRIPPER_ID:
        return float(frac * 100.0)
    return float(frac * 200.0 - 100.0)


def unnormalize_action(sid: int, value: float, f_min: int, f_max: int) -> int:
    if sid == GRIPPER_ID:
        bounded = max(0.0, min(100.0, value))
        frac = bounded / 100.0
    else:
        bounded = max(-100.0, min(100.0, value))
        frac = (bounded + 100.0) / 200.0
    return int(round(f_min + frac * (f_max - f_min)))


def parse_action_bias(s: str) -> np.ndarray:
    """Parse a "joint=val,joint=val" string into a (6,) per-joint bias array
    indexed by JOINT_NAMES order. Values are in LeRobot-normalized units
    (-100..100 for body joints, 0..100 for gripper)."""
    bias = np.zeros(6, dtype=np.float32)
    if not s.strip():
        return bias
    name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"--action-bias token {token!r} missing '='")
        name, val = token.split("=", 1)
        name = name.strip()
        if name not in name_to_idx:
            raise ValueError(
                f"--action-bias joint {name!r} unknown; must be one of {list(name_to_idx)}"
            )
        bias[name_to_idx[name]] = float(val)
    return bias


# ── Cleanup ────────────────────────────────────────────────────────────────
_cleaned = False


def cleanup() -> None:
    """Disable follower torque + close port so the arm is safe at rest."""
    global _cleaned
    if _cleaned:
        return
    _cleaned = True
    if safe_teleop._f_ph is not None and safe_teleop._pkt is not None:
        try:
            for sid in safe_teleop.IDS:
                safe_teleop._pkt.write1ByteTxRx(
                    safe_teleop._f_ph, sid, safe_teleop.ADDR_TORQUE_ENABLE, 0
                )
        except Exception:
            pass
        try:
            safe_teleop._f_ph.closePort()
        except Exception:
            pass


def on_signal(*_) -> None:
    print("\nstopping — torque off, port closed…", flush=True)
    cleanup()
    sys.exit(0)


# ── Checkpoint compat shim (lerobot 0.5 → 0.4 config) ─────────────────────
def make_compat_checkpoint(src: Path) -> Path:
    """Return a path to a checkpoint dir whose config.json is loadable by the
    local (older) lerobot. If no patching is needed, returns `src` unchanged.

    Otherwise symlinks every file from `src` into a fresh temp dir, then
    writes a sanitized `config.json` there with the unknown fields removed.
    Aborts if a stripped field has a non-default value (would silently change
    inference behavior).
    """
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return src  # let downstream raise a clean "missing config" error

    cfg = json.loads(cfg_path.read_text())
    fields_to_drop: list[str] = []

    # Behavior-critical fields: abort if anything but the expected value
    for k, expected in _PI05_NEW_FIELDS_CRITICAL.items():
        if k not in cfg:
            continue
        if cfg[k] != expected:
            sys.stderr.write(
                f"\nABORT: checkpoint has {k}={cfg[k]!r} (expected {expected!r}).\n"
                f"This field changes inference semantics. Update local src/lerobot to v0.5+ "
                f"before running inference on this checkpoint.\n"
            )
            sys.exit(2)
        fields_to_drop.append(k)

    # Always-safe-to-drop fields: descriptive metadata
    for k in _PI05_NEW_FIELDS_ALWAYS_DROP:
        if k in cfg:
            fields_to_drop.append(k)

    if not fields_to_drop:
        return src  # nothing to patch — load as-is

    tmp = Path(tempfile.mkdtemp(prefix="pi05_compat_"))
    # Symlink all files except config.json so we don't duplicate 8.8 GB
    for child in src.iterdir():
        if child.name == "config.json":
            continue
        (tmp / child.name).symlink_to(child.resolve())
    # Write sanitized config
    for k in fields_to_drop:
        cfg.pop(k, None)
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"  compat: stripped {sorted(fields_to_drop)} from config.json (safe to drop)\n"
        f"  compat: using sanitized checkpoint at {tmp}"
    )
    return tmp


# ── Follower-only bus setup ────────────────────────────────────────────────
def connect_follower() -> tuple[dict[int, tuple[int, int, int, int]], GroupSyncRead]:
    """Follower-only version of safe_teleop.connect_arms.

    Opens only the follower port (skips leader since it isn't connected),
    loads calibration, applies safe per-motor limits, and builds a sync_read
    for Present_Position.
    """
    safe_teleop._f_ph = safe_teleop._open(FOLLOWER_PORT)
    safe_teleop._pkt = PacketHandler(0)

    print("Loading calibration…")
    calib = safe_teleop._load_calibration()

    print("Applying safe motor limits to follower (torque caps, accel caps, P/D)…")
    # _apply_safe_limits only touches the follower (_f_ph); safe to call without
    # leader connected.
    safe_teleop._apply_safe_limits()

    f_read = GroupSyncRead(
        safe_teleop._f_ph, safe_teleop._pkt, safe_teleop.ADDR_PRESENT_POSITION, 2
    )
    for sid in safe_teleop.IDS:
        f_read.addParam(sid)
    return calib, f_read


# ── Build observation dict for pi05 ────────────────────────────────────────
def build_observation(
    state_norm: np.ndarray,
    frames: dict[str, np.ndarray],
    task: str,
    device: str,
) -> dict:
    """Construct the observation dict that pi05's preprocessor expects.

    Args:
        state_norm: (6,) float32 — LeRobot-normalized joint positions.
        frames: {dataset_key: (H, W, 3) uint8 RGB}, one entry per camera.
        task: natural-language instruction.
        device: torch device string ("cuda", "cpu").

    Returns:
        Dict matching the format the training dataloader produced:
            observation.images.{top,front,wrist}: (1, 3, H, W) float32 in [0, 1]
            observation.state: (1, 6) float32
            task: [str] of length 1
    """
    obs: dict = {}
    for key, frame in frames.items():
        # uint8 (H, W, 3) → float32 (1, 3, H, W) ∈ [0, 1]
        t = torch.from_numpy(frame).to(device=device, dtype=torch.float32) / 255.0
        obs[key] = t.permute(2, 0, 1).unsqueeze(0)
    obs["observation.state"] = (
        torch.from_numpy(state_norm).to(device=device, dtype=torch.float32).unsqueeze(0)
    )
    obs["task"] = [task]
    return obs


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pi0.5 closed-loop inference on a connected SO-ARM101 follower"
    )
    parser.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the checkpoint's pretrained_model/ directory (or HF Hub repo id)",
    )
    parser.add_argument(
        "--task", required=True, type=str,
        help="Natural-language task description; should match what was used during data collection",
    )
    parser.add_argument("--fps", type=int, default=FPS, help="Control-loop frequency in Hz")
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float32", "float16"],
        help="Inference precision (bf16 is the training precision)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help='Torch device (e.g. "cuda", "cuda:0", "cpu"). pi05 needs GPU for real-time.',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run inference but skip motor writes (follower stays put). Use to debug shapes/devices.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=0,
        help="Stop after this many successful inference steps (0 = run forever, default).",
    )
    parser.add_argument(
        "--action-bias", type=str, default="",
        help=(
            'Per-joint additive bias on the normalized action, applied AFTER postprocessor '
            'but BEFORE un-normalization. Format: "shoulder_lift=-8,elbow_flex=+5". Units are '
            'LeRobot-normalized (-100..100 for body joints, 0..100 for gripper). Use to '
            'compensate for small calibration drift or distribution shift without retraining.'
        ),
    )
    args = parser.parse_args()
    action_bias = parse_action_bias(args.action_bias)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # ── 1. Cameras (top/front/wrist via ROS2 topics) ───────────────────────
    cam_cfg = load_camera_config()
    print("\nCamera config:")
    for name, info in cam_cfg.items():
        print(f"  {name:10s} → {info['topic']:38s} → {info['label']}")
    cameras = [{"label": info["label"], "topic": info["topic"]} for info in cam_cfg.values()]

    rclpy.init()
    streams = CameraStreams(cameras, resize=True)
    executor, _ = spin_in_background(streams)

    print("\nWaiting for camera streams (run `so101-dashboard` and Publish all 3)", end="", flush=True)
    deadline = time.perf_counter() + 30.0
    while not streams.ready:
        if time.perf_counter() > deadline:
            sys.stderr.write(
                "\nERROR: cameras not publishing after 30s. Open `so101-dashboard` and Publish "
                "top/front/wrist before re-running.\n"
            )
            executor.shutdown()
            rclpy.shutdown()
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(0.2)
    print(" ready.")

    # ── 2. Follower arm (no leader needed) ─────────────────────────────────
    print("\nConnecting follower (leader is NOT used in inference)…")
    calib, f_read = connect_follower()

    # ── 3. Policy + processors ─────────────────────────────────────────────
    print(f"\nLoading policy from {args.checkpoint} …")
    ckpt = make_compat_checkpoint(args.checkpoint)
    config = PreTrainedConfig.from_pretrained(ckpt)
    config.device = str(device)
    policy = PI05Policy.from_pretrained(ckpt, config=config)
    policy = policy.to(device, dtype=dtype)
    policy.eval()
    policy.reset()
    n_params = sum(p.numel() for p in policy.parameters()) / 1e9
    print(f"Policy: {type(policy).__name__}  ({n_params:.2f}B params)  device={device}  dtype={dtype}")

    print("Loading preprocessor/postprocessor from checkpoint…")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── 4. Main inference loop ─────────────────────────────────────────────
    print(
        f"\nRunning at {args.fps} Hz. Task: {args.task!r}. "
        f"{'[DRY RUN — no motor writes]' if args.dry_run else ''}"
    )
    if np.any(action_bias != 0):
        print("Action biases (normalized units, added to policy output before un-normalization):")
        for i, n in enumerate(JOINT_NAMES):
            if action_bias[i] != 0:
                print(f"  {n}: {action_bias[i]:+.2f}")
    print("Ctrl-C to stop.\n")

    dt = 1.0 / args.fps
    iter_count = 0
    bus_glitches = 0
    try:
        while True:
            t0 = time.perf_counter()

            # Read follower joint positions
            try:
                pos = safe_teleop._sync_read_retry(safe_teleop._f_ph, f_read)
            except Exception as e:
                bus_glitches += 1
                if bus_glitches <= 3 or bus_glitches % 30 == 0:
                    print(f"  bus glitch #{bus_glitches}: {e}", flush=True)
                time.sleep(0.005)
                continue

            # Read latest camera frames
            frames = streams.get_frames()
            if any(v is None for v in frames.values()):
                time.sleep(0.005)
                continue

            # LeRobot-normalize follower state (matches the training data format)
            state_norm = np.empty(6, dtype=np.float32)
            for sid in safe_teleop.IDS:
                _, _, f_min, f_max = calib[sid]
                state_norm[sid - 1] = normalize_state(sid, pos[sid], f_min, f_max)

            # Forward through preprocessor → policy → postprocessor
            observation = build_observation(state_norm, frames, args.task, str(device))
            # autocast covers pi05's internal noisy_actions (created as fp32) being
            # multiplied against bf16 weights — without it we hit a dtype mismatch.
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype):
                batch = preprocessor(observation)
                action = policy.select_action(batch)
                action = postprocessor(action)

            # action: (1, 6) tensor in LeRobot-normalized space
            action_np = action.detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
            # Apply per-joint additive bias (no-op if --action-bias was empty)
            action_np = action_np + action_bias

            # Un-normalize per joint, clip per-tick motion, build goal positions
            goals: list[int] = []
            for sid in safe_teleop.IDS:
                _, _, f_min, f_max = calib[sid]
                present = pos[sid]
                target = unnormalize_action(sid, float(action_np[sid - 1]), f_min, f_max)
                delta = target - present
                if delta >  safe_teleop.MAX_STEP: delta =  safe_teleop.MAX_STEP
                if delta < -safe_teleop.MAX_STEP: delta = -safe_teleop.MAX_STEP
                goals.append(present + delta)

            if not args.dry_run:
                try:
                    safe_teleop._sync_write_goals(goals)
                except Exception as e:
                    bus_glitches += 1
                    print(f"  write error: {e}", flush=True)

            iter_count += 1
            state_str = " ".join(f"{v:+6.1f}" for v in state_norm)
            action_str = " ".join(f"{v:+6.1f}" for v in action_np)
            # Print every iteration for the first 5 (fast feedback), then every 30
            if iter_count <= 5 or iter_count % 30 == 0:
                ok_pct = 100.0 * (iter_count - bus_glitches) / max(iter_count, 1)
                print(
                    f"  iter={iter_count:>5}  ok={ok_pct:5.1f}%  "
                    f"state=[{state_str}]  action=[{action_str}]",
                    flush=True,
                )

            if args.max_iters and iter_count >= args.max_iters:
                print(f"\nReached --max-iters={args.max_iters}, exiting.")
                break

            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        try:
            streams.destroy_node()
            executor.shutdown()
            rclpy.shutdown()
        except Exception:
            pass
        print(f"\nDone. {iter_count} inference steps, {bus_glitches} bus glitches.")


if __name__ == "__main__":
    main()
