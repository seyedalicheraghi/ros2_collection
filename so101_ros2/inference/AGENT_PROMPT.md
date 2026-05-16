# Agent prompt — set up Pi0.5 inference on this checkout

You are helping a user run real-time Pi0.5 (VLA policy) inference on a
SO-ARM101 follower arm using the code in this repository. **The code is
tested and working** — your job is environment setup, not code changes.

---

## Strict rules (do not violate)

1. **Do NOT edit any Python code in `so101_ros2/`.** Every script in here
   was verified end-to-end on 2026-05-16 (3.62B-param Pi0.5 ran at 30 Hz on
   an RTX 3060 12 GB, controlled an SO-ARM101 follower for 100+ steps at
   100% bus success). Bugs you think you see in the code are almost
   certainly environment issues — debug those first.
2. **Do NOT run `lerobot-calibrate`** on this machine. Calibration files
   live in `so101_ros2/calibration/` and are version-controlled. Running
   `lerobot-calibrate` would rewrite both the JSON AND the motor's
   `Homing_Offset` register, silently breaking the trained model.
3. **Do NOT modify any file in `so101_ros2/calibration/`.** They are paired
   with motor EEPROM state on the physical arm.
4. **Do NOT downgrade or bump lerobot version** in `src/lerobot/`. The
   inference script (`so101_ros2/inference/run_inference.py`) contains a
   compat shim (`make_compat_checkpoint`) that handles version skew between
   the v0.4 local lerobot and v0.5-trained checkpoints. Leave it alone.
5. **Do NOT change** the `torch.autocast(...)` wrap in the inference loop.
   It's there because Pi0.5 internally creates fp32 tensors that need
   casting to match bf16 weights. Removing it causes a dtype error on the
   first inference step.

---

## What this project does

Closed-loop Pi0.5 inference on a SO-ARM101 follower arm:

- 3 cameras (1 Intel RealSense D415, 2 ArduCam B0578) publish to ROS2
  topics (top / front / wrist).
- Follower joint positions read via Feetech STS3215 servo SDK.
- Each tick (~30 Hz): build observation → Pi0.5 forward → unnormalize
  action → send goal position to motors.

Entry point: `so101_ros2/inference/run_inference.py`
(also installed as `poetry run so101-inference`).

---

## Hardware required

| Item | Required |
|---|---|
| SO-ARM101 follower arm | **Yes — the same physical arm trained on** (motor EEPROM stores per-motor `Homing_Offset` that pairs with the calibration JSON) |
| SO-ARM101 leader arm | **No** — inference doesn't use it; can be unplugged |
| 1 × Intel RealSense D415 | Yes — top camera |
| 2 × ArduCam B0578 | Yes — front and wrist cameras |
| GPU with ≥ 8 GB VRAM | Yes — model is 3.6B params; loads in ~7 GB bf16 + a few GB activations. **Tested on RTX 3060 12 GB.** Anything Ampere or newer works; older needs verification. |
| USB hub (powered) | Recommended if host has < 4 USB ports |

Per-USB power: ArduCams draw ~500-800 mA at startup. A bus-powered hub
will brown out. Use a hub with its own brick.

---

## Software stack (in install order)

```
Ubuntu 24.04 (or JetPack 6.x on Jetson)
└── ROS2 Jazzy (apt: ros-jazzy-ros-base, ros-jazzy-cv-bridge,
                     ros-jazzy-image-transport, ros-jazzy-realsense2-camera)
└── Python 3.12 (system)
└── FFmpeg 6.x (apt: ffmpeg libavutil-dev)
└── Poetry (https://python-poetry.org/docs/#installation)
└── Project venv:
    cd so101_ros2 && poetry install
└── Python deps NOT in the default Poetry install:
    poetry run pip install "numpy<2"                     # cv_bridge needs NumPy 1.x
    poetry run pip install \
        "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
        "scipy>=1.10.1,<1.15" \
        "accelerate>=1.0.0,<2.0.0"
└── HuggingFace auth (for the gated PaliGemma model and checkpoint pull):
    hf auth login --token hf_... --add-to-git-credential
```

### Why each non-default dep is required

- **`numpy<2`** — `poetry install` may upgrade NumPy to 2.x, which breaks
  the apt-installed `cv_bridge` (compiled against NumPy 1.x). Symptoms:
  `ImportError: A module that was compiled using NumPy 1.x cannot be run
  in NumPy 2.2.6`. Fix is always to pin `numpy<2`.
- **`transformers` from the lerobot_openpi branch** — Pi0.5 calls
  `CONFIG_MAPPING["paligemma"]()`. Vanilla HF transformers' paligemma
  config has a subtly different field layout than what lerobot pi0.5
  expects. Without this exact fork, you get `'NoneType' object is not
  subscriptable` (because lerobot's import-or-set-None pattern leaves the
  symbol as None when the right transformers isn't found).
- **`scipy`** — used internally by Pi0.5 for flow-matching scheduler.
- **`accelerate`** — used by lerobot's processor pipeline for device
  placement.

---

## HuggingFace gated model access (one-time)

Pi0.5 uses Google's PaliGemma-3B as its vision-language backbone. You
must request access **once per HF account**:

1. Open https://huggingface.co/google/paligemma-3b-pt-224
2. Click "Request access" / accept the license.
3. Approval is usually instant.

Without this, the first inference attempt fails with `GatedRepoError: 403`.

---

## Calibration setup (critical)

The calibration JSONs are version-controlled in `so101_ros2/calibration/`.
The lerobot SDK looks them up at `~/.cache/huggingface/lerobot/calibration/...`.
Bridge the two with symlinks (do this once per machine):

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower \
         ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader

ln -s "$(pwd)/so101_ros2/calibration/robots/so_follower/my_follower_arm.json" \
      ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower_arm.json

ln -s "$(pwd)/so101_ros2/calibration/teleoperators/so_leader/my_leader_arm.json" \
      ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/my_leader_arm.json
```

Verify both sides match before running inference:
```bash
sha256sum so101_ros2/calibration/robots/so_follower/my_follower_arm.json
```
The hash must equal what trained the model. If it doesn't, the inference
will map normalized actions to wrong physical positions and the arm will
move incorrectly.

---

## Port mapping (per-machine)

The SO-ARM uses `/dev/serial/by-id/...` symlinks for stable identification
across reboots and replugs. Confirm with:

```bash
ls -l /dev/serial/by-id/
```

Look for `usb-1a86_USB_Single_Serial_...` entries. The follower's serial
is whatever was used during training. If you don't know it, ask the user
or check `so101_ros2/settings.py`'s `FOLLOWER_PORT`. If the new machine
has a different physical arm or USB controller, you may need to update
`FOLLOWER_PORT` — that's the **one settings.py edit allowed** for porting
to a new device.

---

## Camera topics

Inference reads frames from these three ROS2 topics:
- `/realsense/top/color/image_raw`
- `/arducam/front/image_raw`
- `/arducam/wrist/image_raw`

These must be publishing **before** inference starts. Easiest:

```bash
# Terminal A
cd so101_ros2 && poetry run so101-dashboard
# Click "Publish" on top, front, wrist tiles.
```

If a camera index doesn't match the physical port assignment, run
`poetry run so101-configure` (interactive labeller) to re-write
`so101_ros2/camera_config.json`. That JSON is the only camera-config
file allowed to differ per-machine.

---

## The checkpoint

The trained model is on HuggingFace Hub:
**`alicheraghi/pi05_so101_lerobotv2`** (public, ~8.8 GB).

Download to local disk (one-time):
```bash
poetry run python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='alicheraghi/pi05_so101_lerobotv2',
    local_dir='/some/path/pi05_so101_lerobotv2',
)
print('weights at:', path)
"
```

---

## Run inference

Two terminals:

```bash
# Terminal A — cameras
source /opt/ros/jazzy/setup.bash
cd so101_ros2 && poetry run so101-dashboard
# Click Publish on top, front, wrist. Verify FPS > 25 on each.
```

```bash
# Terminal B — inference (in a second shell)
source /opt/ros/jazzy/setup.bash
cd so101_ros2

# Always start with a dry-run to confirm the pipeline works without
# moving the arm:
poetry run so101-inference \
    --checkpoint /path/to/pi05_so101_lerobotv2 \
    --task "Pick up the white box and place it in the white target area." \
    --dry-run \
    --max-iters 5
```

**The task string must match what was used during data collection
exactly**: `"Pick up the white box and place it in the white target area."`

If 5 dry-run iterations print state+action lines successfully, run live:

```bash
poetry run so101-inference \
    --checkpoint /path/to/pi05_so101_lerobotv2 \
    --task "Pick up the white box and place it in the white target area."
```

Press Ctrl+C to stop — the script disables follower torque on exit.

---

## Known issues with documented fixes

### "`'NoneType' object is not subscriptable`" during policy load
Cause: `transformers` not installed or wrong version.
Fix: `pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"`

### "`mat1 and mat2 must have the same dtype, but got Float and BFloat16`"
Cause: missing `torch.autocast` wrap. **This should never happen if you
didn't edit `run_inference.py`.** If you see it, you (or a previous
agent) modified the inference loop — revert your changes.

### "`The fields use_relative_actions, ... are not valid for PI05Config`"
Cause: compat shim was disabled or the checkpoint dir was edited.
**This should never happen if you didn't edit `run_inference.py`.** If
it does, revert.

### "`GatedRepoError: 403`" on PaliGemma
Cause: HF account doesn't have access.
Fix: request access at https://huggingface.co/google/paligemma-3b-pt-224

### "`libavutil.so.X: cannot open shared object file`"
Cause: FFmpeg system library missing.
Fix: `sudo apt install ffmpeg libavutil-dev`

### "`ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`"
Cause: `poetry install` upgraded NumPy past the cv_bridge boundary.
Fix: `poetry run pip install "numpy<2"`

### "`Permission denied` on `/dev/serial/by-id/...`"
Cause: user not in dialout group, or fresh boot reset permissions.
Fix: `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1` (matches whatever
the by-id symlinks point at).

### Camera shows "blue/black" frames or wrong scene
Cause: camera USB enumeration order changed.
Fix: re-run `poetry run so101-configure`.

### Arm doesn't pick up the box (closes gripper too high)
Cause: small calibration drift or distribution shift in box position.
Fix: use the `--action-bias` flag, e.g. `--action-bias "shoulder_lift=-8"`,
and iterate values. **Do NOT retrain or recalibrate to fix this.**

### Arm bus glitches drop success rate below ~95%
Cause: marginal 12V supply or loose data cable on follower.
Fix: physical — reseat data cable at motor 1; check 12V brick is at full
voltage. Do not modify `_apply_safe_limits` in `safe_teleop.py`.

---

## Hardware-specific notes

### Jetson AGX Thor (ARM64)
- PyTorch wheel must come from NVIDIA's JetPack 6.x repository (PyPI
  wheels are x86_64-only).
- Other deps install identically.
- 128 GB unified memory means you can drop bf16 if you want and run fp32
  for marginally better accuracy.
- See [project memory: pi05-inference-setup-quirks] for details.

### RTX 3060 12 GB (workstation)
- bf16 is required (fp32 won't fit).
- Verified working with PyTorch 2.7.1+cu126.

### RTX 5090 (Blackwell)
- Needs PyTorch ≥ 2.7 with CUDA 12.8 kernels for native sm_120 support.
- The `runpod-torch-v280` template has the right combination.

---

## Smoke test (what success looks like)

After all setup steps, a dry-run should print (within ~60 s of starting):

```
Camera config:
  top        → /realsense/top/color/image_raw         → observation.images.top
  wrist      → /arducam/wrist/image_raw               → observation.images.wrist
  front      → /arducam/front/image_raw               → observation.images.front
[INFO] [...] Subscribed to 3 camera topic(s).
Waiting for camera streams ... ready.
Connecting follower (leader is NOT used in inference)…
Loading calibration…
Applying safe motor limits to follower (torque caps, accel caps, P/D)…
applying per-motor settings (upstream so101_follower.configure() values):
  id=1 shoulder_pan    torque=1000  accel=254  P=16  D=32
  ... (5 more)
Loading policy from /path/to/checkpoint …
  compat: stripped ['action_feature_names', 'relative_exclude_joints', 'use_relative_actions'] from config.json (safe to drop)
✓ Loaded state dict from model.safetensors
Policy: PI05Policy  (3.62B params)  device=cuda  dtype=torch.bfloat16
Loading preprocessor/postprocessor from checkpoint…

Running at 30 Hz. Task: 'Pick up the white box and place it in the white target area.'. [DRY RUN — no motor writes]
Ctrl-C to stop.

  iter=    1  ok=100.0%  state=[ ... 6 values ... ]  action=[ ... 6 values ... ]
  iter=    2  ok=100.0%  state=[ ... ]  action=[ ... ]
  ... etc
```

If you see this, you're set. Hand control back to the user.

---

## What you should NOT do as an agent

- **Don't edit any `.py` file in `so101_ros2/`** — including the inference
  script. Every line is there for a reason documented above.
- **Don't run `lerobot-calibrate`** — destroys reproducibility.
- **Don't try to "fix" the version skew** by upgrading the vendored
  `src/lerobot/` — the compat shim handles it cleanly.
- **Don't add image preprocessing** — the loaded preprocessor handles
  resize, normalization, tokenization. Adding more breaks the model.
- **Don't suggest retraining** as a first response to inference issues —
  try `--action-bias` first.
- **Don't change `--dtype`** to fp16 — Pi0.5 was trained in bf16 and
  fp16 numerics differ enough to break action predictions.

---

## When you're done

The user should be able to run the live inference command above and see
the SO-ARM follower arm move in response to the camera input. If they
report "the gripper closes too high" or similar accuracy issues, point
them to the `--action-bias` flag — that's intentional for live tuning
and does NOT require code changes.

If anything in this prompt fails despite following it exactly, ask the
user for the full error output before proposing changes. Most failures
will be environment misconfigurations (missing dep, wrong port, ROS2 not
sourced, calibration symlink missing) — not code bugs.
