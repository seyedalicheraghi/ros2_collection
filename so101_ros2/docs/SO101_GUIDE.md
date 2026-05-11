# SO101 Complete Guide — Setup · Data Collection · Pi0.5 Training

End-to-end guide for the SO101 robot arm on an **Intel Mac (macOS x86_64)**:
software install → arm calibration → camera setup → dataset recording → model training.

---

## Table of Contents

1. [Hardware Overview](#1-hardware-overview)
2. [Software Setup](#2-software-setup)
3. [Camera Setup (ROS2)](#3-camera-setup-ros2)
4. [Collecting a Dataset](#4-collecting-a-dataset)
5. [Dataset Format](#5-dataset-format)
6. [How the Collector Works](#6-how-the-collector-works)
7. [Configuring the Script](#7-configuring-the-script)
8. [Training Pi0.5](#8-training-pi05)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Hardware Overview

```
┌─────────────────────────────────────────────────────────┐
│                      Mac (Intel)                        │
│                                                         │
│   collect_data_ros2.py                                  │
│   ├── ROS2 thread  ←── ArduCam shoulder  (USB, index 0) │
│   │                ←── ArduCam wrist     (USB, index 1) │
│   │                ←── RealSense RGB     (USB, index 3) │
│   │                                                     │
│   └── lerobot     ←── Leader board      (USB-C)        │
│                   ←── Follower board    (USB-C)        │
└─────────────────────────────────────────────────────────┘
```

| Device | Role | Connection | Port / Topic |
|---|---|---|---|
| Leader arm controller board | Human input | USB-C | `/dev/cu.usbmodem5B415319121` |
| Follower arm controller board | Robot being recorded | USB-C | `/dev/cu.usbmodem5B415328921` |
| ArduCam — base | Lowest position, workspace level view | USB, index **1** | `/arducam/shoulder/image_raw` |
| ArduCam — wrist | Next to the gripper | USB, index **2** | `/arducam/wrist/image_raw` |
| Intel RealSense | Top-down world scene | USB, index **0** | `/camera/color/image_raw` |

> Camera indices verified by capturing test frames. Index 3 is the MacBook built-in webcam — do not use it.
> If ports change after unplugging, run `ls /dev/cu.usbmodem*` with each board plugged in separately.

---

## 2. Software Setup

> **All packages below are already installed** in the `lerobot` conda env.
> This section is for reference only — skip to section 3 to start using the system.

> Intel Mac has no CUDA/MPS. Robot control and data collection work fine on CPU.
> Use a Linux GPU machine for real training runs.

### Installed packages (verified)

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 1.26.4 | Must be <2 — cv_bridge and torch are compiled against NumPy 1.x |
| `torch` | 2.2.2 | Last PyTorch build for Intel Mac |
| `torchvision` | 0.17.2 | Compatible with torch 2.2.2 |
| `lerobot` | — | Robot control framework |
| `opencv-python-headless` | 4.10.0.84 | Camera capture via AVFoundation |
| `rerun-sdk` | 0.24.0 | Visualisation |
| `rclpy` + ROS2 Humble | — | Camera message publishing |
| `cv_bridge` | — | ROS2 ↔ numpy image conversion |
| `message_filters` | — | Camera timestamp synchronisation |
| `pyrealsense2` | — | RealSense SDK (accessed via OpenCV on macOS) |

### If setting up from scratch

```bash
# 1. Clone and create env
git clone https://github.com/SeeedStudio/lerobot.git ~/Projects/lerobot
cd ~/Projects/lerobot
conda create -n lerobot python=3.10 -y
conda activate lerobot

# 2. PyTorch (Intel Mac — last supported build)
pip install torch==2.2.2 torchvision==0.17.2

# 3. lerobot + dependencies
pip install -e ".[so101]"
pip install opencv-python-headless==4.10.0.84 rerun-sdk==0.24.0

# 4. Pin NumPy to 1.x — cv_bridge and torch are compiled against NumPy 1.x
pip install "numpy<2"

# 4. ROS2 Humble via RoboStack
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults || true
conda install ros-humble-desktop ros-humble-cv-bridge ros-humble-message-filters

# 5. RealSense SDK
conda install -c conda-forge pyrealsense2
```

---

## 3. Camera Setup (ROS2)

### Step 1 — Identify and label your cameras (run once)

```bash
conda activate lerobot
cd ~/Projects/lerobot
python configure_cameras.py
```

The script opens a snapshot from each detected camera in Preview. For each one, type:

| Input | Camera |
|---|---|
| `0` | RealSense — world scene (top-down overview) |
| `1` | ArduCam — base (lowest position on arm) |
| `2` | ArduCam — wrist / grip (next to gripper) |
| `s` | Skip (MacBook built-in or unknown) |

This saves `camera_config.json` and prints the exact publisher commands to run.

### Step 2 — Start the camera publishers (3 terminals)

Use the commands printed by `configure_cameras.py`. They look like:

```bash
# Terminal 1 — base camera
python opencv_camera_publisher.py --index 1 --topic /arducam/shoulder/image_raw

# Terminal 2 — wrist/grip camera
python opencv_camera_publisher.py --index 2 --topic /arducam/wrist/image_raw

# Terminal 3 — world scene (RealSense)
python opencv_camera_publisher.py --index 0 --topic /camera/color/image_raw
```

> Indices are set by `configure_cameras.py` and may differ on your machine.
> Re-run it any time cameras are replugged in a different order.

### Verify all topics are publishing

```bash
conda activate lerobot
ros2 topic list
# Expected:
#   /arducam/shoulder/image_raw
#   /arducam/wrist/image_raw
#   /camera/color/image_raw

ros2 topic hz /arducam/shoulder/image_raw   # ~30 Hz
ros2 topic hz /arducam/wrist/image_raw      # ~30 Hz
ros2 topic hz /camera/color/image_raw       # ~30 Hz
```

---

## 4. Collecting a Dataset

With all 3 camera terminals from section 4 running, open a **4th terminal**:

```bash
conda activate lerobot
cd ~/Projects/lerobot
python collect_data_ros2.py
```

### Recording workflow

```
Script starts
  └── connects arms
  └── waits for all 3 camera streams ...........  ready

Task description: pick up the red block
Start from episode [0]: 0

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Task : pick up the red block
  Start: episode 0
  Keys : [e] save & next  [c] cancel & retry  [q] save & quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Episode 0  |  task: 'pick up the red block'
  [e] save & next   [c] cancel & retry   [q] save & quit

  203 frames                ← live counter updates as you move the arm

[press e]
  ✓ Episode 0 saved  (203 frames)

  Episode 1  |  task: 'pick up the red block'
  ...
  87 frames

[press c]                   ← bad run, discard it
  ✗ Episode 1 cancelled — retrying.

  Episode 1  |  task: 'pick up the red block'
  ...
  198 frames

[press e]
  ✓ Episode 1 saved  (198 frames)

  Episode 2 ...

[press q]
  ✓ Episode 2 saved  (211 frames)

Disconnecting arms ...
Done. Saved 3 episode(s) to:
  /Users/ali/lerobot_datasets/local/so101_arducam
```

### Single-key controls (no Enter needed)

| Key | Action |
|---|---|
| **e** | Save current episode, automatically start next episode |
| **c** | Discard current episode, retry the same episode number |
| **q** | Save current episode and quit |

### Resuming a previous session

If the dataset folder already exists the script resumes automatically —
new episodes are appended without overwriting existing data.

---

## 5. Dataset Format

### Location on disk

```
~/lerobot_datasets/
└── local/
    └── so101_arducam/
        ├── meta/
        │   ├── info.json          dataset schema, fps, total episodes
        │   ├── stats.json         per-feature mean/std/min/max
        │   ├── tasks.jsonl        list of unique task strings
        │   └── episodes.jsonl     per-episode metadata (length, task)
        │
        ├── data/
        │   └── chunk-000/
        │       ├── episode_000000.parquet
        │       ├── episode_000001.parquet
        │       └── ...
        │
        └── videos/
            └── chunk-000/
                ├── observation.images.shoulder/
                │   └── episode_000000.mp4
                ├── observation.images.wrist/
                │   └── episode_000000.mp4
                └── observation.images.realsense/
                    └── episode_000000.mp4
```

### Per-frame data keys

| Key | Shape | Contents |
|---|---|---|
| `observation.images.shoulder` | (480, 640, 3) | RGB frame from shoulder ArduCam |
| `observation.images.wrist` | (480, 640, 3) | RGB frame from wrist ArduCam |
| `observation.images.realsense` | (480, 640, 3) | RGB frame from RealSense |
| `observation.state` | float32[6] | Follower joint angles (degrees) |
| `action` | float32[6] | Leader positions sent to follower |
| `task` | str | Natural language task description |

Joint order (index 0–5): `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`, `gripper`.

---

## 6. How the Collector Works

```
┌──────────────────────────────────────────────────────────────┐
│  Main thread                                                 │
│                                                              │
│  while recording:                                            │
│    1. read latest camera frames  (from ROS2 thread)          │
│    2. read leader joint positions (serial)                   │
│    3. send positions to follower  (serial)  ← teleoperation  │
│    4. read follower joint positions (serial)                 │
│    5. pack into frame dict                                   │
│    6. dataset.add_frame(frame)                               │
│    7. sleep to maintain 30 fps                               │
│                                                              │
│  on Ctrl+C → prompt keep/discard → save_episode or clear    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  ROS2 background thread  (daemon)                            │
│                                                              │
│  ApproximateTimeSynchronizer waits for matching timestamps   │
│  from all 3 camera topics, then fires _on_frames():          │
│    - decode ROS Image → numpy RGB array                      │
│    - resize to 640×480 if needed                             │
│    - store under a threading.Lock                            │
└──────────────────────────────────────────────────────────────┘
```

`opencv_camera_publisher.py` runs in each camera terminal, capturing from
the USB camera via OpenCV (AVFoundation on macOS) and publishing 640×480
RGB8 images at 30 fps to the appropriate ROS2 topic.

---

## 7. Configuring the Script

All settings are constants at the top of `collect_data_ros2.py`:

| Constant | Default | Change when |
|---|---|---|
| `FOLLOWER_PORT` | `/dev/cu.usbmodem5B415328921` | Follower board gets a new port |
| `LEADER_PORT` | `/dev/cu.usbmodem5B415319121` | Leader board gets a new port |
| `SHOULDER_TOPIC` | `/arducam/shoulder/image_raw` | Topic name changes |
| `WRIST_TOPIC` | `/arducam/wrist/image_raw` | Topic name changes |
| `REALSENSE_TOPIC` | `/camera/color/image_raw` | Topic name changes |
| `DATASET_REPO_ID` | `local/so101_arducam` | Renaming the dataset |
| `DATASET_ROOT` | `~/lerobot_datasets` | Saving data to a different drive |
| `FPS` | `30` | Changing recording frame rate |
| `IMAGE_H / IMAGE_W` | `480 / 640` | Changing camera resolution |

> **Important:** Changing `IMAGE_H`, `IMAGE_W`, or `MOTOR_NAMES` after a
> dataset has been started makes new episodes incompatible. Delete the
> dataset folder and start fresh if you change these.

---

## 8. Training Pi0.5

### Collect enough episodes

50–200 episodes is typical for a single task. Aim for variety in object
position and arm approach angle.

### Train on the Mac (CPU — slow, for testing only)

```bash
conda activate lerobot
cd ~/Projects/lerobot
lerobot-train --policy.type=pi05 --dataset.repo_id=local/so101_arducam --dataset.root=~/lerobot_datasets --output_dir=outputs/pi05_so101 --job_name=pi05_so101 --policy.pretrained_path=lerobot/pi05_base --policy.device=cpu --policy.dtype=float32 --wandb.enable=false --steps=20000 --batch_size=4
```

### Copy dataset to a GPU machine for real training

```bash
rsync -avz ~/lerobot_datasets/local/so101_arducam/ user@gpu-machine:~/lerobot_datasets/local/so101_arducam/
```

Then on the GPU machine:

```bash
conda activate lerobot
cd ~/Projects/lerobot
lerobot-train --policy.type=pi05 --dataset.repo_id=local/so101_arducam --dataset.root=~/lerobot_datasets --output_dir=outputs/pi05_so101 --job_name=pi05_so101 --policy.pretrained_path=lerobot/pi05_base --policy.device=cuda --policy.dtype=bfloat16 --wandb.enable=false --steps=20000 --batch_size=16
```

### Run inference (move follower from trained policy)

```bash
conda activate lerobot
lerobot-record --robot.type=so101_follower --robot.port=/dev/cu.usbmodem5B415328921 --robot.id=my_follower_arm --robot.cameras="{shoulder: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, realsense: {type: opencv, index_or_path: 3, width: 640, height: 480, fps: 30}}" --dataset.repo_id=local/so101_eval --dataset.root=~/lerobot_datasets --dataset.single_task="pick up the red block" --policy.path=outputs/pi05_so101/checkpoints/last/pretrained_model
```

---

## 9. Troubleshooting

**`ros2: command not found`**
```bash
conda activate lerobot   # RoboStack sets up ROS2 on env activation
```

**Script hangs at "Waiting for camera streams"**
- Check all 3 publisher terminals are running: `ros2 topic list`
- Check topics are publishing: `ros2 topic hz /arducam/shoulder/image_raw`
- Verify topic names match `SHOULDER_TOPIC`, `WRIST_TOPIC`, `REALSENSE_TOPIC` in `collect_data_ros2.py`

**Frames never synchronize**
- Try increasing `slop` in `ApproximateTimeSynchronizer` from `1/FPS` to `0.1`
- Check all cameras publish at similar rates: `ros2 topic hz <topic>`

**Wrong camera (shoulder/wrist views swapped)**
Run the snapshot script from section 4 to identify which index is which,
then update the `--index` in the publisher terminal.

**Camera index changed after replug**
USB camera indices are assigned by macOS at connection time. If you unplug
and replug cameras in a different order, re-run the snapshot script to
re-identify indices.

**ConnectionError on arm connect**
```bash
ls /dev/cu.usbmodem*          # confirm both boards detected
sudo chmod 666 /dev/cu.usbmodem*
ls ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
ls ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/
```

**Image colours look wrong (blue/red swapped)**
The publisher already converts BGR→RGB. If colours still look wrong, check
`desired_encoding` in `CameraNode._on_frames()` in `collect_data_ros2.py`.

**Dataset folder exists but is from an old config**
```bash
rm -rf ~/lerobot_datasets/local/so101_arducam
```
Then rerun — it will create a fresh dataset.

**`ValueError: Magnitude X exceeds 2047` during teleoperation**
The arm was not centred at the middle-position prompt during calibration.
Re-run `lerobot-calibrate` and ensure the arm is physically centred before
pressing Enter.
