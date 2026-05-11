# so101_ros2 — SO-ARM101 + ROS2 data collection & training

End-to-end workflow for recording teleoperation episodes with the SO-ARM101
arm + 3 cameras — **top** (Intel RealSense D415, served by the
realsense2_camera ROS2 driver), **front** and **wrist** (ArduCam UVC) — and
training a policy on the result. The dataset is in **LeRobotDataset v3**
format, directly consumable by Hugging Face / LeRobot tooling and by
**[openpi](https://github.com/Physical-Intelligence/openpi)** for pi0 / pi0.5
fine-tuning.

This is a **separate package** that *uses* [LeRobot](../README.md) (for the arm
drivers and `LeRobotDataset` format) but is not integrated into it. Nothing
under `src/lerobot/` is modified — your code lives only in this directory.

---

## Contents

| Module                                | Purpose                                                                                                                                                              |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`configure_cameras`](configure_cameras.py) | Interactive labeller. Snaps each USB camera, asks which role it is, writes `camera_config.json`. Run once per machine.                                              |
| [`camera_publisher`](camera_publisher.py)   | Single-camera USB → ROS2 `Image` publisher. One process per camera.                                                                                                  |
| [`dashboard`](dashboard.py)                 | PySide6 GUI. Start/stop publishers, live preview, FPS / sharpness / brightness / clipping metrics, snapshot, RGB histogram. Use this to verify camera quality before recording. |
| [`data_collector`](data_collector.py)       | Teleop loop (leader → follower) that records `LeRobotDataset` episodes synced with the camera streams.                                                              |
| [`camera_streams`](camera_streams.py)       | Shared rclpy node — one `Image` subscription per topic, latest-frame slot per camera. Reused by `data_collector` and `dashboard`.                                    |
| [`settings`](settings.py)                   | Constants — FPS, image size, motor names, hardware ports, dataset paths. **Edit this file for your machine.**                                                        |

All modules ship as **console scripts** — after `poetry install` you can run:

```bash
poetry run so101-configure       # one-time camera labelling
poetry run so101-publisher --index 0 --topic /camera/color/image_raw
poetry run so101-dashboard       # GUI: start publishers, verify quality
poetry run so101-record          # teleop + dataset recording
```

The verbose `poetry run python -m so101_ros2.<module>` form also works.
Drop the `poetry run` prefix after `poetry env activate`.

---

## Quick start

This project is managed with [Poetry](https://python-poetry.org/).
Poetry creates an isolated Python venv, installs LeRobot as an editable path
dependency, and pins NumPy to the version cv_bridge was built against. ROS2
itself is not a Python package — it must be installed at the system level and
sourced into your shell.

### 0. Prerequisites (one-time per machine)

```bash
# 0.1 — Python 3.10+ on PATH.
python3 --version

# 0.2 — Poetry. Install method per the official docs:
curl -sSL https://install.python-poetry.org | python3 -
poetry --version

# 0.3 — ROS2 at the system level. Pick the distro that matches your Ubuntu:
#   Ubuntu 22.04 (jammy)  → Humble (Python 3.10)
#   Ubuntu 24.04 (noble)  → Jazzy  (Python 3.12)
#   sudo apt install ros-$ROS_DISTRO-desktop ros-$ROS_DISTRO-cv-bridge
#   macOS Intel: see docs/ROS2_SETUP.md (RoboStack is the only path that works).
ls /opt/ros/                            # confirms which distro is installed

```

> **The Poetry venv's Python version must match the ROS2 distro's Python**
> (3.10 for Humble, 3.12 for Jazzy). `rclpy` and `cv_bridge` are compiled
> C-extensions and won't import from a different interpreter.

### 1. Install Python deps with Poetry

> ⚠ **All `poetry` commands must be run from inside `so101_ros2/`** — this is
> the directory that contains *our* `pyproject.toml`. Running `poetry`
> from the repo root makes Poetry read the **upstream LeRobot** `pyproject.toml`
> instead and creates a venv named `lerobot-...` in the wrong place.

Pin Poetry to the Python that matches your ROS2 distro (3.10 for Humble,
3.12 for Jazzy) before installing — otherwise `import rclpy` will fail
with a Python-version mismatch:

```bash
cd ~/Projects/ros2_collection/so101_ros2

# One-time per machine: keep all Poetry venvs next to their projects.
poetry config virtualenvs.in-project true

poetry env use /usr/bin/python3.12          # Jazzy/Noble — use python3.10 for Humble/Jammy
poetry install
```

This creates a project-local venv at **`so101_ros2/.venv/`** (configured via
`poetry.toml`'s `virtualenvs.in-project = true` — keeps the env next to the
code it serves, instead of in `~/.cache/pypoetry/`) and installs:

- `lerobot[so101]` from `..` as an **editable path dep** (so upstream changes you pull in are picked up automatically).
- `PySide6` for the dashboard GUI.

The first `poetry install` takes a few minutes — torch + transformers come in via LeRobot's `[so101]` extra.

> If the LeRobot editable install is too slow or fails, comment out the
> `lerobot = { path = ... }` line in `pyproject.toml` and instead run:
> `poetry run pip install -e "..[so101]"`.

### 2. Source ROS2 — every shell

Sourcing puts `rclpy`, `cv_bridge`, `sensor_msgs`, and the `ros2` CLI on
`$PATH` / `$PYTHONPATH`. The Poetry venv inherits these, so `import rclpy`
works from inside `poetry run`.

```bash
source /opt/ros/$(ls /opt/ros)/setup.bash   # picks up whatever distro is installed
# or:  conda activate ros_env               # macOS RoboStack — see docs/ROS2_SETUP.md
echo $ROS_DISTRO                             # should print: humble  or  jazzy
```

Add the `source` line to your `~/.bashrc` if you want it automatic.

### 3. Identify and set the arm ports (one-time per machine)

Run the LeRobot port helper — it tells you to unplug each board in turn and
prints the path that disappeared:

```bash
poetry run lerobot-find-port
```

Then open [`settings.py`](settings.py) and set the two paths it gave you:

```python
FOLLOWER_PORT = "/dev/ttyACM0"     # Linux; macOS uses "/dev/cu.usbmodemXXX"
LEADER_PORT   = "/dev/ttyACM1"
```

> If the script errors out, you can also identify them by hand:
> `ls /dev/ttyACM*` (Linux) or `ls /dev/cu.usbmodem*` (macOS), with each
> board plugged in alone.

### 4. Calibrate the leader and follower arms (one-time per arm)

Calibration teaches LeRobot the **homing offsets** and **per-joint motion
ranges** of each physical arm. Without it, joint positions reported by the
motors don't map correctly to the policy's action space, and you'll see
errors like `ValueError: Magnitude X exceeds 2047` during teleoperation.

Calibrate **both** arms — once each, the first time you set up the hardware,
and any time you swap a motor or rebuild an arm. Use the `--id` you set in
`settings.py` (`my_follower_arm` / `my_leader_arm`); that becomes the
filename of the saved calibration.

#### Follower arm

```bash
poetry run lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
```

The script will then prompt you for two physical actions in sequence:

1. **"Move `{follower}` to the middle of its range of motion and press ENTER..."**
   Pose the arm in a neutral, *centred* posture — every joint roughly
   half-way between its mechanical limits, gripper half-open. This is what
   defines the homing offset; off-centre at this prompt is the most common
   cause of the "Magnitude X exceeds 2047" error later.
2. **"Move all joints except `wrist_roll` sequentially through their entire
   ranges of motion. Recording positions. Press ENTER to stop..."**
   Slowly walk each joint through its full physical travel — shoulder pan
   left↔right, shoulder lift up↔down, elbow flex, wrist flex, gripper
   open↔close. `wrist_roll` is auto-set to a full 360° because it has no
   hard stops. Press ENTER once you've covered every joint.

#### Leader arm

```bash
poetry run lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm
```

Same two prompts, same procedure. Both arms must be calibrated before
teleoperation will work.

#### Where calibration lives

Calibration JSON files are written to:

```
~/.cache/huggingface/lerobot/calibration/robots/so101_follower/my_follower_arm.json
~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/my_leader_arm.json
```

(Override the parent dir with the `HF_LEROBOT_CALIBRATION` env var if you
want it elsewhere.)

#### Re-running

If the files already exist, `lerobot-calibrate` will overwrite them. If you
just connect a previously-calibrated arm and the values look wrong, you can
also re-trigger calibration interactively from the connect-time prompt by
pressing `c` instead of ENTER ("Press ENTER to use provided calibration
file…, or type 'c' and press ENTER to run calibration").

#### Verify

After calibrating both arms, sanity-check with the LeRobot teleop CLI
(no recording, no cameras — just leader→follower mirroring):

```bash
poetry run lerobot-teleoperate \
    --robot.type=so101_follower    --robot.port=/dev/ttyACM0 --robot.id=my_follower_arm \
    --teleop.type=so101_leader     --teleop.port=/dev/ttyACM1 --teleop.id=my_leader_arm
```

Move the leader; the follower should track smoothly through the full range.
If a joint hits the calibrated limit too early, re-calibrate that arm and
sweep further during the second prompt.

### 5. Configure cameras (one-time per machine)

```bash
poetry run so101-configure
```

For each detected camera, a Qt window opens showing **live video** from
that camera (so you can wave your hand or move the arm to identify which
is which — much easier than a static snapshot). Press a key inside the
window to label it:

**Phase 1** of `so101-configure` auto-detects the RealSense via
`pyrealsense2` and adds it as the `top` role (no key press needed).
**Phase 2** scans the UVC ArduCams; for each one a live-video Qt window
opens and you press:

| Key | Role                                       |
| --- | ------------------------------------------ |
| `1` | front — workspace-level ArduCam            |
| `2` | wrist — gripper-mounted ArduCam            |
| `s` | skip (built-in webcam or unknown)          |
| `q` | quit early                                 |

Writes `so101_ros2/camera_config.json` (gitignored). Re-run any time
cameras are replugged in a different order.

### 6. Launch the dashboard — verify camera quality

```bash
poetry run so101-dashboard
```

In the toolbar click **Start all publishers**. Each panel should show:

| Metric     | Healthy range          |
| ---------- | ---------------------- |
| FPS        | ≈ 30                   |
| Resolution | 640×480 (matches `IMAGE_W` × `IMAGE_H`) |
| Latency    | < 50 ms                |
| Sharpness  | > 200 (rule of thumb — < 100 = soft / out of focus) |
| Brightness | 50–200 (avoid extremes)|
| Clipping % | < 1 %                  |

Per-panel buttons: **Snapshot** (PNG to `~/lerobot_datasets/snapshots/<ts>/`),
**Histogram** (RGB curves for exposure check), **Pause view**.

Toolbar: **Snapshot all**, **Stop all publishers**.

> Closing the dashboard terminates any publisher subprocesses it launched.

### 7. Record a dataset

With publishers running (from the dashboard or in dedicated terminals), in a
separate shell (don't forget to `source /opt/ros/$(ls /opt/ros)/setup.bash` there too):

```bash
poetry run so101-record
```

Single-keypress controls (no Enter needed):

| Key | Action                            |
| --- | --------------------------------- |
| `e` | Save episode, start the next one  |
| `c` | Cancel & retry the same episode   |
| `q` | Save episode and quit             |

The dataset is written to `~/lerobot_datasets/<DATASET_REPO_ID>/` in
`LeRobotDataset` v3 format (Parquet + MP4 + `meta/`). Re-running the script
**resumes** an existing dataset — new episodes are appended.

#### Per-frame schema

| Key                              | Shape         | Contents                                                      |
| -------------------------------- | ------------- | ------------------------------------------------------------- |
| `observation.images.top`         | `(480,640,3)` | RGB from the Intel RealSense D415 (top / overhead view)       |
| `observation.images.front`       | `(480,640,3)` | RGB from the front ArduCam (workspace level)                  |
| `observation.images.wrist`       | `(480,640,3)` | RGB from the wrist ArduCam (gripper-mounted)                  |
| `observation.state`              | `float32[6]`  | Follower **joint positions** — what openpi feeds the policy   |
| `observation.velocity`           | `float32[6]`  | Follower joint velocities (`Present_Velocity`)                |
| `observation.effort`             | `float32[6]`  | Follower joint loads — signed torque proxy (`Present_Load`)   |
| `action`                         | `float32[6]`  | Leader joint targets — teleop signal sent to follower         |
| `task`                           | str           | Task description typed at episode start                       |

> `observation.velocity` and `observation.effort` are recorded only if the
> motor bus reports those registers (probed at startup; STS3215 motors do).
> openpi's pi0 / pi0.5 read `observation.state` + `action` by default and
> ignore the extras — they're stored for analysis or future
> velocity/torque-aware policies.

Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.

### 8. Train Pi0.5

50–200 episodes per task is typical. On a CUDA box (LeRobot installs the
`lerobot-train` console script into the Poetry venv):

```bash
poetry run lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=local/so101_arducam \
  --dataset.root=~/lerobot_datasets \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --steps=20000 --batch_size=16
```

To copy a dataset between machines:

```bash
rsync -avz ~/lerobot_datasets/local/so101_arducam/ \
  user@gpu-host:~/lerobot_datasets/local/so101_arducam/
```

### 9. Run inference on the follower arm

```bash
poetry run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_follower_arm \
  --robot.cameras="{top: {type: realsense, serial: ?, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=local/so101_eval \
  --dataset.root=~/lerobot_datasets \
  --dataset.single_task="pick up the red block" \
  --policy.path=outputs/pi05_so101/checkpoints/last/pretrained_model
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Daily workflow                                                          │
│                                                                          │
│  configure_cameras  ──── camera_config.json  (one-time per machine)      │
│                                                                          │
│  dashboard ──┬── camera_publisher (×3)  ──► /arducam/.., /camera/..      │
│              │                                                           │
│              └── CameraStreams ◄── ROS2 topics  (live preview + metrics) │
│                                                                          │
│  data_collector ──┬── CameraStreams ◄── ROS2 topics                      │
│                   ├── lerobot.SOFollower  (serial)                       │
│                   ├── lerobot.SOLeader    (serial)                       │
│                   └── lerobot.LeRobotDataset  ──► ~/lerobot_datasets/    │
└──────────────────────────────────────────────────────────────────────────┘
```

`CameraStreams` keeps the **latest frame per topic** under a lock — if one
camera is publishing slower than the others (USB-bus contention, low-light
auto-exposure on one ArduCam, etc.) it does **not** bottleneck the others.
The recording loop polls at `FPS` and packs the latest frame from each
into one dataset row.

### Why this structure

* `so101_ros2/` is a **standalone package**: it imports from `lerobot` but
  ships nothing inside `src/lerobot/`. Pulling new upstream LeRobot changes
  is a clean merge — no conflicts in your code.
* Shared logic (`camera_streams.py`, `settings.py`) is extracted so the GUI
  and the recorder can't drift out of sync on what a frame looks like.
* `pyproject.toml` is **scoped to this directory** — Poetry only manages
  this package's Python venv, not the whole repo. LeRobot is pulled in as
  an editable path dep so its own pyproject still drives its install.
* All entry points use the `python -m so101_ros2.<module>` form (after
  Poetry env activation) so relative imports work without `sys.path` hacks.

---

## Daily-workflow tips

Two repetitive bits — sourcing ROS2 and prefixing every command with
`poetry run` — are easy to fold away:

```bash
# In ~/.bashrc:
source /opt/ros/$(ls /opt/ros)/setup.bash      # auto-source whatever ROS2 is installed

# In any shell, once per session — gives you a sub-shell with the venv active
# so you can drop the `poetry run` prefix:
poetry -C /home/ali/Projects/ros2_collection/so101_ros2 env activate
# (Poetry 2.x — for older Poetry, use `poetry shell` instead)
```

After that, just `python -m so101_ros2.dashboard` works.

---

## Troubleshooting

**`ros2: command not found`** — source ROS2: `source /opt/ros/$(ls /opt/ros)/setup.bash`
(Linux apt) or activate your RoboStack env (macOS).

**`ImportError: No module named rclpy` from inside `poetry run`** — either
(a) you forgot to source ROS2 before `poetry run`, so `PYTHONPATH` doesn't
include `/opt/ros/<distro>/lib/python<X.Y>/site-packages`; or (b) your venv
Python version doesn't match the distro's Python (e.g. a 3.11 venv against
Jazzy's 3.12 build). Fix (b) with `poetry env use /usr/bin/python3.12 && poetry install`.

**`poetry install` is hung resolving torch** — LeRobot's `[so101]` extra
pulls a heavy stack. Comment out the `lerobot = { path = ... }` line in
`pyproject.toml` and run `poetry install` first, then
`poetry run pip install -e "..[so101]"` to let pip resolve LeRobot directly.

**`ImportError: numpy.core.multiarray failed to import` from cv_bridge** —
your cv_bridge was built against NumPy 1.x but the venv has NumPy 2.x
(LeRobot pulls `rerun-sdk>=0.24` which requires NumPy ≥2, so we don't pin
it). Fix one of:
* RoboStack: `mamba update -c robostack ros-$ROS_DISTRO-cv-bridge` so it picks up the NumPy-2-compatible build.
* Force NumPy 1 in the venv (loses `rerun` viz, keeps everything else): `poetry run pip install "numpy<2"`.

**Dashboard says "PySide6 not found"** — you ran the script outside the
Poetry venv. Use `poetry run so101-dashboard` (or
`poetry env activate` first).

**Script hangs at "Waiting for camera streams"** — check publishers are alive
in the dashboard, or `ros2 topic list` and `ros2 topic hz <topic>`. Topic
names must match what's in `camera_config.json`.

**Camera index changed after replug** — re-run `configure_cameras`.

**`ConnectionError` on arm connect** — check both ports are detected
(`ls /dev/ttyACM*` on Linux, `ls /dev/cu.usbmodem*` on macOS), then make sure
your user can read/write them: `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1`,
or add yourself to the `dialout` group permanently:
`sudo usermod -aG dialout $USER` (logout/login to apply).
Make sure calibration files exist:
`ls ~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/`.

**`ValueError: Magnitude X exceeds 2047` during teleop** — the arm wasn't
centred at the middle-position prompt during calibration. Re-run
`lerobot-calibrate`.

**Image colours look wrong (R/B swapped)** — `camera_publisher.py` already
converts BGR→RGB. If they're still wrong, check `desired_encoding` in
`camera_streams.py`.

**Dataset folder exists but is from an old config** —
`rm -rf ~/lerobot_datasets/<DATASET_REPO_ID>` and re-run.

**Frames come in but FPS = 0** — the publisher is alive but its `header.stamp`
is far behind `time.time_ns()`; check the publisher's clock or restart the
publisher process.


---

## Layout

```
so101_ros2/
├── README.md                 ← you are here
├── pyproject.toml            ← Poetry deps (package-mode = false)
├── poetry.toml               ← project-local Poetry config (in-project venv)
├── poetry.lock               ← generated by `poetry install` (commit this)
├── .venv/                    ← Poetry-managed venv lives here (gitignored)
├── camera_config.json        ← generated, machine-specific (gitignored)
├── __init__.py
├── settings.py               ← edit ports / dataset paths here
├── camera_streams.py         ← shared rclpy bridge
├── camera_publisher.py
├── configure_cameras.py
├── data_collector.py
├── dashboard.py
└── docs/
    ├── ROS2_SETUP.md         ← deeper reference (was written for conda/RoboStack)
    └── SO101_GUIDE.md        ← deeper reference (was written for conda/RoboStack)
```

> The two files in `docs/` predate the Poetry switch and still describe the
> conda/RoboStack flow. They remain useful for the **macOS** RoboStack path
> and for the original dataset-format walkthrough — read them through that
> lens; the canonical workflow is this README.

---

## Reference docs

* [`docs/ROS2_SETUP.md`](docs/ROS2_SETUP.md) — detailed ROS2 install (macOS RoboStack, USB camera driver, RealSense).
* [`docs/SO101_GUIDE.md`](docs/SO101_GUIDE.md) — original end-to-end macOS guide (env setup, calibration, training).
* [`../CLAUDE.md`](../CLAUDE.md) — project-level pointers for working in this repo.
