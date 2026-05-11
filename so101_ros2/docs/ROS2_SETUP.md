# ROS2 Setup for ArduCam + RealSense + SO101 Data Collection

> **macOS Intel (x86_64) note:** ROS2 Humble has no official macOS binary.
> Install it via **RoboStack** (conda-forge), which provides pre-built ROS2
> packages for Intel Mac inside the existing `lerobot` conda environment.

---

## 1. Install ROS2 Humble via RoboStack

Run all of these in one terminal session with the **base** conda env active
(not `lerobot` yet):

```bash
# Add the required channels to the lerobot env
conda activate lerobot
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults

# Install ROS2 Humble + packages needed by the collector
conda install ros-humble-desktop \
              ros-humble-cv-bridge \
              ros-humble-message-filters
```

After this, activating `lerobot` also sets up ROS2 — no separate
`source setup.zsh` step is needed.

Verify:

```bash
conda activate lerobot
echo $ROS_DISTRO              # should print: humble
python -c "import rclpy; print('rclpy ok')"
```

---

## 2. Install USB camera driver for ArduCam

ArduCam cameras are standard UVC USB devices — no custom driver needed.
Install the ROS2 `usb_cam` package directly:

```bash
conda activate lerobot
conda install ros-humble-usb-cam
```

Verify:

```bash
ros2 pkg list | grep usb_cam   # should print: usb_cam
```

> Find which `/dev/videoN` index each physical camera maps to:
> ```bash
> lerobot-find-cameras opencv
> ```

---

## 3. Install Intel RealSense support

```bash
conda activate lerobot
conda install realsense2-camera   # RoboStack provides this package
```

Verify the RealSense is detected:

```bash
python -c "import pyrealsense2; print('realsense ok')"
```

---

## 4. Launch the camera nodes (4 terminals)

Open four terminals. In **each one**, activate the conda env first:

```bash
conda activate lerobot
```

### Terminal 1 — ArduCam shoulder

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args -r /image_raw:=/arducam/shoulder/image_raw -p video_device:=/dev/video0
```

### Terminal 2 — ArduCam wrist

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args -r /image_raw:=/arducam/wrist/image_raw -p video_device:=/dev/video2
```

> Run `lerobot-find-cameras opencv` to confirm which index maps to which
> physical camera, then use `/dev/videoN` where N is that index.

### Terminal 3 — Intel RealSense (RGB only)

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=false \
  enable_infra1:=false \
  enable_infra2:=false \
  color_width:=640 color_height:=480 color_fps:=30
```

### Terminal 4 — Data collector

```bash
cd ~/Projects/lerobot
python collect_data_ros2.py
```

---

## 5. Verify all three topics are publishing

```bash
conda activate lerobot
ros2 topic list
# Expected:
#   /arducam/shoulder/image_raw
#   /arducam/wrist/image_raw
#   /camera/color/image_raw

ros2 topic hz /arducam/shoulder/image_raw   # should show ~30 Hz
ros2 topic hz /camera/color/image_raw       # should show ~30 Hz
```

---

## Troubleshooting

**`ros2: command not found`**
```bash
conda activate lerobot   # RoboStack wires up ros2 on env activation
```

**Camera not found / wrong index:**
```bash
lerobot-find-cameras opencv
```

**Topics not showing up:**
```bash
ros2 topic list   # if empty, make sure the conda env is active
```

**Image encoding error (bgr8 vs rgb8):**
Edit `collect_data_ros2.py` and change `desired_encoding="rgb8"` to
`desired_encoding="bgr8"` in `CameraNode._on_frames()` for the affected camera.

**Wrong image resolution:**
Edit `IMAGE_H` and `IMAGE_W` in `collect_data_ros2.py`. Both cameras are
auto-resized to match, so all episodes stay consistent.
