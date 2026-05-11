"""Interactive camera labeller — discover cameras and assign roles.

Two phases:

  1. **Top camera** is the Intel RealSense D415. Detected automatically via
     pyrealsense2 (no live preview needed — `so101-realsense-check` already
     covers that). The RealSense uses the realsense2_camera ROS2 driver,
     not cv2.

  2. **Front + wrist cameras** are UVC ArduCams. Each detected /dev/videoN
     capture endpoint opens a Qt live-video dialog. Press inside it:

        1  →  front     (workspace-level view)
        2  →  wrist     (gripper-mounted)
        s  →  skip      (unrelated webcam)
        q  →  quit early

Writes the result to camera_config.json next to this file. Partial configs
are valid: re-run any time cameras are replugged.

    poetry run so101-configure
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import cv2

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    sys.stderr.write(
        "PySide6 not found. Install it inside the Poetry env:\n"
        "    poetry install\n"
    )
    sys.exit(1)

from so101_ros2.settings import (
    CAMERA_ROLES,
    CONFIG_FILE,
    IMAGE_H,
    IMAGE_W,
    REQUIRED_ROLES,
)

DISPLAY_W   = 640
DISPLAY_H   = 480
PREVIEW_FPS = 30


def v4l2_card_name(idx: int) -> str:
    """Return the V4L2 'card name' for /dev/videoN, or empty string."""
    try:
        with open(f"/sys/class/video4linux/video{idx}/name") as f:
            return f.read().strip()
    except OSError:
        return ""


def is_video_capture_endpoint(idx: int) -> bool:
    """True if /dev/videoN has V4L2_CAP_VIDEO_CAPTURE (bit 0x1) in its Device Caps.

    Many USB cameras (ArduCam GS, RealSense, etc.) expose multiple /dev/video
    nodes per physical device — one for video capture, one for metadata, and
    sometimes IR / depth on top. Without this filter cv2 will happily open
    the metadata node and then hang for ~10 s waiting for frames that never
    arrive.
    """
    try:
        out = subprocess.run(
            ["v4l2-ctl", f"--device=/dev/video{idx}", "--all"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # v4l-utils not installed — can't filter; fall back to trying anyway.
        # Recommend `sudo apt install v4l-utils`.
        return True
    m = re.search(r"Device Caps\s*:\s*(0x[0-9a-fA-F]+)", out)
    if not m:
        return True                                 # parsing failed; try anyway
    return bool(int(m.group(1), 16) & 0x00000001)   # V4L2_CAP_VIDEO_CAPTURE


def probe_camera(idx: int) -> tuple[int, int] | None:
    """Quick openability + first-frame check. Returns native (w, h), or None.

    Bails out fast if the first read fails — without this, V4L2's default
    10 s select() timeout per read would freeze the scan when an endpoint
    can be opened but doesn't actually stream (USB conflict, wrong format).
    """
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)
    # Bound a single read to 1.5 s if the backend honours this property.
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
    # Bail early on the first read — no point re-trying 5x if the endpoint
    # is fundamentally broken; a single 1.5 s timeout is enough confirmation.
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    # Drain a couple of warmup frames now that we know the device streams.
    for _ in range(2):
        cap.read()
    ret, frame = cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if ret and frame is not None else None


class LiveCameraDialog(QtWidgets.QDialog):
    """Modal dialog showing live video from one UVC camera, blocks until labelled."""

    VALID_KEYS = ("1", "2", "s", "q")

    def __init__(self, idx: int, native_w: int, native_h: int):
        super().__init__()
        self.idx = idx
        self.choice: str | None = None

        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)

        self.setWindowTitle(f"Camera {idx} — press 1 / 2 / s / q")
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        instructions = QtWidgets.QLabel(
            f"<b>UVC camera at /dev/video{idx}</b> &nbsp; "
            f"<span style='color:#666'>native {native_w}×{native_h}, "
            f"capture {IMAGE_W}×{IMAGE_H}</span><br>"
            "What is this camera? Press one of:<br>"
            "<b>1</b> front (workspace) &nbsp; "
            "<b>2</b> wrist (gripper) &nbsp; "
            "<b>s</b> skip &nbsp; "
            "<b>q</b> quit"
        )
        instructions.setTextFormat(QtCore.Qt.TextFormat.RichText)
        instructions.setWordWrap(True)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.image_label.setStyleSheet("background-color: #111; color: #888;")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("opening camera…")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(instructions)
        layout.addWidget(self.image_label)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / PREVIEW_FPS))
        self.timer.timeout.connect(self._update_frame)
        self.timer.start()

    def _update_frame(self) -> None:
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(rgb, f"INDEX {self.idx}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.tobytes(), w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            DISPLAY_W, DISPLAY_H,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        key = ev.text().lower()
        if key in self.VALID_KEYS:
            self.choice = key
            self.accept()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(ev)


def detect_realsense() -> str | None:
    """Return the serial of the first RealSense D-series camera, or None.

    Uses pyrealsense2 if installed; otherwise falls back to scanning lsusb
    for 'RealSense' in the device name.
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        try:
            out = subprocess.run(
                ["lsusb"], capture_output=True, text=True, timeout=2,
            ).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return "lsusb" if "RealSense" in out else None
    ctx = rs.context()
    for dev in ctx.query_devices():
        try:
            return dev.get_info(rs.camera_info.serial_number)
        except RuntimeError:
            return "unknown"
    return None


def main() -> None:
    QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    config: dict = {}

    print()
    print("━" * 55)
    print("  SO101 Camera Configuration")
    print("━" * 55)

    # ── Phase 1: RealSense (top) ──────────────────────────────────────────────
    print()
    print("Phase 1: RealSense (top)")
    rs_serial = detect_realsense()
    if rs_serial:
        role = CAMERA_ROLES["0"]
        config[role["name"]] = {
            "index": None,                   # ROS2 driver, not /dev/video
            "topic": role["topic"],
            "label": role["label"],
            "description": role["description"],
            "realsense_serial": rs_serial,
        }
        print(f"  ✓ RealSense detected (serial {rs_serial}) → {role['topic']}")
        print("    (uses the realsense2_camera ROS2 driver — install once with:")
        print("       sudo apt install ros-$ROS_DISTRO-realsense2-camera )")
    else:
        print("  – No RealSense detected. Skipping the 'top' role.")
        print("    Verify with `poetry run so101-realsense-check` and re-run.")

    # ── Phase 2: UVC ArduCams (front + wrist) ─────────────────────────────────
    print()
    print("Phase 2: UVC ArduCams (front + wrist)")
    print("Scanning /dev/video* — for each detected camera a live preview opens.")
    print("Press a key inside the window to label it:")
    print("  1  →  front (workspace view)")
    print("  2  →  wrist (gripper-mounted)")
    print("  s  →  skip")
    print("  q  →  quit early")
    print()

    UVC_KEYS = {"1": CAMERA_ROLES["1"], "2": CAMERA_ROLES["2"]}
    aborted = False
    found_any = False
    for idx in range(16):
        # Skip nonexistent device nodes — calling VideoCapture on them just
        # produces noisy V4L2 stderr warnings.
        if not os.path.exists(f"/dev/video{idx}"):
            continue
        card = v4l2_card_name(idx)
        # Skip RealSense endpoints if one ever gets plugged back in — cv2
        # treats them as garbage UVC cams and shows IR/depth as 8-bit grey.
        if "RealSense" in card:
            print(f"  skipping /dev/video{idx}  ({card}) — not a regular UVC camera")
            continue
        # Many UVC cameras expose multiple /dev/video nodes per physical
        # device — typically one for video capture, one for metadata. Only
        # the capture endpoint will produce frames; the metadata one hangs.
        if not is_video_capture_endpoint(idx):
            print(f"  skipping /dev/video{idx}  ({card}) — metadata endpoint, not video capture")
            continue
        size = probe_camera(idx)
        if size is None:
            print(f"  skipping /dev/video{idx}  ({card}) — opens but produces no frames "
                  f"(USB bandwidth, busy device, or unsupported pixel format?)")
            continue
        found_any = True
        w, h = size
        label = f" — {card}" if card else ""
        print(f"Camera at /dev/video{idx}  ({w}×{h}){label} — opening live preview…")

        dlg = LiveCameraDialog(idx, w, h)
        dlg.exec()

        choice = dlg.choice
        if choice in UVC_KEYS:
            role = UVC_KEYS[choice]
            config[role["name"]] = {
                "index": idx,
                "topic": role["topic"],
                "label": role["label"],
                "description": role["description"],
            }
            print(f"  ✓ Camera {idx}: {role['description']}\n")
        elif choice == "s":
            print(f"  – Camera {idx}: skipped\n")
        elif choice == "q":
            print("  ⏹ quit pressed — stopping scan\n")
            aborted = True
            break
        else:
            print(f"  – Camera {idx}: closed without labelling — skipped\n")

    if not found_any:
        print("No UVC cameras detected. Make sure the cameras are plugged in.")
        return

    if not config:
        print("Nothing labelled. camera_config.json was not written.")
        return

    missing = REQUIRED_ROLES - config.keys()
    if missing and not aborted:
        print(f"Note: roles not yet assigned: {', '.join(sorted(missing))}")
        print("Re-run  poetry run so101-configure  to add them later.\n")

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    print("━" * 55)
    print(f"  Config saved → {CONFIG_FILE}")
    print("━" * 55)
    print()
    print("Camera mapping:")
    for name, info in config.items():
        idx_str = "ROS2 driver" if info["index"] is None else f"index {info['index']}"
        print(f"  {idx_str:>13s}  {name:7s}  {info['topic']}")

    print()
    print("Next — start the dashboard and click 'Start all publishers':")
    print("  poetry run so101-dashboard")
    print()


if __name__ == "__main__":
    main()
