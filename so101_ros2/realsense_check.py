"""Standalone Intel RealSense D415 quality check.

Opens the D415 directly through the librealsense2 SDK (via the `pyrealsense2`
Python bindings). This is the cleanest path to see the *real* image quality
the D415 produces — `cv2.VideoCapture` reads the same camera as a generic
UVC webcam and bypasses the on-camera ISP, which gives noticeably worse
colour and may even pick up the IR / depth endpoint by mistake.

What you get:
  • Live colour preview in a Qt window.
  • Hardware info: model, serial, firmware, USB type (look for 'USB 3.X').
  • Live quality metrics: FPS, resolution, Laplacian-variance sharpness,
    mean brightness, clipping percentage.
  • Snapshot button — saves to /tmp/d415_<timestamp>.png.

Run:
    poetry run so101-realsense-check

Prereq:
    poetry add pyrealsense2
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    sys.stderr.write(
        "pyrealsense2 not installed in this venv.  Install with:\n"
        "    poetry add pyrealsense2\n"
    )
    sys.exit(1)

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    sys.stderr.write("PySide6 missing — run `poetry install`\n")
    sys.exit(1)


WIDTH, HEIGHT, FPS = 640, 480, 30
DISPLAY_W, DISPLAY_H = 800, 600


def device_info(device: rs.device) -> dict[str, str]:
    """Pull every supported camera_info field for a device."""
    out: dict[str, str] = {}
    for field in (
        rs.camera_info.name,
        rs.camera_info.serial_number,
        rs.camera_info.firmware_version,
        rs.camera_info.usb_type_descriptor,
        rs.camera_info.physical_port,
        rs.camera_info.product_id,
    ):
        try:
            if device.supports(field):
                out[str(field).split(".")[-1]] = device.get_info(field)
        except RuntimeError:
            pass
    return out


class D415QualityWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intel RealSense D415 — quality check")
        self.resize(900, 800)

        # Open the pipeline up-front so failure shows in the terminal, not silently.
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
        try:
            profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            QtWidgets.QMessageBox.critical(
                self, "RealSense error",
                f"Failed to start the colour stream:\n\n{e}\n\n"
                "Common causes:\n"
                "  • D415 not plugged in\n"
                "  • Plugged into a USB 2.0 port (needs USB 3.x for full quality)\n"
                "  • Another process holds the camera (rs-viewer, ros2 launch, …)"
            )
            sys.exit(1)
        info = device_info(profile.get_device())

        # ── header ──────────────────────────────────────────────────────────
        usb = info.get("usb_type_descriptor", "?")
        usb_warn = "  ⚠ USB 2.0 — move to USB 3.x port" if usb.startswith("2.") else ""
        header = QtWidgets.QLabel(
            f"<b>{info.get('name', 'RealSense')}</b><br>"
            f"<span style='color:#666;font-family:monospace'>"
            f"serial {info.get('serial_number','?')} &nbsp; "
            f"firmware {info.get('firmware_version','?')} &nbsp; "
            f"USB {usb}{usb_warn}</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)

        # ── viewport ────────────────────────────────────────────────────────
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(DISPLAY_W, DISPLAY_H)
        self.image_label.setStyleSheet("background-color: #111; color: #888;")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("waiting for first frame from D415 …")

        # ── metrics + buttons ───────────────────────────────────────────────
        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setStyleSheet("font-family: monospace; color: #222;")

        self.snapshot_btn = QtWidgets.QPushButton("Snapshot → /tmp")
        self.snapshot_btn.clicked.connect(self._snapshot)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.snapshot_btn)
        btn_row.addStretch(1)

        # ── layout ──────────────────────────────────────────────────────────
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(header)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.metrics_label)
        layout.addLayout(btn_row)

        # ── frame loop ──────────────────────────────────────────────────────
        self._last_frame: np.ndarray | None = None
        self._recent: list[float] = []
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # ──────────────────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # poll_for_frames() returns an empty composite if nothing's ready.
        # try_wait_for_frames blocks up to N ms — we use the latter to avoid
        # busy spinning if the pipeline is bursty.
        ok, frames = self.pipeline.try_wait_for_frames(timeout_ms=10)
        if not ok:
            return
        color = frames.get_color_frame()
        if not color:
            return

        rgb = np.asanyarray(color.get_data())   # already RGB
        self._last_frame = rgb

        now = time.time()
        self._recent.append(now)
        self._recent = [t for t in self._recent if now - t < 1.0]

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        clipping   = float(((gray <= 2).mean() + (gray >= 253).mean()) * 100)

        h, w = rgb.shape[:2]
        self.metrics_label.setText(
            f"FPS:        {len(self._recent):>6.0f}\n"
            f"Resolution: {w}×{h}\n"
            f"Sharpness:  {sharpness:>6.1f}   (rule of thumb: <100 soft, >200 sharp)\n"
            f"Brightness: {brightness:>6.1f}   (healthy 50–200)\n"
            f"Clipping:   {clipping:>5.2f} %  (over/under-exposed pixels)"
        )

        qimg = QtGui.QImage(rgb.tobytes(), w, h, w * 3,
                            QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pix)

    def _snapshot(self) -> None:
        if self._last_frame is None:
            QtWidgets.QMessageBox.information(self, "Snapshot", "No frame yet.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"/tmp/d415_{ts}.png"
        bgr = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        QtWidgets.QMessageBox.information(self, "Snapshot", f"Saved:\n{path}")

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().closeEvent(ev)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = D415QualityWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
