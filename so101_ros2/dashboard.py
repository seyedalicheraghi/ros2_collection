"""Unified Qt GUI for camera inspection and publisher management.

A single window that:
  • starts / stops each camera_publisher subprocess from camera_config.json,
  • subscribes to the resulting ROS2 image topics via CameraStreams,
  • shows a live preview + numeric quality metrics per camera,
  • lets you snapshot a single camera or all of them at once,
  • shows an RGB histogram for exposure inspection.

Quality metrics (computed live, per-camera):
  FPS         — frames received in the last 1.0 s wall-clock window
  Resolution  — width × height of the incoming frame
  Latency     — now() - msg.header.stamp, in ms (publisher → receive)
  Sharpness   — variance of the Laplacian (rule of thumb: <100 soft, >200 sharp)
  Brightness  — mean grayscale intensity (0–255)
  Clipping    — % of pixels at 0 or 255 (under- / over-exposed)

Run:
    poetry run so101-dashboard
"""

from __future__ import annotations

import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    sys.stderr.write(
        "PySide6 not found.  Install it inside the lerobot env:\n"
        "    pip install PySide6\n"
        "If you hit Qt plugin conflicts with RoboStack, fall back to PyQt5\n"
        "and replace the PySide6 imports.\n"
    )
    sys.exit(1)

from so101_ros2.camera_streams import CameraSlot, CameraStreams, spin_in_background
from so101_ros2.settings import (
    PUBLISHER_MODULE,
    REPO_ROOT,
    SNAPSHOT_DIR,
    load_camera_config,
    realsense_launch_cmd,
)

DISPLAY_FPS = 30
DISPLAY_W, DISPLAY_H = 480, 360


# ── Subprocess management ─────────────────────────────────────────────────────

class PublisherProcess:
    """Wraps the publisher subprocess for one camera.

    Routing:
      * role == "top"  →  `ros2 launch realsense2_camera rs_launch.py …`
        (uses librealsense2's ISP — needed for correct D415 colour).
      * everything else  →  `python -m so101_ros2.camera_publisher …`
        (generic UVC capture via OpenCV, fine for ArduCams).
    """

    def __init__(self, name: str, index: int | None, topic: str):
        self.name = name
        self.index = index             # None for the RealSense top
        self.topic = topic
        self.is_realsense = (name == "top")
        self.proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _build_command(self) -> list[str]:
        if self.is_realsense:
            return realsense_launch_cmd()
        return [
            sys.executable, "-m", PUBLISHER_MODULE,
            "--index", str(self.index), "--topic", self.topic,
        ]

    def start(self) -> None:
        if self.is_running():
            return
        # cwd=REPO_ROOT so `python -m so101_ros2.camera_publisher` resolves
        # regardless of where the dashboard was launched from.
        # stdout/stderr deliberately *not* suppressed: launch failures
        # (driver missing, wrong topic name, USB error) are otherwise
        # invisible — the panel just stays at "stopped" with no clue why.
        # The cost is some log noise in the dashboard's terminal.
        print(f"[so101_ros2] launching publisher for '{self.name}': "
              f"{' '.join(self._build_command())}", flush=True)
        self.proc = subprocess.Popen(
            self._build_command(),
            cwd=str(REPO_ROOT),
            start_new_session=True,
        )

    def stop(self) -> None:
        if not self.is_running():
            self.proc = None
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            self.proc = None


# ── Qt widgets ────────────────────────────────────────────────────────────────

class CameraPanel(QtWidgets.QFrame):
    """Live preview + quality metrics + per-camera controls for a single camera."""

    def __init__(self, name: str, label: str, topic: str, publisher: PublisherProcess):
        super().__init__()
        self.name = name
        self.label = label
        self.topic = topic
        self.publisher = publisher
        self.paused = False
        self._last_frame: np.ndarray | None = None

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { padding: 6px; }")

        source = (
            "ROS2 driver (realsense2_camera)"
            if publisher.is_realsense
            else f"index {publisher.index}"
        )
        header = QtWidgets.QLabel(
            f"<b>{name}</b> &nbsp;&nbsp; "
            f"<span style='color:#666'>{label}</span><br>"
            f"<span style='color:#888;font-family:monospace'>"
            f"{source} → {topic}</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setFixedSize(DISPLAY_W, DISPLAY_H)
        self.image_label.setStyleSheet("background-color: #111; color: #888;")
        self.image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("waiting for frames…")

        self.metrics_label = QtWidgets.QLabel()
        self.metrics_label.setStyleSheet("font-family: monospace; color: #222;")
        self.metrics_label.setText(
            self._format_metrics(None, None, None, None, None, None)
        )

        self.start_btn = QtWidgets.QPushButton("Start publisher")
        self.start_btn.clicked.connect(self._start_publisher)
        self.stop_btn = QtWidgets.QPushButton("Stop publisher")
        self.stop_btn.clicked.connect(self._stop_publisher)
        self.stop_btn.setEnabled(False)
        self.pub_status = QtWidgets.QLabel("●  stopped")
        self.pub_status.setStyleSheet("color: #888;")

        pub_row = QtWidgets.QHBoxLayout()
        pub_row.addWidget(self.pub_status)
        pub_row.addWidget(self.start_btn)
        pub_row.addWidget(self.stop_btn)
        pub_row.addStretch(1)

        self.pause_btn = QtWidgets.QPushButton("Pause view")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._toggle_pause)
        self.snapshot_btn = QtWidgets.QPushButton("Snapshot")
        self.snapshot_btn.clicked.connect(self._save_snapshot)
        self.histogram_btn = QtWidgets.QPushButton("Histogram")
        self.histogram_btn.clicked.connect(self._show_histogram)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(self.pause_btn)
        view_row.addWidget(self.snapshot_btn)
        view_row.addWidget(self.histogram_btn)
        view_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.image_label)
        layout.addWidget(self.metrics_label)
        layout.addLayout(pub_row)
        layout.addLayout(view_row)

    # --- per-tick refresh -----------------------------------------------------

    def update_from_slot(self, slot: CameraSlot) -> None:
        self._refresh_publisher_status()

        if slot.frame is None:
            return
        self._last_frame = slot.frame
        if self.paused:
            return

        now = time.time()
        recent = [t for t in slot.recv_times if now - t < 1.0]
        fps = len(recent)
        latency_ms = (time.time_ns() - slot.stamp_ns) / 1e6 if slot.stamp_ns else None

        gray = cv2.cvtColor(slot.frame, cv2.COLOR_RGB2GRAY)
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        clipping   = float(((gray <= 2).mean() + (gray >= 253).mean()) * 100)

        h, w = slot.frame.shape[:2]
        self.metrics_label.setText(
            self._format_metrics(fps, (w, h), latency_ms, sharpness, brightness, clipping)
        )

        disp = cv2.resize(slot.frame, (DISPLAY_W, DISPLAY_H))
        qimg = QtGui.QImage(
            disp.tobytes(), disp.shape[1], disp.shape[0], disp.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _format_metrics(fps, res, latency_ms, sharpness, brightness, clipping) -> str:
        def fmt(v, suffix=""):
            return "—" if v is None else f"{v:>7.1f}{suffix}"
        res_str = "—" if res is None else f"{res[0]}×{res[1]}"
        return (
            f"FPS:        {fmt(fps)}\n"
            f"Resolution: {res_str:>13}\n"
            f"Latency:    {fmt(latency_ms, ' ms')}\n"
            f"Sharpness:  {fmt(sharpness)}\n"
            f"Brightness: {fmt(brightness)}\n"
            f"Clipping:   {fmt(clipping, ' %')}"
        )

    def _refresh_publisher_status(self) -> None:
        running = self.publisher.is_running()
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.pub_status.setText(f"●  running  (pid {self.publisher.proc.pid})")
            self.pub_status.setStyleSheet("color: #2a7;")
        else:
            self.pub_status.setText("●  stopped")
            self.pub_status.setStyleSheet("color: #888;")

    # --- buttons --------------------------------------------------------------

    def _start_publisher(self) -> None:
        self.publisher.start()
        self._refresh_publisher_status()

    def _stop_publisher(self) -> None:
        self.publisher.stop()
        self._refresh_publisher_status()

    def _toggle_pause(self, on: bool) -> None:
        self.paused = on
        self.pause_btn.setText("Resume view" if on else "Pause view")

    def _save_snapshot(self) -> None:
        if self._last_frame is None:
            QtWidgets.QMessageBox.information(self, "Snapshot", "No frame yet.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = SNAPSHOT_DIR / ts
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.name}.png"
        bgr = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        QtWidgets.QMessageBox.information(self, "Snapshot", f"Saved:\n{path}")

    def _show_histogram(self) -> None:
        if self._last_frame is None:
            return
        canvas = self._render_histogram(self._last_frame)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Histogram — {self.name}")
        lab = QtWidgets.QLabel(dlg)
        qimg = QtGui.QImage(
            canvas.tobytes(), canvas.shape[1], canvas.shape[0], canvas.strides[0],
            QtGui.QImage.Format.Format_RGB888,
        )
        lab.setPixmap(QtGui.QPixmap.fromImage(qimg))
        QtWidgets.QVBoxLayout(dlg).addWidget(lab)
        dlg.show()

    @staticmethod
    def _render_histogram(rgb: np.ndarray) -> np.ndarray:
        h, w = 220, 520
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
        colors = [(255, 80, 80), (80, 220, 80), (80, 130, 255)]  # R, G, B
        for ch in range(3):
            hist = cv2.calcHist([rgb], [ch], None, [256], [0, 256]).flatten()
            hist = (hist / max(hist.max(), 1) * (h - 20)).astype(np.int32)
            for x in range(256):
                cv2.line(
                    canvas,
                    (4 + x * 2, h - 4),
                    (4 + x * 2, h - 4 - int(hist[x])),
                    colors[ch], 1,
                )
        return canvas


class SensorDashboard(QtWidgets.QMainWindow):
    def __init__(
        self,
        streams: CameraStreams,
        cameras: list[dict],
        publishers: dict[str, PublisherProcess],
    ):
        super().__init__()
        self.streams = streams
        self.publishers = publishers
        self.setWindowTitle("SO101 Sensor Dashboard")
        self.resize(1200, 920)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.panels: dict[str, CameraPanel] = {}
        for i, c in enumerate(cameras):
            panel = CameraPanel(
                name=c["name"], label=c["label"], topic=c["topic"],
                publisher=publishers[c["name"]],
            )
            grid.addWidget(panel, i // 2, i % 2)
            self.panels[c["label"]] = panel

        toolbar = self.addToolBar("Main")
        start_all = QtGui.QAction("Start all publishers", self)
        start_all.triggered.connect(self._start_all)
        toolbar.addAction(start_all)
        stop_all = QtGui.QAction("Stop all publishers", self)
        stop_all.triggered.connect(self._stop_all)
        toolbar.addAction(stop_all)
        toolbar.addSeparator()
        snap_all = QtGui.QAction("Snapshot all", self)
        snap_all.triggered.connect(self._snapshot_all)
        toolbar.addAction(snap_all)

        self.status_label = QtWidgets.QLabel("Connecting…")
        self.statusBar().addWidget(self.status_label)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(int(1000 / DISPLAY_FPS))
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        slots = self.streams.get_slots()
        active = 0
        for label, panel in self.panels.items():
            slot = slots.get(label)
            if slot is None:
                continue
            if slot.frame is not None:
                active += 1
            panel.update_from_slot(slot)
        running = sum(1 for p in self.publishers.values() if p.is_running())
        self.status_label.setText(
            f"Streaming: {active}/{len(self.panels)} cameras  ·  "
            f"Publishers running: {running}/{len(self.publishers)}"
        )

    def _start_all(self) -> None:
        for p in self.publishers.values():
            p.start()

    def _stop_all(self) -> None:
        for p in self.publishers.values():
            p.stop()

    def _snapshot_all(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = SNAPSHOT_DIR / ts
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for panel in self.panels.values():
            if panel._last_frame is not None:
                bgr = cv2.cvtColor(panel._last_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / f"{panel.name}.png"), bgr)
                n += 1
        QtWidgets.QMessageBox.information(
            self, "Snapshot all", f"Saved {n} image(s) to:\n{out}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for p in self.publishers.values():
            p.stop()
        super().closeEvent(event)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    cam_cfg = load_camera_config(require_all=False)
    cameras: list[dict] = []
    for name, info in cam_cfg.items():
        cameras.append({
            "name":  name,
            "index": int(info["index"]) if info.get("index") is not None else None,
            "topic": info["topic"],
            "label": info["label"],
        })
    print(f"Loaded {len(cameras)} cameras from camera_config.json")
    for c in cameras:
        print(f"  {c['name']:10s}  index {c['index']}  {c['topic']}  →  {c['label']}")

    publishers = {
        c["name"]: PublisherProcess(name=c["name"], index=c["index"], topic=c["topic"])
        for c in cameras
    }

    rclpy.init()
    streams = CameraStreams(
        [{"label": c["label"], "topic": c["topic"]} for c in cameras],
        resize=False,  # show actual published resolution in the metrics panel
    )
    executor, _ = spin_in_background(streams)

    app = QtWidgets.QApplication(sys.argv)
    win = SensorDashboard(streams, cameras, publishers)
    win.show()
    try:
        ret = app.exec()
    finally:
        for p in publishers.values():
            p.stop()
        streams.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()
