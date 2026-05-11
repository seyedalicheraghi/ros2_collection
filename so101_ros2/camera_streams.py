"""Shared rclpy node — one Image subscription per camera, latest frame per slot.

Used by both `data_collector` (recording loop) and `dashboard` (Qt redraw).
Per-topic latest-frame model: a slow camera does not bottleneck fast ones.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from so101_ros2.settings import IMAGE_H, IMAGE_W


@dataclass
class CameraSlot:
    """Per-topic state held by CameraStreams."""
    label: str
    topic: str
    frame: np.ndarray | None = None        # Latest RGB frame
    stamp_ns: int = 0                      # msg.header.stamp in nanoseconds
    recv_times: deque = field(default_factory=lambda: deque(maxlen=120))


class CameraStreams(Node):
    """Subscribes to N camera topics and exposes their latest frames.

    Args:
        cameras: list of {"label": str, "topic": str} dicts.
        resize:  when True, every incoming frame is resized to (IMAGE_H, IMAGE_W).
                 Use True for recording (dataset consistency); False for the
                 dashboard (so you see the actual published resolution).
    """

    def __init__(self, cameras: list[dict], *, resize: bool = True):
        super().__init__("so101_camera_streams")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._resize = resize
        self._slots: dict[str, CameraSlot] = {}
        for c in cameras:
            slot = CameraSlot(label=c["label"], topic=c["topic"])
            self._slots[slot.label] = slot
            self.create_subscription(
                Image, slot.topic,
                lambda msg, lbl=slot.label: self._on_image(msg, lbl),
                10,
            )
        self.get_logger().info(f"Subscribed to {len(cameras)} camera topic(s).")

    def _on_image(self, msg: Image, label: str) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().warn(
                f"{label}: cv_bridge error: {e}", throttle_duration_sec=2.0,
            )
            return
        if self._resize and frame.shape[:2] != (IMAGE_H, IMAGE_W):
            frame = cv2.resize(frame, (IMAGE_W, IMAGE_H))
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        with self._lock:
            slot = self._slots[label]
            slot.frame = frame
            slot.stamp_ns = stamp_ns
            slot.recv_times.append(time.time())

    # ── read APIs ────────────────────────────────────────────────────────────

    def get_frames(self) -> dict[str, np.ndarray | None]:
        """{label → latest frame} (no copy). Used by the recording loop."""
        with self._lock:
            return {lbl: s.frame for lbl, s in self._slots.items()}

    def get_slots(self) -> dict[str, CameraSlot]:
        """{label → copy of slot}. Used by the dashboard for metric computation."""
        with self._lock:
            return {
                lbl: CameraSlot(
                    label=s.label,
                    topic=s.topic,
                    frame=None if s.frame is None else s.frame.copy(),
                    stamp_ns=s.stamp_ns,
                    recv_times=deque(s.recv_times, maxlen=120),
                )
                for lbl, s in self._slots.items()
            }

    @property
    def ready(self) -> bool:
        with self._lock:
            return all(s.frame is not None for s in self._slots.values())


def spin_in_background(node: Node) -> tuple[rclpy.executors.SingleThreadedExecutor, threading.Thread]:
    """Common rclpy setup: SingleThreadedExecutor on a daemon thread.

    Returns (executor, thread) so callers can shut them down on exit.
    """
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread
