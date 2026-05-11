"""Single USB camera → ROS2 Image publisher.

Run one process per ArduCam (do NOT point this at a RealSense — cv2 reads
RealSense `/dev/video*` nodes as generic UVC and may grab the IR/depth
endpoint instead of colour).

    poetry run so101-publisher --index 1 --topic /arducam/front/image_raw
    poetry run so101-publisher --index 2 --topic /arducam/wrist/image_raw

Captures via OpenCV (works on macOS via AVFoundation and on Linux via V4L2),
publishes RGB8 at IMAGE_W × IMAGE_H @ FPS from settings.py.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from so101_ros2.settings import FPS, IMAGE_H, IMAGE_W


class OpenCVCameraPublisher(Node):
    def __init__(self, index: int, topic: str):
        node_name = topic.strip("/").replace("/", "_")
        super().__init__(node_name)
        self._pub = self.create_publisher(Image, topic, 10)

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open camera index {index}")
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_H)
        self._cap.set(cv2.CAP_PROP_FPS, FPS)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"Camera {index} → {topic}  "
            f"(native {actual_w}×{actual_h}, publishing {IMAGE_W}×{IMAGE_H})"
        )
        self._topic = topic

        # Warm up: drop the first few frames so we don't publish a stale buffer.
        for _ in range(5):
            self._cap.read()

        self._timer = self.create_timer(1.0 / FPS, self._publish)

    def _publish(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame", throttle_duration_sec=2.0)
            return
        if frame.shape[1] != IMAGE_W or frame.shape[0] != IMAGE_H:
            frame = cv2.resize(frame, (IMAGE_W, IMAGE_H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._topic.strip("/").replace("/", "_")
        msg.height = IMAGE_H
        msg.width = IMAGE_W
        msg.encoding = "rgb8"
        msg.step = IMAGE_W * 3
        msg.data = frame_rgb.tobytes()
        self._pub.publish(msg)

    def destroy_node(self) -> None:
        self._cap.release()
        super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV → ROS2 single camera publisher")
    parser.add_argument("--index", type=int, required=True, help="OpenCV camera index")
    parser.add_argument("--topic", type=str, required=True, help="ROS2 topic to publish on")
    args = parser.parse_args()

    rclpy.init()
    node = OpenCVCameraPublisher(args.index, args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
