#!/usr/bin/env python3
"""
Publishes a single USB/UVC camera as a ROS2 Image topic using OpenCV + AVFoundation.

Works on macOS (no V4L2 required). Use one instance per camera.

Usage:
  python opencv_camera_publisher.py --index 0 --topic /arducam/shoulder/image_raw
  python opencv_camera_publisher.py --index 1 --topic /arducam/wrist/image_raw
  python opencv_camera_publisher.py --index 3 --topic /camera/color/image_raw
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


WIDTH  = 640
HEIGHT = 480
FPS    = 30


class OpenCVCameraPublisher(Node):
    def __init__(self, index: int, topic: str):
        node_name = topic.strip("/").replace("/", "_")
        super().__init__(node_name)

        self._pub = self.create_publisher(Image, topic, 10)

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open camera index {index}")
            sys.exit(1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, FPS)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"Camera {index} → {topic}  (native {actual_w}×{actual_h}, publishing {WIDTH}×{HEIGHT})"
        )

        self._topic = topic
        # Warm up: read a few frames so the first publish isn't a stale buffer frame.
        for _ in range(5):
            self._cap.read()

        self._timer = self.create_timer(1.0 / FPS, self._publish)

    def _publish(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame", throttle_duration_sec=2.0)
            return

        # Resize to target resolution if the camera doesn't support it natively.
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # OpenCV gives BGR; convert to RGB for ROS convention.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._topic.strip("/").replace("/", "_")
        msg.height   = HEIGHT
        msg.width    = WIDTH
        msg.encoding = "rgb8"
        msg.step     = WIDTH * 3
        msg.data     = frame_rgb.tobytes()
        self._pub.publish(msg)

    def destroy_node(self):
        self._cap.release()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="OpenCV → ROS2 camera publisher (macOS)")
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
