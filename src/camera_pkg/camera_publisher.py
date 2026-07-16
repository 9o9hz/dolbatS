#!/usr/bin/env python3
import argparse
from typing import Optional, Sequence, Tuple

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    """Read an OpenCV camera and publish raw BGR frames."""

    def __init__(
        self,
        node_name: str,
        default_camera_index: int,
        default_raw_topic: str,
        default_frame_id: str,
        camera_index: Optional[int] = None,
    ) -> None:
        super().__init__(node_name)

        self.declare_parameter(
            "camera_index",
            default_camera_index if camera_index is None else camera_index,
        )
        self.declare_parameter("timer_period", 0.03)
        self.declare_parameter("raw_image_topic", default_raw_topic)
        self.declare_parameter("frame_id", default_frame_id)

        self.camera_index = int(self.get_parameter("camera_index").value)
        timer_period = float(self.get_parameter("timer_period").value)
        raw_image_topic = str(self.get_parameter("raw_image_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.publisher = self.create_publisher(Image, raw_image_topic, 10)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        self.timer = self.create_timer(timer_period, self.publish_frame)
        self.logged_first_frame = False
        self.get_logger().info(
            f"Camera {self.camera_index} publishing BGR frames on {raw_image_topic}"
        )

    def publish_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warning("Failed to read camera frame")
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(frame.strides[0])
        msg.data = frame.tobytes()
        self.publisher.publish(msg)

        if not self.logged_first_frame:
            self.get_logger().info(f"First frame: {msg.width}x{msg.height}")
            self.logged_first_frame = True

    def destroy_node(self) -> bool:
        if hasattr(self, "cap"):
            self.cap.release()
        return super().destroy_node()


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(description="Publish raw OpenCV camera frames.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="OpenCV camera index. Overrides the camera_index parameter default.",
    )
    return parser.parse_known_args(argv)


def run_camera(
    node_name: str,
    default_camera_index: int,
    default_raw_topic: str,
    default_frame_id: str,
    argv: Optional[Sequence[str]] = None,
) -> None:
    cli_args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = CameraPublisher(
            node_name=node_name,
            default_camera_index=default_camera_index,
            default_raw_topic=default_raw_topic,
            default_frame_id=default_frame_id,
            camera_index=cli_args.camera_index,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def traffic_light_main(argv: Optional[Sequence[str]] = None) -> None:
    run_camera(
        node_name="traffic_light_camera_publisher",
        default_camera_index=0,
        default_raw_topic="/camera/traffic_light/raw",
        default_frame_id="traffic_light_camera",
        argv=argv,
    )


def lane_main(argv: Optional[Sequence[str]] = None) -> None:
    run_camera(
        node_name="lane_camera_publisher",
        default_camera_index=1,
        default_raw_topic="/camera/lane/raw",
        default_frame_id="lane_camera",
        argv=argv,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    traffic_light_main(argv)


if __name__ == "__main__":
    main()
