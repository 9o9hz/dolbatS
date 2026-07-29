#!/usr/bin/env python3
import argparse
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension


BBox = Tuple[float, float, float, float]


def get_default_model_path() -> str:
    try:
        return os.path.join(
            get_package_share_directory("detect_pkg"),
            "config",
            "dolsoi-model-v2.pt",
        )
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "dolsoi-model-v2.pt"
        )


class ObstacleDetectorPublisher(Node):
    def __init__(
        self,
        show_window: Optional[bool] = None,
    ) -> None:
        super().__init__("obstacle_detector_publisher")

        self.declare_parameter("model_path", get_default_model_path())
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("enable_topic", "/detect/obstacle/enable")
        self.declare_parameter("enabled_at_startup", False)
        self.declare_parameter("detected_topic", "/detect/obstacle/detected")
        self.declare_parameter("bbox_topic", "/detect/obstacle/bbox")
        self.declare_parameter(
            "compressed_image_topic", "/image_raw/compressed"
        )
        self.declare_parameter(
            "detection_image_topic", "/camera/lane/detection_view"
        )
        self.declare_parameter("show_window", False if show_window is None else show_window)

        self.model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold").get_parameter_value().double_value
        )
        self.show_window = (
            self.get_parameter("show_window").get_parameter_value().bool_value
        )
        self.enabled = bool(self.get_parameter("enabled_at_startup").value)

        enable_topic = str(self.get_parameter("enable_topic").value)
        detected_topic = (
            self.get_parameter("detected_topic").get_parameter_value().string_value
        )
        bbox_topic = self.get_parameter("bbox_topic").get_parameter_value().string_value
        compressed_image_topic = (
            self.get_parameter("compressed_image_topic")
            .get_parameter_value()
            .string_value
        )
        detection_image_topic = (
            self.get_parameter("detection_image_topic")
            .get_parameter_value()
            .string_value
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, bbox_topic, 10)
        self.detection_image_pub = self.create_publisher(
            Image, detection_image_topic, 10
        )
        self.logged_first_frame = False

        self.get_logger().info(f"Loading model: {self.model_path}")
        YOLO = self.load_yolo()
        self.model = YOLO(self.model_path)

        if self.show_window:
            self.get_logger().info("Debug window enabled")

        self.subscription = self.create_subscription(
            CompressedImage, compressed_image_topic, self.process_frame, 10
        )
        self.enable_subscription = self.create_subscription(
            Bool, enable_topic, self.on_enable, 10
        )
        self.get_logger().info(
            f"YOLO enabled by {enable_topic} (startup={self.enabled}); "
            f"subscribing to {compressed_image_topic}=CompressedImage; publishing "
            f"{detected_topic}=Bool and {bbox_topic}=[cx, cy, w, h]"
        )

    def on_enable(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested == self.enabled:
            return
        self.enabled = requested
        self.get_logger().info(
            f"YOLO obstacle detection {'enabled' if requested else 'disabled'}"
        )
        if not requested:
            self.publish_detected(False)

    def load_yolo(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import ultralytics. Install compatible Python packages with "
                "`python3 -m pip install -r requirements.txt` from the "
                "workspace root, then rebuild with `colcon build`."
            ) from exc

        return YOLO

    def process_frame(self, image_msg: CompressedImage) -> None:
        if not self.enabled:
            return
        frame = self.image_message_to_bgr(image_msg)
        if frame is None:
            self.publish_detected(False)
            return

        if not self.logged_first_frame:
            height, width = frame.shape[:2]
            self.get_logger().info(f"First subscribed frame: {width}x{height}")
            self.logged_first_frame = True

        bbox = self.detect_best_bbox(frame)
        detected = bbox is not None
        self.publish_detected(detected)

        if detected:
            self.publish_bbox(bbox)

        detection_frame = frame.copy()
        self.draw_detection_overlay(detection_frame, bbox)
        self.publish_image(detection_frame, image_msg)

        if self.show_window:
            self.draw_debug_window(detection_frame)

    def detect_best_bbox(self, frame) -> Optional[BBox]:
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        best_box = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf <= best_conf:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2.0
                center_y = y1 + height / 2.0

                best_conf = conf
                best_box = (center_x, center_y, width, height)

        return best_box

    def publish_detected(self, detected: bool) -> None:
        msg = Bool()
        msg.data = detected
        self.detected_pub.publish(msg)

    def publish_bbox(self, bbox: BBox) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="bbox", size=4, stride=4),
        ]
        msg.data = [float(value) for value in bbox]
        self.bbox_pub.publish(msg)

    def image_message_to_bgr(self, msg: CompressedImage):
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error(
                f"Failed to decode compressed image (format={msg.format!r})"
            )
            return None
        return frame

    def publish_image(self, frame, source_msg: CompressedImage) -> None:
        msg = Image()
        msg.header = source_msg.header
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(frame.strides[0])
        msg.data = frame.tobytes()
        self.detection_image_pub.publish(msg)

    def draw_detection_overlay(self, frame, bbox: Optional[BBox]) -> None:
        if bbox is not None:
            center_x, center_y, width, height = bbox
            x1 = int(center_x - width / 2.0)
            y1 = int(center_y - height / 2.0)
            x2 = int(center_x + width / 2.0)
            y2 = int(center_y + height / 2.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)

    def draw_debug_window(self, frame) -> None:
        cv2.imshow("obstacle detector", frame)
        cv2.waitKey(1)

    def destroy_node(self) -> bool:
        if self.show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description="Detect obstacles from subscribed camera frames."
    )
    parser.add_argument(
        "--debug-window",
        "--show-window",
        dest="show_window",
        action="store_true",
        default=None,
        help="Show an OpenCV debug window with the camera frame and detected box.",
    )
    return parser.parse_known_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli_args, ros_args = parse_args(argv)
    rclpy.init(args=ros_args)
    node = None

    try:
        node = ObstacleDetectorPublisher(
            show_window=cli_args.show_window,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
