#!/usr/bin/env python3
"""ROS 2 node: camera image -> traffic light color, published for a future
decision node to subscribe to. Structured after
``obstacle_detector_publisher.py`` in this same package."""

import argparse
import os
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, String


DEFAULT_MODEL_FILENAME = "traffic_light_best.pt"
COLOR_LABELS = ("red", "yellow", "green")


def get_default_model_path() -> str:
    try:
        return os.path.join(
            get_package_share_directory("detect_pkg"),
            "config",
            DEFAULT_MODEL_FILENAME,
        )
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", DEFAULT_MODEL_FILENAME
        )


def normalize_color_label(class_name: str) -> Optional[str]:
    """Map a trained class name (e.g. "red_light", "Red", "green") onto
    one of the fixed ``COLOR_LABELS``, regardless of the exact naming used
    when the model was trained."""

    lowered = class_name.strip().lower()
    for label in COLOR_LABELS:
        if label in lowered:
            return label
    return None


class TrafficLightDetectorPublisher(Node):
    def __init__(
        self,
        show_window: Optional[bool] = None,
    ) -> None:
        super().__init__("traffic_light_detector_publisher")

        self.declare_parameter("model_path", get_default_model_path())
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("detected_topic", "/detect/traffic_light/detected")
        self.declare_parameter("color_topic", "/detect/traffic_light/color")
        self.declare_parameter(
            "confidence_topic", "/detect/traffic_light/confidence"
        )
        self.declare_parameter("raw_image_topic", "/camera/traffic_light/raw")
        self.declare_parameter(
            "detection_image_topic", "/detect/traffic_light/detection_view"
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

        detected_topic = (
            self.get_parameter("detected_topic").get_parameter_value().string_value
        )
        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        confidence_topic = (
            self.get_parameter("confidence_topic").get_parameter_value().string_value
        )
        raw_image_topic = (
            self.get_parameter("raw_image_topic").get_parameter_value().string_value
        )
        detection_image_topic = (
            self.get_parameter("detection_image_topic")
            .get_parameter_value()
            .string_value
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.color_pub = self.create_publisher(String, color_topic, 10)
        self.confidence_pub = self.create_publisher(
            Float32, confidence_topic, 10
        )
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
            Image, raw_image_topic, self.process_frame, 10
        )
        self.get_logger().info(
            f"Subscribing to {raw_image_topic}=Image; publishing "
            f"{detected_topic}=Bool, {color_topic}=String(red/yellow/green/none), "
            f"{confidence_topic}=Float32, "
            f"{detection_image_topic}=Image"
        )

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

    def process_frame(self, image_msg: Image) -> None:
        frame = self.image_message_to_bgr(image_msg)
        if frame is None:
            self.publish_detected(False)
            self.publish_color(None)
            self.publish_confidence(0.0)
            return

        if not self.logged_first_frame:
            height, width = frame.shape[:2]
            self.get_logger().info(f"First subscribed frame: {width}x{height}")
            self.logged_first_frame = True

        color, bbox, confidence = self.detect_best_color(frame)
        detected = color is not None
        self.publish_detected(detected)
        self.publish_color(color)
        self.publish_confidence(confidence)

        detection_frame = frame.copy()
        self.draw_detection_overlay(detection_frame, color, bbox)
        self.publish_image(detection_frame, image_msg)

        if self.show_window:
            self.draw_debug_window(detection_frame)

    def detect_best_color(
        self, frame
    ) -> Tuple[
        Optional[str],
        Optional[Tuple[float, float, float, float]],
        float,
    ]:
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        best_color: Optional[str] = None
        best_bbox = None
        best_conf = -1.0

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf <= best_conf:
                    continue

                class_id = int(box.cls[0])
                class_name = str(result.names.get(class_id, ""))
                color = normalize_color_label(class_name)
                if color is None:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                best_conf = conf
                best_color = color
                best_bbox = (x1, y1, x2 - x1, y2 - y1)

        return best_color, best_bbox, max(0.0, best_conf)

    def publish_detected(self, detected: bool) -> None:
        msg = Bool()
        msg.data = detected
        self.detected_pub.publish(msg)

    def publish_color(self, color: Optional[str]) -> None:
        msg = String()
        msg.data = color if color is not None else "none"
        self.color_pub.publish(msg)

    def publish_confidence(self, confidence: float) -> None:
        self.confidence_pub.publish(Float32(data=float(confidence)))

    def image_message_to_bgr(self, msg: Image):
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
            return None

        channels = 3
        expected_row_bytes = msg.width * channels
        if msg.step < expected_row_bytes or len(msg.data) < msg.step * msg.height:
            self.get_logger().error("Invalid image data size or step")
            return None

        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        frame = rows[:, :expected_row_bytes].reshape(msg.height, msg.width, channels)
        if msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame.copy()

    def publish_image(self, frame, source_msg: Image) -> None:
        msg = Image()
        msg.header = source_msg.header
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(frame.strides[0])
        msg.data = frame.tobytes()
        self.detection_image_pub.publish(msg)

    def draw_detection_overlay(
        self,
        frame,
        color: Optional[str],
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> None:
        if bbox is None:
            return
        x, y, width, height = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        overlay_color = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
        }.get(color or "", (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), overlay_color, 2)
        cv2.putText(
            frame,
            color or "unknown",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            overlay_color,
            2,
        )

    def draw_debug_window(self, frame) -> None:
        cv2.imshow("traffic light detector", frame)
        cv2.waitKey(1)

    def destroy_node(self) -> bool:
        if self.show_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description="Detect traffic light color from subscribed camera frames."
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
        node = TrafficLightDetectorPublisher(
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
