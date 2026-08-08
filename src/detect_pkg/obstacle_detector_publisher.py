#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import (
    Bool,
    Float32,
    Float32MultiArray,
    Int8MultiArray,
    MultiArrayDimension,
    String,
)


BBox = Tuple[float, float, float, float]
Detection = Tuple[BBox, float]
DEFAULT_MODEL_FILENAME = "dolsoi-model-v2.pt"
EVENT_DETECTED = 1
DIRECTION_RIGHT = -1
DIRECTION_NONE = 0
DIRECTION_LEFT = 1


def normalized_overlay_geometry(
    image_width: int,
    image_height: int,
    roi_left_ratio: float,
    roi_right_ratio: float,
    roi_top_ratio: float,
    roi_bottom_ratio: float,
    exit_left_ratio: float,
    exit_right_ratio: float,
) -> Dict[str, int]:
    """Convert normalized YOLO-only guides to drawable pixel coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (
        0.0 <= roi_left_ratio < roi_right_ratio <= 1.0
        and 0.0 <= roi_top_ratio < roi_bottom_ratio <= 1.0
        and 0.0 <= exit_left_ratio < exit_right_ratio <= 1.0
    ):
        raise ValueError("overlay ratios must be ordered within [0, 1]")

    def pixel(ratio: float, size: int) -> int:
        return min(size - 1, max(0, int(round(size * ratio))))

    return {
        "roi_left": pixel(roi_left_ratio, image_width),
        "roi_right": pixel(roi_right_ratio, image_width),
        "roi_top": pixel(roi_top_ratio, image_height),
        "roi_bottom": pixel(roi_bottom_ratio, image_height),
        "exit_left": pixel(exit_left_ratio, image_width),
        "exit_right": pixel(exit_right_ratio, image_width),
    }


def get_model_path(model_filename: str) -> str:
    try:
        return os.path.join(
            get_package_share_directory("detect_pkg"),
            "config",
            model_filename,
        )
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", model_filename
        )


class ObstacleDetectorPublisher(Node):
    def __init__(
        self,
        show_window: Optional[bool] = None,
    ) -> None:
        super().__init__("obstacle_detector_publisher")

        self.declare_parameter("model_filename", DEFAULT_MODEL_FILENAME)
        self.declare_parameter("model_path", "")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("enable_topic", "/detect/obstacle/enable")
        self.declare_parameter("enabled_at_startup", False)
        self.declare_parameter("detected_topic", "/detect/obstacle/detected")
        self.declare_parameter("bbox_topic", "/detect/obstacle/bbox")
        self.declare_parameter(
            "ultrasonic_enable_topic",
            "/detect/obstacle/ultrasonic_enabled",
        )
        self.declare_parameter(
            "ultrasonic_event_topic", "/detect/obstacle_event"
        )
        self.declare_parameter(
            "avoidance_steer_topic",
            "/control/candidate/obstacle/steer_angle",
        )
        self.declare_parameter(
            "avoidance_valid_topic",
            "/control/candidate/obstacle/valid",
        )
        self.declare_parameter(
            "compressed_image_topic", "/image_raw/compressed"
        )
        self.declare_parameter(
            "detection_image_topic",
            "/camera/lane/detection_view/compressed",
        )
        self.declare_parameter("detection_jpeg_quality", 80)
        self.declare_parameter("show_window", False if show_window is None else show_window)
        self.declare_parameter("show_yolo_only_guides", False)
        self.declare_parameter(
            "avoidance_status_topic", "/detect/avoidance/status"
        )

        configured_model_path = str(self.get_parameter("model_path").value).strip()
        model_filename = str(self.get_parameter("model_filename").value).strip()
        if not configured_model_path and not model_filename:
            raise ValueError("model_filename must not be empty when model_path is unset")
        self.model_path = configured_model_path or get_model_path(model_filename)
        self.confidence_threshold = (
            self.get_parameter("confidence_threshold").get_parameter_value().double_value
        )
        self.show_window = (
            self.get_parameter("show_window").get_parameter_value().bool_value
        )
        self.detection_jpeg_quality = max(
            1,
            min(
                100,
                int(self.get_parameter("detection_jpeg_quality").value),
            ),
        )
        self.enabled = bool(self.get_parameter("enabled_at_startup").value)
        self.show_yolo_only_guides = bool(
            self.get_parameter("show_yolo_only_guides").value
        )

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
        avoidance_status_topic = str(
            self.get_parameter("avoidance_status_topic").value
        )
        ultrasonic_enable_topic = str(
            self.get_parameter("ultrasonic_enable_topic").value
        )
        ultrasonic_event_topic = str(
            self.get_parameter("ultrasonic_event_topic").value
        )
        avoidance_steer_topic = str(
            self.get_parameter("avoidance_steer_topic").value
        )
        avoidance_valid_topic = str(
            self.get_parameter("avoidance_valid_topic").value
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, bbox_topic, 10)
        self.detection_image_pub = self.create_publisher(
            CompressedImage, detection_image_topic, 10
        )
        self.logged_first_frame = False
        self.yolo_detected = False
        self.ultrasonic_enabled = False
        self.ultrasonic_detected: Optional[bool] = None
        self.avoid_direction = DIRECTION_NONE
        self.avoidance_steer_deg = 0.0
        self.avoidance_valid = False
        self.yolo_only_guide_config: Optional[Dict[str, object]] = None

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
        self.ultrasonic_enable_subscription = self.create_subscription(
            Bool,
            ultrasonic_enable_topic,
            self.on_ultrasonic_enable,
            10,
        )
        self.ultrasonic_event_subscription = self.create_subscription(
            Int8MultiArray,
            ultrasonic_event_topic,
            self.on_ultrasonic_event,
            10,
        )
        self.avoidance_steer_subscription = self.create_subscription(
            Float32,
            avoidance_steer_topic,
            self.on_avoidance_steer,
            10,
        )
        self.avoidance_valid_subscription = self.create_subscription(
            Bool,
            avoidance_valid_topic,
            self.on_avoidance_valid,
            10,
        )
        if self.show_yolo_only_guides:
            status_qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.avoidance_status_subscription = self.create_subscription(
                String,
                avoidance_status_topic,
                self.on_avoidance_status,
                status_qos,
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

    def on_ultrasonic_enable(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested != self.ultrasonic_enabled:
            self.ultrasonic_detected = None
        self.ultrasonic_enabled = requested
        if not requested:
            self.avoid_direction = DIRECTION_NONE

    def on_ultrasonic_event(self, msg: Int8MultiArray) -> None:
        if not self.ultrasonic_enabled or len(msg.data) < 2:
            return
        self.ultrasonic_detected = int(msg.data[0]) == EVENT_DETECTED
        direction = int(msg.data[1])
        self.avoid_direction = (
            direction
            if direction in (DIRECTION_LEFT, DIRECTION_RIGHT)
            else DIRECTION_NONE
        )

    def on_avoidance_steer(self, msg: Float32) -> None:
        value = float(msg.data)
        if np.isfinite(value):
            self.avoidance_steer_deg = value

    def on_avoidance_valid(self, msg: Bool) -> None:
        self.avoidance_valid = bool(msg.data)

    def on_avoidance_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
            required_ratios = (
                "roi_left_ratio",
                "roi_right_ratio",
                "roi_top_ratio",
                "roi_bottom_ratio",
                "bbox_left_boundary_ratio",
                "bbox_right_boundary_ratio",
                "min_bbox_area_ratio",
            )
            if status.get("mode") != "yolo_only":
                return
            if not all(key in status for key in required_ratios):
                return
            normalized_overlay_geometry(
                2,
                2,
                float(status["roi_left_ratio"]),
                float(status["roi_right_ratio"]),
                float(status["roi_top_ratio"]),
                float(status["roi_bottom_ratio"]),
                float(status["bbox_left_boundary_ratio"]),
                float(status["bbox_right_boundary_ratio"]),
            )
            min_area_ratio = float(status["min_bbox_area_ratio"])
            if not np.isfinite(min_area_ratio) or not (
                0.0 < min_area_ratio <= 1.0
            ):
                return
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        self.yolo_only_guide_config = status

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

        detections = self.detect_bboxes(frame)
        bbox = self.select_best_bbox(detections)
        detected = bbox is not None
        self.yolo_detected = detected
        self.publish_detected(detected)

        if detected:
            self.publish_bbox(bbox)

        detection_frame = frame.copy()
        self.draw_detection_overlay(detection_frame, detections)
        self.publish_image(detection_frame, image_msg)

        if self.show_window:
            self.draw_debug_window(detection_frame)

    def detect_bboxes(self, frame) -> List[Detection]:
        results = self.model.predict(
            frame, conf=self.confidence_threshold, verbose=False
        )
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2.0
                center_y = y1 + height / 2.0
                detections.append(((center_x, center_y, width, height), conf))

        return detections

    @staticmethod
    def select_best_bbox(detections: Sequence[Detection]) -> Optional[BBox]:
        if not detections:
            return None
        return max(detections, key=lambda detection: detection[1])[0]

    def detect_best_bbox(self, frame) -> Optional[BBox]:
        """Return the control bbox while retaining the original public helper."""
        return self.select_best_bbox(self.detect_bboxes(frame))

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
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.detection_jpeg_quality],
        )
        if not success:
            self.get_logger().error(
                "Failed to encode obstacle detection image as JPEG"
            )
            return

        msg = CompressedImage()
        msg.header = source_msg.header
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.detection_image_pub.publish(msg)

    def draw_detection_overlay(
        self, frame, detections: Sequence[Detection]
    ) -> None:
        frame_height, frame_width = frame.shape[:2]
        self.draw_yolo_only_guides(frame)
        for bbox, confidence in detections:
            center_x, center_y, width, height = bbox
            x1 = int(center_x - width / 2.0)
            y1 = int(center_y - height / 2.0)
            x2 = int(center_x + width / 2.0)
            y2 = int(center_y + height / 2.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)

            label = f"confidence {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            label_x = min(
                max(x1, 0), max(frame_width - label_width - 6, 0)
            )
            label_y = max(y1 - 8, label_height + baseline + 4)
            label_y = min(label_y, max(frame_height - baseline - 2, 0))
            cv2.rectangle(
                frame,
                (label_x, label_y - label_height - baseline - 4),
                (label_x + label_width + 6, label_y + baseline + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                label,
                (label_x + 3, label_y - baseline),
                font,
                font_scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

        self.draw_status_panel(frame)

    @staticmethod
    def draw_dashed_vertical_line(
        frame,
        x: int,
        color: Tuple[int, int, int],
        thickness: int,
        dash_px: int = 12,
        gap_px: int = 8,
    ) -> None:
        height = frame.shape[0]
        for y1 in range(0, height, dash_px + gap_px):
            y2 = min(height - 1, y1 + dash_px)
            cv2.line(frame, (x, y1), (x, y2), color, thickness)

    def draw_yolo_only_guides(self, frame) -> None:
        status = self.yolo_only_guide_config
        if not self.show_yolo_only_guides or status is None:
            return

        try:
            geometry = normalized_overlay_geometry(
                frame.shape[1],
                frame.shape[0],
                float(status["roi_left_ratio"]),
                float(status["roi_right_ratio"]),
                float(status["roi_top_ratio"]),
                float(status["roi_bottom_ratio"]),
                float(status["bbox_left_boundary_ratio"]),
                float(status["bbox_right_boundary_ratio"]),
            )
            min_area_percent = 100.0 * float(
                status["min_bbox_area_ratio"]
            )
        except (KeyError, TypeError, ValueError):
            return

        # The translucent fill makes the actual center-point trigger region
        # visible while retaining the complete lower edge of the ROI.
        roi_overlay = frame.copy()
        cv2.rectangle(
            roi_overlay,
            (geometry["roi_left"], geometry["roi_top"]),
            (geometry["roi_right"], geometry["roi_bottom"]),
            (120, 80, 0),
            -1,
        )
        cv2.addWeighted(roi_overlay, 0.18, frame, 0.82, 0.0, frame)
        cv2.rectangle(
            frame,
            (geometry["roi_left"], geometry["roi_top"]),
            (geometry["roi_right"], geometry["roi_bottom"]),
            (255, 200, 0),
            2,
        )
        cv2.line(
            frame,
            (0, geometry["roi_bottom"]),
            (frame.shape[1] - 1, geometry["roi_bottom"]),
            (255, 0, 255),
            3,
        )
        bottom_label_y = min(
            frame.shape[0] - 10,
            geometry["roi_bottom"] + 20,
        )
        cv2.putText(
            frame,
            "ROI BOTTOM LIMIT",
            (8, bottom_label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        state = str(status.get("state", ""))
        active_exit = (
            "right" if state == "turn_left"
            else "left" if state == "turn_right"
            else ""
        )
        for side in ("left", "right"):
            active = side == active_exit
            color = (0, 0, 255) if active else (0, 165, 255)
            self.draw_dashed_vertical_line(
                frame,
                geometry[f"exit_{side}"],
                color,
                4 if active else 2,
            )

        label_y = max(
            20,
            min(frame.shape[0] - 12, geometry["roi_bottom"] - 10),
        )
        label_x = min(
            geometry["roi_left"] + 6,
            max(frame.shape[1] - 360, 0),
        )
        cv2.putText(
            frame,
            f"TRIGGER ROI: bbox center, area >= {min_area_percent:.1f}%",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        exit_label_y = min(frame.shape[0] - 8, 132)
        for side, turn in (("left", "RIGHT"), ("right", "LEFT")):
            x = geometry[f"exit_{side}"]
            label = f"{turn} TURN EXIT"
            label_width = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1
            )[0][0]
            label_x = min(
                max(x - label_width // 2, 0),
                max(frame.shape[1] - label_width, 0),
            )
            cv2.putText(
                frame,
                label,
                (label_x, exit_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 255) if side == active_exit else (0, 165, 255),
                2 if side == active_exit else 1,
                cv2.LINE_AA,
            )

    def draw_status_panel(self, frame) -> None:
        if self.avoidance_valid and abs(self.avoidance_steer_deg) > 0.1:
            direction = (
                "LEFT" if self.avoidance_steer_deg > 0.0 else "RIGHT"
            )
            direction_text = f"{direction} ({self.avoidance_steer_deg:+.1f} deg)"
            direction_active = True
        else:
            direction = {
                DIRECTION_LEFT: "LEFT",
                DIRECTION_RIGHT: "RIGHT",
            }.get(self.avoid_direction, "NONE")
            direction_text = direction
            direction_active = direction != "NONE"

        if not self.ultrasonic_enabled:
            ultrasonic_text = "DISABLED"
            ultrasonic_color = (160, 160, 160)
        elif self.ultrasonic_detected is None:
            ultrasonic_text = "WAITING"
            ultrasonic_color = (0, 200, 255)
        elif self.ultrasonic_detected:
            ultrasonic_text = "DETECTED"
            ultrasonic_color = (0, 0, 255)
        else:
            ultrasonic_text = "CLEAR"
            ultrasonic_color = (0, 255, 0)

        rows = (
            (
                f"YOLO: {'DETECTED' if self.yolo_detected else 'CLEAR'}",
                (0, 255, 0) if self.yolo_detected else (200, 200, 200),
            ),
            (f"ULTRASONIC: {ultrasonic_text}", ultrasonic_color),
            (
                f"AVOID DIR: {direction_text}",
                (0, 200, 255) if direction_active else (200, 200, 200),
            ),
        )

        panel_width = min(390, max(1, frame.shape[1] - 20))
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + panel_width, 10 + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0.0, frame)
        for index, (text, color) in enumerate(rows):
            cv2.putText(
                frame,
                text,
                (22, 38 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

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
