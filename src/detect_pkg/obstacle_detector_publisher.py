#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
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


def get_default_bev_path() -> str:
    try:
        return os.path.join(
            get_package_share_directory("drive_pkg"),
            "resource",
            "bev_params_7.npz",
        )
    except Exception:
        return str(
            Path(__file__).resolve().parent.parent
            / "drive_pkg"
            / "resource"
            / "bev_params_7.npz"
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
            "bottom_center_topic", "/detect/obstacle/bottom_center"
        )
        self.declare_parameter(
            "compressed_image_topic", "/image_raw/compressed"
        )
        self.declare_parameter(
            "detection_image_topic", "/camera/lane/detection_view"
        )
        self.declare_parameter(
            "bev_footprint_topic", "/detect/obstacle/bev_footprint"
        )
        self.declare_parameter(
            "bev_detection_image_topic", "/detect/obstacle/bev_view"
        )
        self.declare_parameter("bev_params", get_default_bev_path())
        self.declare_parameter("calibration_width", 640)
        self.declare_parameter("calibration_height", 480)
        self.declare_parameter("pixels_per_meter", 600.0)
        self.declare_parameter("vehicle_width_m", 0.64)
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
        self.calibration_width = max(
            1, int(self.get_parameter("calibration_width").value)
        )
        self.calibration_height = max(
            1, int(self.get_parameter("calibration_height").value)
        )
        self.pixels_per_meter = float(
            self.get_parameter("pixels_per_meter").value
        )
        self.vehicle_width_m = float(
            self.get_parameter("vehicle_width_m").value
        )
        if self.pixels_per_meter <= 0.0 or self.vehicle_width_m <= 0.0:
            raise ValueError("pixels_per_meter and vehicle_width_m must be positive")
        self._load_bev(str(self.get_parameter("bev_params").value))

        enable_topic = str(self.get_parameter("enable_topic").value)
        detected_topic = (
            self.get_parameter("detected_topic").get_parameter_value().string_value
        )
        bbox_topic = self.get_parameter("bbox_topic").get_parameter_value().string_value
        bottom_center_topic = (
            self.get_parameter("bottom_center_topic").get_parameter_value().string_value
        )
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
        bev_footprint_topic = str(
            self.get_parameter("bev_footprint_topic").value
        )
        bev_detection_image_topic = str(
            self.get_parameter("bev_detection_image_topic").value
        )

        self.detected_pub = self.create_publisher(Bool, detected_topic, 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, bbox_topic, 10)
        self.bottom_center_pub = self.create_publisher(
            Float32MultiArray, bottom_center_topic, 10
        )
        self.bev_footprint_pub = self.create_publisher(
            Float32MultiArray, bev_footprint_topic, 10
        )
        self.detection_image_pub = self.create_publisher(
            Image, detection_image_topic, 10
        )
        self.bev_detection_image_pub = self.create_publisher(
            Image, bev_detection_image_topic, 10
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
            f"{detected_topic}=Bool, {bbox_topic}=[cx, cy, w, h] and "
            f"{bottom_center_topic}=[x, y], "
            f"{bev_footprint_topic}=[center_x, center_y, left_x, left_y, "
            f"right_x, right_y, width_px]"
        )

    def _load_bev(self, path_value: str) -> None:
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"BEV parameter file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"src_points", "dst_points", "warp_w", "warp_h"}
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"BEV parameter keys missing: {sorted(missing)}")
            self.bev_source_points = np.asarray(
                data["src_points"], dtype=np.float32
            )
            self.bev_destination_points = np.asarray(
                data["dst_points"], dtype=np.float32
            )
            self.bev_width = int(np.asarray(data["warp_w"]).reshape(-1)[0])
            self.bev_height = int(np.asarray(data["warp_h"]).reshape(-1)[0])
        if (
            self.bev_source_points.shape != (4, 2)
            or self.bev_destination_points.shape != (4, 2)
            or self.bev_width <= 0
            or self.bev_height <= 0
        ):
            raise ValueError(f"Invalid BEV parameters: {path}")

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
            self.publish_bottom_center(bbox)

        detection_frame = frame.copy()
        self.draw_detection_overlay(detection_frame, bbox)
        self.publish_image(detection_frame, image_msg)
        bev_frame = self.make_bev(frame)
        if bbox is not None:
            footprint = self.project_bev_footprint(bbox, frame.shape[:2])
            self.publish_bev_footprint(footprint)
            self.draw_bev_footprint(bev_frame, footprint)
        self.publish_image(
            bev_frame, image_msg, publisher=self.bev_detection_image_pub
        )

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

    def publish_bottom_center(self, bbox: BBox) -> None:
        center_x, center_y, _, height = bbox
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="bottom_center", size=2, stride=2),
        ]
        msg.data = [float(center_x), float(center_y + height / 2.0)]
        self.bottom_center_pub.publish(msg)

    def bev_matrix(self, image_shape) -> np.ndarray:
        height, width = image_shape
        source = self.bev_source_points.copy()
        source[:, 0] *= width / float(self.calibration_width)
        source[:, 1] *= height / float(self.calibration_height)
        return cv2.getPerspectiveTransform(
            source, self.bev_destination_points
        )

    def make_bev(self, frame):
        return cv2.warpPerspective(
            frame,
            self.bev_matrix(frame.shape[:2]),
            (self.bev_width, self.bev_height),
        )

    def project_bev_footprint(self, bbox: BBox, image_shape) -> Tuple[float, ...]:
        center_x, center_y, _, height = bbox
        camera_point = np.asarray(
            [[[center_x, center_y + height / 2.0]]], dtype=np.float32
        )
        bev_center = cv2.perspectiveTransform(
            camera_point, self.bev_matrix(image_shape)
        )[0, 0]
        width_px = self.vehicle_width_m * self.pixels_per_meter
        half_width = width_px / 2.0
        return (
            float(bev_center[0]),
            float(bev_center[1]),
            float(bev_center[0] - half_width),
            float(bev_center[1]),
            float(bev_center[0] + half_width),
            float(bev_center[1]),
            float(width_px),
        )

    def publish_bev_footprint(self, footprint: Tuple[float, ...]) -> None:
        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="bev_footprint", size=7, stride=7),
        ]
        msg.data = list(footprint)
        self.bev_footprint_pub.publish(msg)

    @staticmethod
    def draw_bev_footprint(frame, footprint: Tuple[float, ...]) -> None:
        center_x, center_y, left_x, left_y, right_x, right_y, _ = footprint
        center = (int(round(center_x)), int(round(center_y)))
        left = (int(round(left_x)), int(round(left_y)))
        right = (int(round(right_x)), int(round(right_y)))
        cv2.line(frame, left, right, (0, 200, 255), 5)
        cv2.circle(frame, center, 7, (255, 0, 255), -1)
        cv2.putText(
            frame,
            f"obstacle BEV: ({center[0]}, {center[1]})",
            (max(0, center[0] - 120), max(20, center[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    def image_message_to_bgr(self, msg: CompressedImage):
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().error(
                f"Failed to decode compressed image (format={msg.format!r})"
            )
            return None
        return frame

    def publish_image(
        self, frame, source_msg: CompressedImage, publisher=None
    ) -> None:
        msg = Image()
        msg.header = source_msg.header
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(frame.strides[0])
        msg.data = frame.tobytes()
        (publisher or self.detection_image_pub).publish(msg)

    def draw_detection_overlay(self, frame, bbox: Optional[BBox]) -> None:
        if bbox is not None:
            center_x, center_y, width, height = bbox
            x1 = int(center_x - width / 2.0)
            y1 = int(center_y - height / 2.0)
            x2 = int(center_x + width / 2.0)
            y2 = int(center_y + height / 2.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
            cv2.circle(
                frame,
                (int(center_x), int(center_y + height / 2.0)),
                5,
                (255, 0, 255),
                -1,
            )
            cv2.putText(
                frame,
                f"bottom: ({int(center_x)}, {int(center_y + height / 2.0)})",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 255),
                2,
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
