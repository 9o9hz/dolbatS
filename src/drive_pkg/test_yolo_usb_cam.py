#!/usr/bin/env python3
"""Test an Ultralytics YOLO model on a usb_cam compressed ROS2 topic."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from ultralytics import YOLO


DEFAULT_MODEL = Path(
    "/home/tak/yolo11/data/lane/runs/detect/train/weights/best.pt"
)
DEFAULT_INPUT_TOPIC = "/camera1/image_raw/compressed"
DEFAULT_OUTPUT_TOPIC = "/yolo/detection_view/compressed"


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO directly on usb_cam compressed frames without "
            "calibration or BEV conversion."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--topic", default=DEFAULT_INPUT_TOPIC)
    parser.add_argument("--output-topic", default=DEFAULT_OUTPUT_TOPIC)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, for example 0 or cpu. Default is auto.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV detection preview window.",
    )
    return parser.parse_known_args(argv)


class YoloUsbCamTester(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("yolo_usb_cam_tester")
        self.args = args
        self.model = YOLO(str(args.model))
        self.frame_count = 0
        self.detection_count = 0
        self.class_counts: Counter[str] = Counter()
        self.stop_requested = False
        self.window_name = "YOLO usb_cam test"

        self.publisher = self.create_publisher(
            CompressedImage,
            args.output_topic,
            qos_profile_sensor_data,
        )
        self.subscription = self.create_subscription(
            CompressedImage,
            args.topic,
            self.process_image,
            qos_profile_sensor_data,
        )
        self.stop_timer = self.create_timer(0.1, self.stop_if_requested)

        self.get_logger().info(f"model: {args.model}")
        self.get_logger().info(f"input: {args.topic}")
        self.get_logger().info(f"annotated output: {args.output_topic}")

    def process_image(self, message: CompressedImage) -> None:
        if self.stop_requested:
            return

        try:
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise ValueError("Could not decode the compressed camera frame")

            predict_options = {
                "conf": self.args.confidence,
                "verbose": False,
            }
            if self.args.image_size is not None:
                predict_options["imgsz"] = self.args.image_size
            if self.args.device is not None:
                predict_options["device"] = self.args.device

            result = self.model.predict(frame, **predict_options)[0]
            annotated = result.plot(labels=True, boxes=True, conf=True)
            frame_detections = 0

            if result.boxes is not None:
                frame_detections = len(result.boxes)
                if frame_detections:
                    class_ids = (
                        result.boxes.cls.detach().cpu().numpy().astype(int)
                    )
                    for class_id in class_ids:
                        self.class_counts[self.model.names[int(class_id)]] += 1

            self.frame_count += 1
            self.detection_count += frame_detections
            cv2.putText(
                annotated,
                f"frame={self.frame_count} detections={frame_detections}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self.publish_annotated(annotated, message)

            if not self.args.no_window:
                cv2.imshow(self.window_name, annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    self.stop_requested = True

            if self.frame_count == 1 or self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"frames={self.frame_count}, "
                    f"detections={self.detection_count}, "
                    f"classes={dict(self.class_counts)}"
                )
            if (
                self.args.max_frames is not None
                and self.frame_count >= self.args.max_frames
            ):
                self.stop_requested = True
        except Exception as exc:
            self.get_logger().error(str(exc))
            self.stop_requested = True

    def publish_annotated(
        self,
        frame: np.ndarray,
        source_message: CompressedImage,
    ) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode the annotated image")

        message = CompressedImage()
        message.header = source_message.header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.publisher.publish(message)

    def stop_if_requested(self) -> None:
        if self.stop_requested:
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        if not self.args.no_window:
            cv2.destroyAllWindows()
        return super().destroy_node()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.image_size is not None and args.image_size < 1:
        raise ValueError("--image-size must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args, ros_args = parse_args(argv)
    validate_args(args)
    rclpy.init(args=ros_args)
    node = YoloUsbCamTester(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        frame_count = node.frame_count
        detection_count = node.detection_count
        class_counts = dict(node.class_counts)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"processed_frames: {frame_count}")
    print(f"detections: {detection_count}")
    print(f"classes: {class_counts}")


if __name__ == "__main__":
    main()
