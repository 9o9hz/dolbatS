#!/usr/bin/env python3
"""ROS 2 node: camera image -> BEV lane-mask topics."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from lane_processing import (
    DEFAULT_BEV_PARAMS,
    DEFAULT_MODEL,
    decode_compressed_image,
    load_bev_parameters,
)


PARAMETER_DEFAULTS = {
    "model_path": str(DEFAULT_MODEL),
    "bev_params": str(DEFAULT_BEV_PARAMS),
    "image_topic": "/image_raw/compressed",
    "bev_topic": "/lane/detection/bev/compressed",
    "mask_topic": "/lane/detection/mask/compressed",
    "segmentation_topic": "/lane/detection/segmentation/compressed",
    "status_topic": "/lane/detection/status",
    "instances_topic": "/lane/detection/instances",
    "confidence": 0.25,
    "image_size": 640,
    "device": "auto",
    "calibration_width": 640,
    "calibration_height": 480,
    "process_every_nth_frame": 1,
    "debug_jpeg_quality": 80,
    "display": False,
    "display_scale": 1.0,
    "display_window_x": 20,
    "display_window_y": 60,
}


def _resolve_device(requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized not in ("", "auto"):
        return int(normalized) if normalized.isdigit() else requested
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


class LaneDetectNode(Node):
    """Runs only perspective conversion and YOLO mask inference."""

    def __init__(self) -> None:
        super().__init__("lane_detect")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)

        parameter = lambda name: self.get_parameter(name).value
        model_path = Path(str(parameter("model_path")))
        bev_path_value = str(parameter("bev_params")).strip()
        bev_path = (
            Path(bev_path_value)
            if bev_path_value
            else DEFAULT_BEV_PARAMS
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if self.model.task != "segment":
            raise ValueError(
                "Lane detector requires a segmentation model, "
                f"got {self.model.task!r}"
            )
        self.lane_class_ids = {
            int(class_id)
            for class_id, name in self.model.names.items()
            if "lane" in str(name).lower()
        }
        if not self.lane_class_ids:
            raise ValueError(
                f"No lane class found in model classes: {self.model.names}"
            )

        bev = load_bev_parameters(bev_path)
        self.source_points = bev.source_points
        self.destination_points = bev.destination_points
        self.warp_size = (bev.width, bev.height)
        self.calibration_width = max(
            1, int(parameter("calibration_width"))
        )
        self.calibration_height = max(
            1, int(parameter("calibration_height"))
        )
        self.confidence = float(parameter("confidence"))
        self.image_size = max(1, int(parameter("image_size")))
        self.device = _resolve_device(str(parameter("device")))
        self.process_every_nth_frame = max(
            1, int(parameter("process_every_nth_frame"))
        )
        self.jpeg_quality = int(
            np.clip(int(parameter("debug_jpeg_quality")), 1, 100)
        )
        self.display = bool(parameter("display"))
        self.display_scale = max(
            0.1, float(parameter("display_scale"))
        )
        self.display_window_x = int(parameter("display_window_x"))
        self.display_window_y = int(parameter("display_window_y"))
        self.display_window_ready = False
        self.frame_count = 0

        image_topic = str(parameter("image_topic"))
        bev_topic = str(parameter("bev_topic"))
        mask_topic = str(parameter("mask_topic"))
        segmentation_topic = str(parameter("segmentation_topic"))
        status_topic = str(parameter("status_topic"))
        instances_topic = str(parameter("instances_topic"))
        self.bev_publisher = self.create_publisher(
            CompressedImage,
            bev_topic,
            qos_profile_sensor_data,
        )
        self.mask_publisher = self.create_publisher(
            CompressedImage,
            mask_topic,
            qos_profile_sensor_data,
        )
        self.segmentation_publisher = self.create_publisher(
            CompressedImage,
            segmentation_topic,
            qos_profile_sensor_data,
        )
        self.status_publisher = self.create_publisher(
            String,
            status_topic,
            10,
        )
        self.instances_publisher = self.create_publisher(
            String,
            instances_topic,
            10,
        )
        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"{image_topic} -> {mask_topic}; model={model_path}, "
            f"bev={bev_path}, device={self.device}"
        )

    def _make_bev(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        source = self.source_points.copy()
        source[:, 0] *= width / float(self.calibration_width)
        source[:, 1] *= height / float(self.calibration_height)
        matrix = cv2.getPerspectiveTransform(
            source,
            self.destination_points,
        )
        return cv2.warpPerspective(
            frame,
            matrix,
            self.warp_size,
            flags=cv2.INTER_LINEAR,
        )

    def on_image(self, message: CompressedImage) -> None:
        self.frame_count += 1
        if (self.frame_count - 1) % self.process_every_nth_frame:
            return

        started = time.perf_counter()
        try:
            bev = self._make_bev(decode_compressed_image(message))
            result = self.model.predict(
                source=bev,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                retina_masks=True,
                verbose=False,
            )[0]
            mask, instances = self._combined_mask(
                result,
                bev.shape[:2],
            )
            instance_count = len(instances)
            segmentation = result.plot(
                labels=True,
                boxes=False,
                masks=True,
                conf=True,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Lane detection failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        inference_ms = (time.perf_counter() - started) * 1000.0
        self._publish_image(
            bev,
            message,
            self.bev_publisher,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        self._publish_image(
            mask,
            message,
            self.mask_publisher,
            ".png",
            [],
        )
        self._publish_image(
            segmentation,
            message,
            self.segmentation_publisher,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "lane_instances": instance_count,
                        "mask_pixels": int(np.count_nonzero(mask)),
                        "inference_ms": round(inference_ms, 1),
                    },
                    ensure_ascii=False,
                )
            )
        )
        timestamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        self.instances_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "timestamp_ns": timestamp_ns,
                        "instances": instances,
                    },
                    ensure_ascii=False,
                )
            )
        )

        if self.display:
            preview = cv2.resize(
                segmentation,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            window_name = "lane_detect: segmentation"
            if not self.display_window_ready:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(
                    window_name,
                    self.display_window_x,
                    self.display_window_y,
                )
                self.display_window_ready = True
            cv2.resizeWindow(
                window_name,
                preview.shape[1],
                preview.shape[0],
            )
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

    def _combined_mask(
        self,
        result: Any,
        image_shape: Sequence[int],
    ) -> tuple[np.ndarray, list[dict]]:
        height, width = int(image_shape[0]), int(image_shape[1])
        combined = np.zeros((height, width), dtype=np.uint8)
        if result.masks is None or result.boxes is None:
            return combined, []

        masks = result.masks.data.detach().cpu().numpy()
        classes = (
            result.boxes.cls.detach().cpu().numpy().astype(int)
        )
        confidences = (
            result.boxes.conf.detach().cpu().numpy().astype(float)
        )
        instances = []
        for raw_mask, class_id, confidence in zip(
            masks,
            classes,
            confidences,
        ):
            if int(class_id) not in self.lane_class_ids:
                continue
            resized = cv2.resize(
                (raw_mask > 0.5).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            combined[resized > 0] = 255
            y_values, x_values = np.nonzero(resized)
            if len(x_values) == 0:
                continue
            instances.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(self.model.names[int(class_id)]),
                    "confidence": round(float(confidence), 4),
                    "x_min": int(np.min(x_values)),
                    "y_min": int(np.min(y_values)),
                    "x_max": int(np.max(x_values)),
                    "y_max": int(np.max(y_values)),
                }
            )
        return combined, instances

    @staticmethod
    def _publish_image(
        image: np.ndarray,
        source: CompressedImage,
        publisher: Any,
        extension: str,
        options: list[int],
    ) -> None:
        success, encoded = cv2.imencode(extension, image, options)
        if not success:
            raise RuntimeError(f"Could not encode {extension} image")
        output = CompressedImage()
        output.header = source.header
        output.format = "png" if extension == ".png" else "jpeg"
        output.data = encoded.tobytes()
        publisher.publish(output)

    def destroy_node(self) -> bool:
        if self.display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[LaneDetectNode] = None
    try:
        node = LaneDetectNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
