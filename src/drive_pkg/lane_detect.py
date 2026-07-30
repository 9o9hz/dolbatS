#!/usr/bin/env python3
"""BEV warp + YOLO lane segmentation core, used directly by drive_main.py.

Extracted from what used to be a standalone ``lane_detect`` ROS node so the
same detection code can be called in-process (no topic round trip); the
standalone node wrapper was removed once ``drive_main.py`` took over that
role -- see howtorun.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np

from lane_processing import load_bev_parameters


def resolve_device(requested: str) -> Any:
    normalized = requested.strip().lower()
    if normalized not in ("", "auto"):
        return int(normalized) if normalized.isdigit() else requested
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


@dataclass
class DetectionOutput:
    """Result of running BEV warp + YOLO segmentation on one frame."""

    bev: np.ndarray
    mask: np.ndarray
    segmentation: np.ndarray
    instances: list[dict]
    inference_ms: float


class LaneDetectorCore:
    """Pure BEV warp + YOLO segmentation logic, independent of ROS."""

    def __init__(
        self,
        model_path: Path,
        bev_path: Path,
        calibration_width: int,
        calibration_height: int,
        confidence: float,
        image_size: int,
        device_request: str,
    ) -> None:
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
        lane_keywords = ("lane", "dashed", "solid")
        self.lane_class_ids = {
            int(class_id)
            for class_id, name in self.model.names.items()
            if any(keyword in str(name).lower() for keyword in lane_keywords)
        }
        if not self.lane_class_ids:
            raise ValueError(
                f"No lane class found in model classes: {self.model.names}"
            )

        bev = load_bev_parameters(bev_path)
        self.source_points = bev.source_points
        self.destination_points = bev.destination_points
        self.warp_size = (bev.width, bev.height)
        self.calibration_width = max(1, int(calibration_width))
        self.calibration_height = max(1, int(calibration_height))
        self.confidence = float(confidence)
        self.image_size = max(1, int(image_size))
        self.device = resolve_device(str(device_request))

    def make_bev(self, frame: np.ndarray) -> np.ndarray:
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

    def detect(self, frame: np.ndarray) -> DetectionOutput:
        started = time.perf_counter()
        bev = self.make_bev(frame)
        result = self.model.predict(
            source=bev,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]
        mask, instances = self._combined_mask(result, bev.shape[:2])
        segmentation = result.plot(
            labels=True,
            boxes=False,
            masks=True,
            conf=True,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        return DetectionOutput(
            bev=bev,
            mask=mask,
            segmentation=segmentation,
            instances=instances,
            inference_ms=inference_ms,
        )

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
                    "pixel_count": int(len(x_values)),
                    "mask": resized,
                }
            )
        return combined, instances
