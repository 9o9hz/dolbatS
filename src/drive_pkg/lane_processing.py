#!/usr/bin/env python3
"""Legacy lane-processing implementation and reusable path planner core.

Input:
    /image_raw/compressed (sensor_msgs/CompressedImage)

Outputs:
    /lane/path                         (nav_msgs/Path)
    /lane/yolo_drive/segmentation/compressed
    /lane/yolo_drive/path/compressed   (sensor_msgs/CompressedImage)
    /lane/yolo_drive/status            (std_msgs/String)
    /which/lane                        (std_msgs/String)
    /cmd_vel                           (geometry_msgs/Twist)

The best1.pt model was trained on 640x640 BEV images with one ``lane`` class.
Incoming 640x480 usb_cam frames are therefore warped to that training view
before inference. Segmentation components are grouped into left/right lane
boundaries, and their midpoint (or a lane-width offset from one boundary) is
converted into a metric path in ``base_link`` coordinates.

Non-zero velocity is disabled unless ``--enable-drive`` is explicitly given.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as PathMessage
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def _default_bev_params_path() -> Path:
    source_path = (
        Path(__file__).resolve().parent / "resource" / "bev_params_7.npz"
    )
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("drive_pkg"))
            / "resource"
            / "bev_params_7.npz"
        )
    except (ImportError, LookupError):
        return source_path


DEFAULT_MODEL = Path("/home/tak/lane_yolo_project/weight/best1.pt")
DEFAULT_BEV_PARAMS = _default_bev_params_path()
DEFAULT_IMAGE_TOPIC = "/image_raw/compressed"
DEFAULT_PATH_TOPIC = "/lane/path"
DEFAULT_SEGMENTATION_TOPIC = "/lane/yolo_drive/segmentation/compressed"
DEFAULT_DEBUG_TOPIC = "/lane/yolo_drive/path/compressed"
DEFAULT_STATUS_TOPIC = "/lane/yolo_drive/status"
DEFAULT_WHICH_LANE_TOPIC = "/which/lane"
DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"

PointArray = Optional[np.ndarray]
LaneGroup = Dict[str, Any]


@dataclass(frozen=True)
class LaneConfig:
    confidence: float = 0.25
    image_size: int = 640
    device: str = "auto"
    calibration_width: int = 640
    calibration_height: int = 480
    # Point order is read unchanged from the NPZ file.
    source_points: Tuple[float, ...] = (
        -61.0,
        483.0,
        678.0,
        483.0,
        177.0,
        252.0,
        576.0,
        252.0,
    )
    destination_points: Tuple[float, ...] = (
        0.0,
        640.0,
        640.0,
        640.0,
        0.0,
        0.0,
        640.0,
        0.0,
    )
    warp_width: int = 640
    warp_height: int = 640
    pixels_per_meter: float = 600.0
    lane_width_m: float = 0.90
    min_component_area: int = 250
    center_sample_step: int = 5
    same_line_threshold_px: float = 75.0
    min_group_span_px: float = 80.0
    min_group_area: int = 500
    dashed_piece_threshold: int = 2
    prefer_solid_when_dashed: bool = True
    lane_track_max_age_frames: int = 7
    lane_track_match_threshold_px: float = 90.0
    solid_enter_frames: int = 4
    solid_exit_frames: int = 6
    path_top_y: int = 180
    path_bottom_margin: int = 30
    path_step_px: int = 10
    path_resample_step_px: int = 5
    path_ema_alpha: float = 0.35
    path_transition_blend_frames: int = 6
    max_path_lateral_step_m: float = 0.04
    max_missing_frames: int = 8
    # Forward distance from the rear-axle center to the BEV bottom reference.
    bev_reference_forward_offset_m: float = 1.04
    lookahead_min_m: float = 1.1
    lookahead_max_m: float = 2.5
    wheelbase_m: float = 0.545
    max_steering_deg: float = 20.0
    steering_ema_alpha: float = 0.35
    steering_deadband_deg: float = 0.8
    max_steering_change_deg: float = 3.0
    target_speed_mps: float = 0.20


@dataclass
class LaneResult:
    segmentation_image: np.ndarray
    debug_image: np.ndarray
    path_pixels: PointArray
    path_meters: PointArray
    steering_rad: float
    lookahead_m: float
    lookahead_target_m: float
    target_speed_mps: float
    reason: str
    used_fallback: bool
    group_count: int
    dashed_region_count: int
    selection_mode: str
    which_lane: Optional[str]
    inference_ms: float


@dataclass
class PathPlanResult:
    """Result of converting a topic-delivered BEV mask into a local path."""

    path_pixels: PointArray
    path_meters: PointArray
    reason: str
    used_fallback: bool
    group_count: int
    dashed_region_count: int
    selection_mode: str
    which_lane: Optional[str]


@dataclass(frozen=True)
class BevParameters:
    source_points: np.ndarray
    destination_points: np.ndarray
    width: int
    height: int


def _scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar")
    return int(value.reshape(-1)[0])


def load_bev_parameters(path: Path) -> BevParameters:
    if not path.is_file():
        raise FileNotFoundError(f"BEV parameter file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"src_points", "dst_points", "warp_w", "warp_h"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"BEV parameter keys missing: {sorted(missing)}")

        source_points = np.asarray(data["src_points"], dtype=np.float32)
        destination_points = np.asarray(
            data["dst_points"], dtype=np.float32
        )
        width = _scalar_int(data, "warp_w")
        height = _scalar_int(data, "warp_h")

    if (
        source_points.shape != (4, 2)
        or destination_points.shape != (4, 2)
        or width <= 0
        or height <= 0
        or not np.all(np.isfinite(source_points))
        or not np.all(np.isfinite(destination_points))
    ):
        raise ValueError(f"Invalid BEV parameters: {path}")

    homography = cv2.getPerspectiveTransform(
        source_points, destination_points
    )
    if not np.all(np.isfinite(homography)):
        raise ValueError(f"Could not calculate BEV homography: {path}")

    return BevParameters(
        source_points=source_points,
        destination_points=destination_points,
        width=width,
        height=height,
    )


def decode_compressed_image(message: CompressedImage) -> np.ndarray:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode compressed camera image")
    return frame


def make_twist(speed_mps: float, steering_rad: float, wheelbase_m: float) -> Twist:
    message = Twist()
    message.linear.x = float(speed_mps)
    if abs(speed_mps) > 1e-6:
        message.angular.z = float(
            speed_mps / wheelbase_m * math.tan(steering_rad)
        )
    return message


class SegmentationLaneProcessor:
    """Turn YOLO lane masks into a smoothed local metric path."""

    def __init__(
        self,
        model_path: Optional[Path],
        config: LaneConfig,
    ) -> None:
        self.config = config
        self._validate_config()
        self.device = (
            self._resolve_device(config.device)
            if model_path is not None
            else "not-used-by-path-planner"
        )
        self.model = None
        self.lane_class_ids = {0}
        if model_path is not None:
            from ultralytics import YOLO

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"YOLO model not found: {model_path}"
                )
            self.model = YOLO(str(model_path))
            if self.model.task != "segment":
                raise ValueError(
                    "Lane model must be a segmentation model, "
                    f"got {self.model.task!r}"
                )

            self.lane_class_ids = {
                int(class_id)
                for class_id, name in self.model.names.items()
                if "lane" in str(name).lower()
            }
            if not self.lane_class_ids:
                raise ValueError(
                    "No lane class found in model classes: "
                    f"{self.model.names}"
                )

        self._perspective_shape: Optional[Tuple[int, int]] = None
        self._perspective_matrix: Optional[np.ndarray] = None
        self._last_path: PointArray = None
        self._last_path_source: Optional[str] = None
        self._path_transition_frames = 0
        self._missing_frames = 0
        self._last_steering_deg = 0.0
        self._tracked_lanes: Dict[str, Dict[str, Any]] = {
            "left": {"group": None, "age": 0},
            "right": {"group": None, "age": 0},
        }
        self._solid_candidate_group: Optional[LaneGroup] = None
        self._solid_candidate_side: Optional[str] = None
        self._solid_candidate_streak = 0
        self._solid_active_group: Optional[LaneGroup] = None
        self._solid_active_side: Optional[str] = None
        self._solid_missing_frames = 0
        self._using_tracked_boundary = False

    def _validate_config(self) -> None:
        cfg = self.config
        if len(cfg.source_points) != 8:
            raise ValueError("source_points must contain exactly eight numbers")
        if len(cfg.destination_points) != 8:
            raise ValueError(
                "destination_points must contain exactly eight numbers"
            )
        if cfg.calibration_width <= 0 or cfg.calibration_height <= 0:
            raise ValueError("Calibration dimensions must be positive")
        if cfg.warp_width <= 0 or cfg.warp_height <= 0:
            raise ValueError("BEV dimensions must be positive")
        if cfg.pixels_per_meter <= 0.0 or cfg.lane_width_m <= 0.0:
            raise ValueError("Metric conversion values must be positive")
        if not math.isfinite(cfg.bev_reference_forward_offset_m):
            raise ValueError(
                "bev_reference_forward_offset_m must be finite"
            )
        if (
            cfg.lookahead_min_m <= 0.0
            or cfg.lookahead_max_m < cfg.lookahead_min_m
        ):
            raise ValueError(
                "Dynamic lookahead range must satisfy "
                "0 < lookahead_min_m <= lookahead_max_m"
            )
        if cfg.max_steering_deg <= 0.0:
            raise ValueError("max_steering_deg must be positive")
        if (
            cfg.center_sample_step <= 0
            or cfg.path_step_px <= 0
            or cfg.path_resample_step_px <= 0
        ):
            raise ValueError("Path sampling steps must be positive")
        if cfg.dashed_piece_threshold < 2:
            raise ValueError("dashed_piece_threshold must be at least 2")
        if cfg.lane_track_max_age_frames < 0:
            raise ValueError(
                "lane_track_max_age_frames cannot be negative"
            )
        if cfg.lane_track_match_threshold_px <= 0.0:
            raise ValueError(
                "lane_track_match_threshold_px must be positive"
            )
        if cfg.solid_enter_frames <= 0 or cfg.solid_exit_frames <= 0:
            raise ValueError(
                "Solid preference hysteresis frames must be positive"
            )
        if cfg.path_bottom_margin < 0:
            raise ValueError("path_bottom_margin cannot be negative")
        if not 0.0 <= cfg.path_ema_alpha <= 1.0:
            raise ValueError("path_ema_alpha must be between 0 and 1")
        if cfg.path_transition_blend_frames <= 0:
            raise ValueError(
                "path_transition_blend_frames must be positive"
            )
        if cfg.max_path_lateral_step_m <= 0.0:
            raise ValueError(
                "max_path_lateral_step_m must be positive"
            )
        if not 0.0 <= cfg.steering_ema_alpha <= 1.0:
            raise ValueError("steering_ema_alpha must be between 0 and 1")
        if cfg.steering_deadband_deg < 0.0:
            raise ValueError("steering_deadband_deg cannot be negative")
        if cfg.max_steering_change_deg <= 0.0:
            raise ValueError(
                "max_steering_change_deg must be positive"
            )

    @staticmethod
    def _resolve_device(requested: str) -> Any:
        normalized = str(requested).strip().lower()
        if normalized in ("", "auto"):
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        if normalized.isdigit():
            return int(normalized)
        return requested

    def _perspective_for(self, image_shape: Sequence[int]) -> np.ndarray:
        height, width = int(image_shape[0]), int(image_shape[1])
        if (
            self._perspective_shape == (height, width)
            and self._perspective_matrix is not None
        ):
            return self._perspective_matrix

        source = np.asarray(
            self.config.source_points, dtype=np.float32
        ).reshape(4, 2)
        source[:, 0] *= width / float(self.config.calibration_width)
        source[:, 1] *= height / float(self.config.calibration_height)
        destination = np.asarray(
            self.config.destination_points, dtype=np.float32
        ).reshape(4, 2)
        self._perspective_matrix = cv2.getPerspectiveTransform(
            source, destination
        )
        self._perspective_shape = (height, width)
        return self._perspective_matrix

    def make_bev(self, frame: np.ndarray) -> np.ndarray:
        matrix = self._perspective_for(frame.shape)
        return cv2.warpPerspective(
            frame,
            matrix,
            (self.config.warp_width, self.config.warp_height),
            flags=cv2.INTER_LINEAR,
        )

    @staticmethod
    def _component_center_points(
        mask: np.ndarray, y_step: int
    ) -> PointArray:
        points: List[Tuple[float, float]] = []
        for y_value in range(0, mask.shape[0], y_step):
            x_values = np.flatnonzero(mask[y_value])
            if x_values.size:
                points.append((float(np.mean(x_values)), float(y_value)))
        if len(points) < 2:
            return None
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _fit_line(points: PointArray) -> Optional[Tuple[float, float]]:
        if points is None or len(points) < 2:
            return None
        try:
            slope, intercept = np.polyfit(points[:, 1], points[:, 0], 1)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return None
        return float(slope), float(intercept)

    @staticmethod
    def _line_x(line: Tuple[float, float], y_value: float) -> float:
        return line[0] * y_value + line[1]

    @classmethod
    def _fit_curve(
        cls, points: PointArray
    ) -> Optional[Tuple[float, float, float]]:
        if points is None or len(points) < 2:
            return None
        unique_y = np.unique(points[:, 1])
        try:
            if len(unique_y) >= 3:
                coefficients = np.polyfit(
                    points[:, 1], points[:, 0], 2
                )
                if np.all(np.isfinite(coefficients)):
                    return tuple(float(value) for value in coefficients)
            line = cls._fit_line(points)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return None
        if line is None:
            return None
        return 0.0, line[0], line[1]

    @staticmethod
    def _curve_x(
        curve: Tuple[float, float, float], y_value: float
    ) -> float:
        return float(np.polyval(curve, y_value))

    @classmethod
    def _curve_rmse(
        cls,
        points: PointArray,
        curve: Tuple[float, float, float],
    ) -> float:
        predicted = np.polyval(curve, points[:, 1])
        return float(
            np.sqrt(np.mean(np.square(points[:, 0] - predicted)))
        )

    @classmethod
    def _curve_proximity(
        cls,
        first: LaneGroup,
        second: LaneGroup,
    ) -> float:
        overlap_min = max(first["y_min"], second["y_min"])
        overlap_max = min(first["y_max"], second["y_max"])
        if overlap_min <= overlap_max:
            sample_y = np.linspace(overlap_min, overlap_max, 5)
        elif first["y_max"] < second["y_min"]:
            sample_y = np.asarray(
                [(first["y_max"] + second["y_min"]) * 0.5]
            )
        else:
            sample_y = np.asarray(
                [(second["y_max"] + first["y_min"]) * 0.5]
            )
        first_x = np.polyval(first["curve"], sample_y)
        second_x = np.polyval(second["curve"], sample_y)
        return float(np.mean(np.abs(first_x - second_x)))

    def _extract_pieces(
        self,
        inference_result: Any,
        image_shape: Sequence[int],
    ) -> Tuple[List[LaneGroup], np.ndarray]:
        height, width = int(image_shape[0]), int(image_shape[1])
        pieces: List[LaneGroup] = []
        total_mask = np.zeros((height, width), dtype=np.uint8)
        if inference_result.masks is None or inference_result.boxes is None:
            return pieces, total_mask

        masks = inference_result.masks.data.detach().cpu().numpy()
        classes = (
            inference_result.boxes.cls.detach().cpu().numpy().astype(int)
        )
        confidences = (
            inference_result.boxes.conf.detach().cpu().numpy().astype(float)
        )
        reference_y = height * 0.88

        for raw_mask, class_id, confidence in zip(
            masks, classes, confidences
        ):
            if int(class_id) not in self.lane_class_ids:
                continue
            mask = cv2.resize(
                (raw_mask > 0.5).astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            total_mask = cv2.bitwise_or(total_mask, mask)
            component_count, labels, stats, _ = (
                cv2.connectedComponentsWithStats(mask, connectivity=8)
            )
            for label in range(1, component_count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < self.config.min_component_area:
                    continue
                component = (labels == label).astype(np.uint8)
                points = self._component_center_points(
                    component, self.config.center_sample_step
                )
                curve = self._fit_curve(points)
                if points is None or curve is None:
                    continue
                y_min = float(np.min(points[:, 1]))
                y_max = float(np.max(points[:, 1]))
                x_reference_y = float(
                    np.clip(reference_y, y_min, y_max)
                )
                pieces.append(
                    {
                        "points": points,
                        "curve": curve,
                        "y_min": y_min,
                        "y_max": y_max,
                        "area": area,
                        "confidence": float(confidence),
                        "x_ref": self._curve_x(
                            curve, x_reference_y
                        ),
                    }
                )
        return pieces, total_mask

    def _extract_mask_pieces(
        self,
        mask: np.ndarray,
    ) -> Tuple[List[LaneGroup], np.ndarray]:
        """Extract lane components from a mask received through a ROS topic."""

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("Lane mask must be a non-empty mono image")

        total_mask = (mask > 0).astype(np.uint8)
        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                total_mask,
                connectivity=8,
            )
        )
        reference_y = total_mask.shape[0] * 0.88
        pieces: List[LaneGroup] = []
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.config.min_component_area:
                continue
            component = (labels == label).astype(np.uint8)
            points = self._component_center_points(
                component,
                self.config.center_sample_step,
            )
            curve = self._fit_curve(points)
            if points is None or curve is None:
                continue
            y_min = float(np.min(points[:, 1]))
            y_max = float(np.max(points[:, 1]))
            x_reference_y = float(
                np.clip(reference_y, y_min, y_max)
            )
            pieces.append(
                {
                    "points": points,
                    "curve": curve,
                    "y_min": y_min,
                    "y_max": y_max,
                    "area": area,
                    "confidence": 1.0,
                    "x_ref": self._curve_x(
                        curve,
                        x_reference_y,
                    ),
                }
            )
        return pieces, total_mask

    def _group_pieces(self, pieces: List[LaneGroup]) -> List[LaneGroup]:
        groups: List[LaneGroup] = []
        reference_y = self.config.warp_height * 0.88

        for piece in sorted(pieces, key=lambda item: item["x_ref"]):
            best_group: Optional[LaneGroup] = None
            best_curve: Optional[Tuple[float, float, float]] = None
            best_points: PointArray = None
            best_score = float("inf")
            for group in groups:
                proximity = self._curve_proximity(group, piece)
                if proximity >= self.config.same_line_threshold_px:
                    continue

                combined_points = np.vstack(
                    (group["points"], piece["points"])
                )
                combined_curve = self._fit_curve(combined_points)
                if combined_curve is None:
                    continue
                curve_rmse = self._curve_rmse(
                    combined_points, combined_curve
                )
                if curve_rmse >= self.config.same_line_threshold_px:
                    continue

                score = proximity + curve_rmse
                if score < best_score:
                    best_group = group
                    best_curve = combined_curve
                    best_points = combined_points
                    best_score = score

            if best_group is None:
                groups.append(
                    {
                        "pieces": [piece],
                        "points": piece["points"].copy(),
                        "curve": piece["curve"],
                        "y_min": piece["y_min"],
                        "y_max": piece["y_max"],
                        "x_ref": piece["x_ref"],
                    }
                )
                continue

            best_group["pieces"].append(piece)
            best_group["points"] = best_points
            best_group["curve"] = best_curve
            best_group["y_min"] = float(
                np.min(best_points[:, 1])
            )
            best_group["y_max"] = float(
                np.max(best_points[:, 1])
            )
            best_group["x_ref"] = self._curve_x(
                best_curve, reference_y
            )

        reliable_groups: List[LaneGroup] = []
        for group in groups:
            span = float(
                np.max(group["points"][:, 1])
                - np.min(group["points"][:, 1])
            )
            area = int(
                sum(piece["area"] for piece in group["pieces"])
            )
            if (
                span < self.config.min_group_span_px
                or area < self.config.min_group_area
            ):
                continue
            group["span"] = span
            group["area"] = area
            reliable_groups.append(group)

        reliable_groups.sort(key=lambda item: item["x_ref"])
        return reliable_groups

    def _group_side(self, group: LaneGroup) -> str:
        center_x = self.config.warp_width * 0.5
        return "left" if group["x_ref"] < center_x else "right"

    def _classify_which_lane(
        self, groups: List[LaneGroup]
    ) -> Optional[str]:
        center_x = self.config.warp_width * 0.5
        nearest_groups: Dict[str, LaneGroup] = {}
        for side in ("left", "right"):
            candidates = [
                group
                for group in groups
                if self._group_side(group) == side
                and self._group_has_path_overlap(group)
            ]
            if candidates:
                nearest_groups[side] = min(
                    candidates,
                    key=lambda group: abs(
                        float(group["x_ref"]) - center_x
                    ),
                )

        if "left" not in nearest_groups or "right" not in nearest_groups:
            return None

        left_is_dashed = (
            len(nearest_groups["left"]["pieces"])
            >= self.config.dashed_piece_threshold
        )
        right_is_dashed = (
            len(nearest_groups["right"]["pieces"])
            >= self.config.dashed_piece_threshold
        )
        if not left_is_dashed and right_is_dashed:
            return "lane_1"
        if left_is_dashed and not right_is_dashed:
            return "lane_2"
        return None

    def _group_has_path_overlap(self, group: LaneGroup) -> bool:
        overlap_min = max(
            float(group["y_min"]),
            float(self.config.path_top_y),
        )
        overlap_max = min(
            float(group["y_max"]),
            float(
                self.config.warp_height
                - self.config.path_bottom_margin
            ),
        )
        return (
            overlap_max - overlap_min
            >= 2.0 * self.config.path_step_px
        )

    def _group_match_distance(
        self,
        previous: LaneGroup,
        current: LaneGroup,
    ) -> float:
        overlap_min = max(
            float(previous["y_min"]),
            float(current["y_min"]),
            float(self.config.path_top_y),
        )
        overlap_max = min(
            float(previous["y_max"]),
            float(current["y_max"]),
            float(
                self.config.warp_height
                - self.config.path_bottom_margin
            ),
        )
        if overlap_max - overlap_min >= 10.0:
            sample_y = np.linspace(overlap_min, overlap_max, 7)
            previous_x = np.polyval(previous["curve"], sample_y)
            current_x = np.polyval(current["curve"], sample_y)
            curve_distance = float(
                np.mean(np.abs(previous_x - current_x))
            )
        else:
            curve_distance = abs(
                float(previous["x_ref"]) - float(current["x_ref"])
            )
        reference_distance = abs(
            float(previous["x_ref"]) - float(current["x_ref"])
        )
        return curve_distance + 0.25 * reference_distance

    def _best_group_match(
        self,
        previous: Optional[LaneGroup],
        groups: List[LaneGroup],
        side: str,
    ) -> Optional[LaneGroup]:
        if previous is None:
            return None
        candidates = [
            group for group in groups
            if self._group_side(group) == side
            and self._group_has_path_overlap(group)
        ]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda group: self._group_match_distance(
                previous, group
            ),
        )
        if (
            self._group_match_distance(previous, best)
            > self.config.lane_track_match_threshold_px
        ):
            return None
        return best

    def _update_lane_tracks(
        self, groups: List[LaneGroup]
    ) -> Tuple[Optional[LaneGroup], Optional[LaneGroup]]:
        match_options: List[Tuple[float, str, LaneGroup]] = []
        for side in ("left", "right"):
            previous = self._tracked_lanes[side]["group"]
            if previous is None:
                continue
            for group in groups:
                if (
                    self._group_side(group) != side
                    or not self._group_has_path_overlap(group)
                ):
                    continue
                score = self._group_match_distance(previous, group)
                if score <= self.config.lane_track_match_threshold_px:
                    match_options.append((score, side, group))

        assignments: Dict[str, LaneGroup] = {}
        used_group_ids = set()
        for _, side, group in sorted(
            match_options, key=lambda item: item[0]
        ):
            if side in assignments or id(group) in used_group_ids:
                continue
            assignments[side] = group
            used_group_ids.add(id(group))

        for side in ("left", "right"):
            track = self._tracked_lanes[side]
            matched = assignments.get(side)
            if matched is not None:
                track["group"] = matched
                track["age"] = 0
                continue
            if track["group"] is not None:
                track["age"] += 1
                if (
                    track["age"]
                    > self.config.lane_track_max_age_frames
                ):
                    track["group"] = None
                    track["age"] = 0

        center_x = self.config.warp_width * 0.5
        for side in ("left", "right"):
            track = self._tracked_lanes[side]
            if track["group"] is not None:
                continue
            candidates = [
                group
                for group in groups
                if self._group_side(group) == side
                and self._group_has_path_overlap(group)
                and id(group) not in used_group_ids
            ]
            if not candidates:
                continue
            initialized = min(
                candidates,
                key=lambda group: abs(
                    float(group["x_ref"]) - center_x
                ),
            )
            track["group"] = initialized
            track["age"] = 0
            used_group_ids.add(id(initialized))

        return (
            self._tracked_lanes["left"]["group"],
            self._tracked_lanes["right"]["group"],
        )

    def _raw_solid_candidate(
        self,
        groups: List[LaneGroup],
        dashed_region_count: int,
    ) -> Tuple[Optional[LaneGroup], Optional[str]]:
        if (
            not self.config.prefer_solid_when_dashed
            or dashed_region_count
            < self.config.dashed_piece_threshold
        ):
            return None, None
        solid_candidates = [
            group
            for group in groups
            if len(group["pieces"]) == 1
            and self._group_has_path_overlap(group)
        ]
        if not solid_candidates:
            return None, None
        solid = max(
            solid_candidates,
            key=lambda item: (
                item.get("span", 0.0),
                item.get("area", 0),
            ),
        )
        return solid, self._group_side(solid)

    def _update_solid_preference(
        self,
        groups: List[LaneGroup],
        candidate: Optional[LaneGroup],
        candidate_side: Optional[str],
    ) -> Tuple[Optional[LaneGroup], Optional[str]]:
        same_candidate = (
            candidate is not None
            and candidate_side == self._solid_candidate_side
            and self._solid_candidate_group is not None
            and self._group_match_distance(
                self._solid_candidate_group, candidate
            )
            <= self.config.lane_track_match_threshold_px
        )
        if candidate is None:
            self._solid_candidate_group = None
            self._solid_candidate_side = None
            self._solid_candidate_streak = 0
        else:
            self._solid_candidate_streak = (
                self._solid_candidate_streak + 1
                if same_candidate
                else 1
            )
            self._solid_candidate_group = candidate
            self._solid_candidate_side = candidate_side

        if self._solid_active_group is not None:
            matched = self._best_group_match(
                self._solid_active_group,
                groups,
                str(self._solid_active_side),
            )
            if matched is not None:
                self._solid_active_group = matched

            confirmed = (
                candidate is not None
                and candidate_side == self._solid_active_side
                and self._group_match_distance(
                    self._solid_active_group, candidate
                )
                <= self.config.lane_track_match_threshold_px
            )
            if confirmed:
                self._solid_active_group = candidate
                self._solid_missing_frames = 0
            else:
                self._solid_missing_frames += 1

            if (
                self._solid_missing_frames
                < self.config.solid_exit_frames
            ):
                return (
                    self._solid_active_group,
                    self._solid_active_side,
                )
            self._solid_active_group = None
            self._solid_active_side = None
            self._solid_missing_frames = 0

        if (
            candidate is not None
            and self._solid_candidate_streak
            >= self.config.solid_enter_frames
        ):
            self._solid_active_group = candidate
            self._solid_active_side = candidate_side
            self._solid_missing_frames = 0
            return candidate, candidate_side
        return None, None

    def _choose_boundaries(
        self, groups: List[LaneGroup]
    ) -> Tuple[
        Optional[LaneGroup],
        Optional[LaneGroup],
        str,
        int,
    ]:
        dashed_region_count = max(
            (len(group["pieces"]) for group in groups),
            default=0,
        )
        left, right = self._update_lane_tracks(groups)
        solid_candidate, solid_side = self._raw_solid_candidate(
            groups, dashed_region_count
        )
        preferred, preferred_side = self._update_solid_preference(
            groups, solid_candidate, solid_side
        )
        if preferred is not None:
            self._using_tracked_boundary = (
                self._solid_missing_frames > 0
            )
            if preferred_side == "left":
                return (
                    preferred,
                    None,
                    "solid_left_preferred",
                    dashed_region_count,
                )
            return (
                None,
                preferred,
                "solid_right_preferred",
                dashed_region_count,
            )

        self._using_tracked_boundary = any(
            self._tracked_lanes[side]["group"] is not None
            and self._tracked_lanes[side]["age"] > 0
            for side in ("left", "right")
        )
        if solid_candidate is not None:
            selection_mode = "solid_candidate_pending"
        elif (
            self.config.prefer_solid_when_dashed
            and dashed_region_count
            >= self.config.dashed_piece_threshold
        ):
            selection_mode = "dashed_detected_no_solid"
        else:
            selection_mode = "normal"
        return left, right, selection_mode, dashed_region_count

    def _path_y_values(self) -> np.ndarray:
        return np.arange(
            self.config.warp_height
            - self.config.path_bottom_margin,
            self.config.path_top_y - 1,
            -self.config.path_resample_step_px,
            dtype=np.float32,
        )

    def _raw_path_y_values(self) -> np.ndarray:
        return np.arange(
            self.config.warp_height
            - self.config.path_bottom_margin,
            self.config.path_top_y - 1,
            -self.config.path_step_px,
            dtype=np.float32,
        )

    @staticmethod
    def _sample_group_on_y(
        group: Optional[LaneGroup],
        y_values: np.ndarray,
    ) -> np.ndarray:
        sampled = np.full(y_values.shape, np.nan, dtype=np.float32)
        if group is None:
            return sampled
        valid = (
            (y_values >= float(group["y_min"]))
            & (y_values <= float(group["y_max"]))
        )
        if np.any(valid):
            sampled[valid] = np.polyval(
                group["curve"], y_values[valid]
            ).astype(np.float32)
        return sampled

    def _normal_offset_on_y(
        self,
        group: Optional[LaneGroup],
        y_values: np.ndarray,
        direction: float,
    ) -> np.ndarray:
        sampled = np.full(y_values.shape, np.nan, dtype=np.float32)
        if group is None:
            return sampled
        y_min = float(group["y_min"])
        y_max = float(group["y_max"])
        if y_max - y_min < 2.0:
            return sampled

        source_y = np.arange(
            y_min, y_max + 1.0, 1.0, dtype=np.float32
        )
        coefficients = np.asarray(group["curve"], dtype=np.float64)
        source_x = np.polyval(coefficients, source_y)
        slope = 2.0 * coefficients[0] * source_y + coefficients[1]
        normal_scale = np.sqrt(1.0 + np.square(slope))
        offset_px = (
            0.5
            * self.config.lane_width_m
            * self.config.pixels_per_meter
        )
        offset_x = (
            source_x + direction * offset_px / normal_scale
        )
        offset_y = (
            source_y - direction * offset_px * slope / normal_scale
        )
        finite = np.isfinite(offset_x) & np.isfinite(offset_y)
        if np.count_nonzero(finite) < 2:
            return sampled

        order = np.argsort(offset_y[finite])
        sorted_y = offset_y[finite][order]
        sorted_x = offset_x[finite][order]
        unique_y, unique_indices = np.unique(
            sorted_y, return_index=True
        )
        unique_x = sorted_x[unique_indices]
        if len(unique_y) < 2:
            return sampled
        valid = (
            (y_values >= unique_y[0])
            & (y_values <= unique_y[-1])
        )
        sampled[valid] = np.interp(
            y_values[valid], unique_y, unique_x
        ).astype(np.float32)

        # A steep curve can move the true offset polyline outside the
        # observed y interval near an endpoint. Fill only those endpoint
        # gaps with the x component of the local normal so the fixed y-grid
        # remains available without reverting to a horizontal offset.
        endpoint_gap = (
            ~np.isfinite(sampled)
            & (y_values >= y_min)
            & (y_values <= y_max)
        )
        if np.any(endpoint_gap):
            gap_y = y_values[endpoint_gap]
            gap_x = np.polyval(coefficients, gap_y)
            gap_slope = (
                2.0 * coefficients[0] * gap_y + coefficients[1]
            )
            gap_scale = np.sqrt(1.0 + np.square(gap_slope))
            sampled[endpoint_gap] = (
                gap_x + direction * offset_px / gap_scale
            ).astype(np.float32)
        return sampled

    def _build_path(
        self,
        left: Optional[LaneGroup],
        right: Optional[LaneGroup],
    ) -> Tuple[PointArray, str]:
        if left is None and right is None:
            return None, "no_boundary"

        y_values = self._raw_path_y_values()
        left_x = self._sample_group_on_y(left, y_values)
        right_x = self._sample_group_on_y(right, y_values)
        left_offset = self._normal_offset_on_y(
            left, y_values, direction=1.0
        )
        right_offset = self._normal_offset_on_y(
            right, y_values, direction=-1.0
        )

        path_x = np.full(y_values.shape, np.nan, dtype=np.float32)
        both = (
            np.isfinite(left_x)
            & np.isfinite(right_x)
            & (right_x > left_x)
        )
        path_x[both] = 0.5 * (left_x[both] + right_x[both])

        left_only = (
            ~np.isfinite(path_x) & np.isfinite(left_offset)
        )
        path_x[left_only] = left_offset[left_only]
        right_only = (
            ~np.isfinite(path_x) & np.isfinite(right_offset)
        )
        path_x[right_only] = right_offset[right_only]

        valid = np.isfinite(path_x)
        if np.count_nonzero(valid) < 3:
            return None, "insufficient_path"
        path_array = np.column_stack(
            (path_x[valid], y_values[valid])
        ).astype(np.float32)
        margin = self.config.warp_width * 0.15
        in_range = (
            (path_array[:, 0] >= -margin)
            & (
                path_array[:, 0]
                < self.config.warp_width + margin
            )
        )
        path_array = path_array[in_range]
        if len(path_array) < 3:
            return None, "path_out_of_range"
        if left is not None and right is not None:
            reason = "fused_boundaries"
        elif left is not None:
            reason = "left_boundary_normal_offset"
        else:
            reason = "right_boundary_normal_offset"
        return path_array, reason

    def _smooth_spatial(self, path: PointArray) -> PointArray:
        if path is None or len(path) < 3:
            return None
        try:
            coefficients = np.polyfit(path[:, 1], path[:, 0], 2)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return path
        y_values = self._path_y_values()
        x_values = np.polyval(coefficients, y_values)
        if not np.all(np.isfinite(x_values)):
            return None
        margin = self.config.warp_width * 0.15
        x_values = np.clip(
            x_values,
            -margin,
            self.config.warp_width + margin,
        )
        return np.column_stack(
            (x_values, y_values)
        ).astype(np.float32)

    def _smooth_temporal(
        self,
        path: PointArray,
        source: str,
    ) -> Tuple[PointArray, bool]:
        if path is None:
            self._missing_frames += 1
            if (
                self._last_path is not None
                and self._missing_frames <= self.config.max_missing_frames
            ):
                return self._last_path.copy(), True
            if self._missing_frames > self.config.max_missing_frames:
                self._last_path = None
                self._last_path_source = None
                self._path_transition_frames = 0
            return None, False

        self._missing_frames = 0
        if self._last_path is None:
            self._last_path = path.copy()
            self._last_path_source = source
            return path, False

        previous = self._last_path
        if (
            len(previous) != len(path)
            or not np.allclose(
                previous[:, 1], path[:, 1], atol=0.1
            )
        ):
            order = np.argsort(previous[:, 1])
            previous_x = np.interp(
                path[:, 1],
                previous[order, 1],
                previous[order, 0],
            )
            previous = np.column_stack(
                (previous_x, path[:, 1])
            ).astype(np.float32)

        max_step_px = (
            self.config.max_path_lateral_step_m
            * self.config.pixels_per_meter
        )
        raw_delta = path[:, 0] - previous[:, 0]
        source_changed = (
            self._last_path_source is not None
            and source != self._last_path_source
        )
        if source_changed or np.max(np.abs(raw_delta)) > max_step_px:
            self._path_transition_frames = max(
                self._path_transition_frames,
                self.config.path_transition_blend_frames,
            )

        smoothed = path.copy()
        alpha = self.config.path_ema_alpha
        if self._path_transition_frames > 0:
            alpha = min(
                alpha,
                1.0 / self.config.path_transition_blend_frames,
            )
            self._path_transition_frames -= 1
        filtered_delta = np.clip(
            alpha * raw_delta,
            -max_step_px,
            max_step_px,
        )
        smoothed[:, 0] = previous[:, 0] + filtered_delta
        self._last_path = smoothed.copy()
        self._last_path_source = source
        return smoothed, False

    def pixels_to_meters(self, path: PointArray) -> PointArray:
        if path is None:
            return None
        vehicle_x = self.config.warp_width * 0.5
        vehicle_y = self.config.warp_height - 1.0
        forward = (
            (vehicle_y - path[:, 1]) / self.config.pixels_per_meter
            + self.config.bev_reference_forward_offset_m
        )
        left = (vehicle_x - path[:, 0]) / self.config.pixels_per_meter
        return np.column_stack((forward, left)).astype(np.float32)

    def _raw_steering_deg(
        self,
        path_meters: PointArray,
        lookahead_m: float,
    ) -> float:
        if path_meters is None or not len(path_meters):
            return 0.0

        distances = np.linalg.norm(path_meters, axis=1)
        target_index = int(
            np.argmin(np.abs(distances - lookahead_m))
        )
        forward, left = path_meters[target_index]
        distance = max(float(distances[target_index]), 1e-3)
        heading_error = math.atan2(
            float(left), max(float(forward), 1e-3)
        )
        raw_steering_rad = math.atan2(
            2.0
            * self.config.wheelbase_m
            * math.sin(heading_error),
            distance,
        )
        return float(
            np.clip(
                math.degrees(raw_steering_rad),
                -self.config.max_steering_deg,
                self.config.max_steering_deg,
            )
        )

    @staticmethod
    def _nearest_path_distance(
        path_meters: PointArray,
        lookahead_m: float,
    ) -> float:
        if path_meters is None or not len(path_meters):
            return 0.0
        distances = np.linalg.norm(path_meters, axis=1)
        target_index = int(
            np.argmin(np.abs(distances - lookahead_m))
        )
        return float(distances[target_index])

    def _lookahead_for_steering(self, steering_deg: float) -> float:
        steering_ratio = float(
            np.clip(
                abs(steering_deg) / self.config.max_steering_deg,
                0.0,
                1.0,
            )
        )
        lookahead_range = (
            self.config.lookahead_max_m
            - self.config.lookahead_min_m
        )
        return (
            self.config.lookahead_max_m
            - lookahead_range * steering_ratio
        )

    def _steering(
        self, path_meters: PointArray
    ) -> Tuple[float, float]:
        # Use the previous filtered command to choose this frame's LD. This
        # avoids an algebraic steering/LD loop and keeps LD changes smooth.
        lookahead_m = self._lookahead_for_steering(
            self._last_steering_deg
        )
        raw_steering_deg = self._raw_steering_deg(
            path_meters, lookahead_m
        )

        if (
            abs(raw_steering_deg)
            < self.config.steering_deadband_deg
        ):
            raw_steering_deg = 0.0

        filtered = (
            self.config.steering_ema_alpha * raw_steering_deg
            + (1.0 - self.config.steering_ema_alpha)
            * self._last_steering_deg
        )
        steering_step = float(
            np.clip(
                filtered - self._last_steering_deg,
                -self.config.max_steering_change_deg,
                self.config.max_steering_change_deg,
            )
        )
        self._last_steering_deg = float(
            np.clip(
                self._last_steering_deg + steering_step,
                -self.config.max_steering_deg,
                self.config.max_steering_deg,
            )
        )
        lookahead_m = self._lookahead_for_steering(
            self._last_steering_deg
        )
        return math.radians(self._last_steering_deg), lookahead_m

    def _debug_image(
        self,
        bev: np.ndarray,
        total_mask: np.ndarray,
        groups: List[LaneGroup],
        left: Optional[LaneGroup],
        right: Optional[LaneGroup],
        path: PointArray,
        reason: str,
        fallback: bool,
        steering_rad: float,
        lookahead_m: float,
        lookahead_target_m: float,
        selection_mode: str,
    ) -> np.ndarray:
        colored_mask = bev.copy()
        colored_mask[total_mask > 0] = (0, 180, 0)
        debug = cv2.addWeighted(bev, 0.70, colored_mask, 0.30, 0.0)

        for group in groups:
            color = (100, 100, 100)
            if group is left:
                color = (255, 0, 0)
            elif group is right:
                color = (0, 255, 255)
            points = np.rint(group["points"]).astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(debug, [points], False, color, 2)
                label_point = tuple(points[len(points) // 2])
                piece_count = len(group["pieces"])
                line_type = (
                    f"DASHED x{piece_count}"
                    if piece_count
                    >= self.config.dashed_piece_threshold
                    else "SOLID"
                )
                cv2.putText(
                    debug,
                    line_type,
                    label_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if path is not None and len(path) >= 2:
            path_points = np.rint(path).astype(np.int32)
            cv2.polylines(
                debug, [path_points], False, (0, 0, 255), 4
            )
            metric_path = self.pixels_to_meters(path)
            distances = np.linalg.norm(metric_path, axis=1)
            target_index = int(
                np.argmin(
                    np.abs(distances - lookahead_m)
                )
            )
            cv2.circle(
                debug,
                tuple(path_points[target_index]),
                8,
                (255, 255, 255),
                -1,
            )

        cv2.circle(
            debug,
            (
                self.config.warp_width // 2,
                self.config.warp_height - 1,
            ),
            7,
            (255, 255, 255),
            -1,
        )
        lines = (
            f"path: {reason}",
            (
                f"groups: {len(groups)} fallback: {fallback} "
                f"select: {selection_mode}"
            ),
            f"steering: {math.degrees(steering_rad):+.2f} deg",
            (
                f"dynamic LD: {lookahead_m:.2f} m "
                f"(target {lookahead_target_m:.2f} m)"
            ),
            f"device: {self.device}",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                debug,
                text,
                (12, 24 + 23 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return debug

    def plan_mask(self, mask: np.ndarray) -> PathPlanResult:
        """Plan only; detection and vehicle control belong to other nodes."""

        pieces, _ = self._extract_mask_pieces(mask)
        groups = self._group_pieces(pieces)
        which_lane = self._classify_which_lane(groups)
        (
            left,
            right,
            selection_mode,
            dashed_region_count,
        ) = self._choose_boundaries(groups)
        raw_path, reason = self._build_path(left, right)
        if selection_mode in (
            "solid_left_preferred",
            "solid_right_preferred",
        ):
            reason = selection_mode
        spatial_path = self._smooth_spatial(raw_path)
        final_path, fallback = self._smooth_temporal(
            spatial_path,
            f"{selection_mode}:{reason}",
        )
        fallback = fallback or self._using_tracked_boundary
        return PathPlanResult(
            path_pixels=final_path,
            path_meters=self.pixels_to_meters(final_path),
            reason=reason,
            used_fallback=fallback,
            group_count=len(groups),
            dashed_region_count=dashed_region_count,
            selection_mode=selection_mode,
            which_lane=which_lane,
        )

    def process(self, frame: np.ndarray) -> LaneResult:
        if self.model is None:
            raise RuntimeError(
                "process(frame) requires a model; use plan_mask(mask) "
                "in the path-planning node"
            )
        bev = self.make_bev(frame)
        started = time.perf_counter()
        inference_result = self.model.predict(
            source=bev,
            imgsz=self.config.image_size,
            conf=self.config.confidence,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]
        inference_ms = (time.perf_counter() - started) * 1000.0
        segmentation_image = inference_result.plot(
            labels=True,
            boxes=False,
            masks=True,
            conf=True,
        )

        pieces, total_mask = self._extract_pieces(
            inference_result, bev.shape
        )
        groups = self._group_pieces(pieces)
        which_lane = self._classify_which_lane(groups)
        (
            left,
            right,
            selection_mode,
            dashed_region_count,
        ) = self._choose_boundaries(groups)
        raw_path, reason = self._build_path(left, right)
        if selection_mode in (
            "solid_left_preferred",
            "solid_right_preferred",
        ):
            reason = selection_mode
        spatial_path = self._smooth_spatial(raw_path)
        final_path, fallback = self._smooth_temporal(
            spatial_path,
            f"{selection_mode}:{reason}",
        )
        fallback = fallback or self._using_tracked_boundary
        path_meters = self.pixels_to_meters(final_path)
        steering_rad, lookahead_m = self._steering(path_meters)
        lookahead_target_m = self._nearest_path_distance(
            path_meters, lookahead_m
        )
        target_speed = (
            self.config.target_speed_mps
            if final_path is not None
            else 0.0
        )
        debug = self._debug_image(
            bev,
            total_mask,
            groups,
            left,
            right,
            final_path,
            reason,
            fallback,
            steering_rad,
            lookahead_m,
            lookahead_target_m,
            selection_mode,
        )
        return LaneResult(
            segmentation_image=segmentation_image,
            debug_image=debug,
            path_pixels=final_path,
            path_meters=path_meters,
            steering_rad=steering_rad,
            lookahead_m=lookahead_m,
            lookahead_target_m=lookahead_target_m,
            target_speed_mps=target_speed,
            reason=reason,
            used_fallback=fallback,
            group_count=len(groups),
            dashed_region_count=dashed_region_count,
            selection_mode=selection_mode,
            which_lane=which_lane,
            inference_ms=inference_ms,
        )


class YoloLaneDriver(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("yolo_lane_driver")
        parameter_defaults = {
            "model_path": str(args.model),
            "bev_params": str(args.bev_params),
            "image_topic": args.image_topic,
            "path_topic": args.path_topic,
            "segmentation_topic": args.segmentation_topic,
            "debug_topic": args.debug_topic,
            "status_topic": args.status_topic,
            "which_lane_topic": args.which_lane_topic,
            "cmd_vel_topic": args.cmd_vel_topic,
            "path_frame_id": args.path_frame_id,
            "confidence": args.confidence,
            "image_size": args.image_size,
            "device": args.device,
            "calibration_width": args.calibration_width,
            "calibration_height": args.calibration_height,
            "pixels_per_meter": args.pixels_per_meter,
            "lane_width_m": args.lane_width_m,
            "min_component_area": args.min_component_area,
            "center_sample_step": args.center_sample_step,
            "same_line_threshold_px": args.same_line_threshold_px,
            "min_group_span_px": args.min_group_span_px,
            "min_group_area": args.min_group_area,
            "dashed_piece_threshold": args.dashed_piece_threshold,
            "prefer_solid_when_dashed": (
                args.prefer_solid_when_dashed
            ),
            "lane_track_max_age_frames": (
                args.lane_track_max_age_frames
            ),
            "lane_track_match_threshold_px": (
                args.lane_track_match_threshold_px
            ),
            "solid_enter_frames": args.solid_enter_frames,
            "solid_exit_frames": args.solid_exit_frames,
            "path_top_y": args.path_top_y,
            "path_bottom_margin": args.path_bottom_margin,
            "path_step_px": args.path_step_px,
            "path_resample_step_px": args.path_resample_step_px,
            "max_missing_frames": args.max_missing_frames,
            "path_ema_alpha": args.path_ema_alpha,
            "path_transition_blend_frames": (
                args.path_transition_blend_frames
            ),
            "max_path_lateral_step_m": (
                args.max_path_lateral_step_m
            ),
            "bev_reference_forward_offset_m": (
                args.bev_reference_forward_offset_m
            ),
            "lookahead_min_m": args.lookahead_min_m,
            "lookahead_max_m": args.lookahead_max_m,
            "lookahead_m": (
                -1.0 if args.lookahead_m is None else args.lookahead_m
            ),
            "wheelbase_m": args.wheelbase_m,
            "max_steer_deg": args.max_steer_deg,
            "steering_ema_alpha": args.steering_ema_alpha,
            "steering_deadband_deg": args.steering_deadband_deg,
            "max_steering_change_deg": (
                args.max_steering_change_deg
            ),
            "speed_mps": args.speed_mps,
            "hold_speed_scale": args.hold_speed_scale,
            "process_every_nth_frame": args.process_every_nth_frame,
            "debug_jpeg_quality": args.debug_jpeg_quality,
            "enable_drive": args.enable_drive,
            "display": args.display,
            "display_scale": args.display_scale,
        }
        for name, default_value in parameter_defaults.items():
            self.declare_parameter(name, default_value)

        def parameter(name: str) -> Any:
            return self.get_parameter(name).value

        model_path = Path(str(parameter("model_path")))
        image_topic = str(parameter("image_topic"))
        path_topic = str(parameter("path_topic"))
        segmentation_topic = str(parameter("segmentation_topic"))
        debug_topic = str(parameter("debug_topic"))
        status_topic = str(parameter("status_topic"))
        which_lane_topic = str(parameter("which_lane_topic"))
        cmd_vel_topic = str(parameter("cmd_vel_topic"))
        bev_params_path = Path(str(parameter("bev_params")))
        bev_parameters = load_bev_parameters(bev_params_path)
        fixed_lookahead_m = float(parameter("lookahead_m"))
        if fixed_lookahead_m > 0.0:
            lookahead_min_m = fixed_lookahead_m
            lookahead_max_m = fixed_lookahead_m
        else:
            lookahead_min_m = float(parameter("lookahead_min_m"))
            lookahead_max_m = float(parameter("lookahead_max_m"))

        config = LaneConfig(
            confidence=float(parameter("confidence")),
            image_size=int(parameter("image_size")),
            device=str(parameter("device")),
            calibration_width=int(parameter("calibration_width")),
            calibration_height=int(parameter("calibration_height")),
            source_points=tuple(bev_parameters.source_points.reshape(-1)),
            destination_points=tuple(
                bev_parameters.destination_points.reshape(-1)
            ),
            warp_width=bev_parameters.width,
            warp_height=bev_parameters.height,
            pixels_per_meter=float(parameter("pixels_per_meter")),
            lane_width_m=float(parameter("lane_width_m")),
            min_component_area=int(parameter("min_component_area")),
            center_sample_step=int(parameter("center_sample_step")),
            same_line_threshold_px=float(
                parameter("same_line_threshold_px")
            ),
            min_group_span_px=float(parameter("min_group_span_px")),
            min_group_area=int(parameter("min_group_area")),
            dashed_piece_threshold=int(
                parameter("dashed_piece_threshold")
            ),
            prefer_solid_when_dashed=bool(
                parameter("prefer_solid_when_dashed")
            ),
            lane_track_max_age_frames=int(
                parameter("lane_track_max_age_frames")
            ),
            lane_track_match_threshold_px=float(
                parameter("lane_track_match_threshold_px")
            ),
            solid_enter_frames=int(parameter("solid_enter_frames")),
            solid_exit_frames=int(parameter("solid_exit_frames")),
            path_top_y=int(parameter("path_top_y")),
            path_bottom_margin=int(parameter("path_bottom_margin")),
            path_step_px=int(parameter("path_step_px")),
            path_resample_step_px=int(
                parameter("path_resample_step_px")
            ),
            max_missing_frames=int(parameter("max_missing_frames")),
            path_ema_alpha=float(parameter("path_ema_alpha")),
            path_transition_blend_frames=int(
                parameter("path_transition_blend_frames")
            ),
            max_path_lateral_step_m=float(
                parameter("max_path_lateral_step_m")
            ),
            bev_reference_forward_offset_m=float(
                parameter("bev_reference_forward_offset_m")
            ),
            lookahead_min_m=lookahead_min_m,
            lookahead_max_m=lookahead_max_m,
            wheelbase_m=float(parameter("wheelbase_m")),
            max_steering_deg=float(parameter("max_steer_deg")),
            steering_ema_alpha=float(
                parameter("steering_ema_alpha")
            ),
            steering_deadband_deg=float(
                parameter("steering_deadband_deg")
            ),
            max_steering_change_deg=float(
                parameter("max_steering_change_deg")
            ),
            target_speed_mps=float(parameter("speed_mps")),
        )
        self.processor = SegmentationLaneProcessor(model_path, config)
        self.enable_drive = bool(parameter("enable_drive"))
        self.display = bool(parameter("display"))
        self.display_scale = max(
            0.1, float(parameter("display_scale"))
        )
        self.path_frame_id = str(parameter("path_frame_id"))
        self.debug_jpeg_quality = int(
            np.clip(int(parameter("debug_jpeg_quality")), 1, 100)
        )
        self.process_every_nth_frame = max(
            1, int(parameter("process_every_nth_frame"))
        )
        self.hold_speed_scale = float(
            np.clip(
                float(parameter("hold_speed_scale")), 0.0, 1.0
            )
        )
        self.frame_count = 0
        self.processed_count = 0
        self.last_log_time = 0.0

        self.path_publisher = self.create_publisher(
            PathMessage, path_topic, 10
        )
        self.segmentation_publisher = self.create_publisher(
            CompressedImage,
            segmentation_topic,
            qos_profile_sensor_data,
        )
        self.debug_publisher = self.create_publisher(
            CompressedImage,
            debug_topic,
            qos_profile_sensor_data,
        )
        self.status_publisher = self.create_publisher(
            String, status_topic, 10
        )
        self.which_lane_publisher = self.create_publisher(
            String, which_lane_topic, 10
        )
        self.cmd_publisher = self.create_publisher(
            Twist, cmd_vel_topic, 10
        )
        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"input={image_topic}, model={model_path}, "
            f"bev_params={bev_params_path}, "
            f"task={self.processor.model.task}, "
            f"classes={self.processor.model.names}, "
            f"device={self.processor.device}"
        )
        self.get_logger().info(
            f"path={path_topic}, "
            f"segmentation={segmentation_topic}, "
            f"path_debug={debug_topic}, "
            f"which_lane={which_lane_topic}, "
            f"cmd_vel={cmd_vel_topic}, "
            f"drive_enabled={self.enable_drive}"
        )
        self.get_logger().info(
            "vehicle geometry: "
            f"BEV reference is "
            f"{self.processor.config.bev_reference_forward_offset_m:.2f}m "
            "ahead of rear axle; dynamic LD="
            f"{self.processor.config.lookahead_min_m:.2f}.."
            f"{self.processor.config.lookahead_max_m:.2f}m"
        )
        if self.enable_drive:
            self.get_logger().warning(
                "Drive is ENABLED: this node may publish non-zero /cmd_vel."
            )
        else:
            self.get_logger().info(
                "Dry-run mode: path is generated but /cmd_vel remains zero."
            )

    def on_image(self, message: CompressedImage) -> None:
        self.frame_count += 1
        if (self.frame_count - 1) % self.process_every_nth_frame:
            return

        try:
            frame = decode_compressed_image(message)
            output = self.processor.process(frame)
        except Exception as exc:
            self.publish_stop()
            self.get_logger().error(
                f"Lane processing failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        self.processed_count += 1
        self.path_publisher.publish(
            self._path_message(output.path_meters, message)
        )
        self._publish_compressed_image(
            output.segmentation_image,
            message,
            self.segmentation_publisher,
            "segmentation",
        )
        self._publish_compressed_image(
            output.debug_image,
            message,
            self.debug_publisher,
            "path",
        )
        self._publish_command(output)
        self._publish_which_lane(output)
        self._publish_status(output)

        if self.display:
            segmentation_preview = cv2.resize(
                output.segmentation_image,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            path_preview = cv2.resize(
                output.debug_image,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("BEV segmentation", segmentation_preview)
            cv2.imshow("BEV generated path", path_preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

        now = time.monotonic()
        if now - self.last_log_time >= 2.0:
            path_points = (
                0
                if output.path_meters is None
                else len(output.path_meters)
            )
            self.get_logger().info(
                f"frames={self.processed_count}, "
                f"path_points={path_points}, "
                f"groups={output.group_count}, "
                f"dashed_regions={output.dashed_region_count}, "
                f"selection={output.selection_mode}, "
                f"which_lane={output.which_lane or 'unknown'}, "
                f"reason={output.reason}, "
                f"fallback={output.used_fallback}, "
                f"steer={math.degrees(output.steering_rad):+.1f}deg, "
                f"ld={output.lookahead_m:.2f}m, "
                f"ld_target={output.lookahead_target_m:.2f}m, "
                f"inference={output.inference_ms:.1f}ms"
            )
            self.last_log_time = now

    def _path_message(
        self,
        path_meters: PointArray,
        source: CompressedImage,
    ) -> PathMessage:
        message = PathMessage()
        message.header.stamp = source.header.stamp
        message.header.frame_id = self.path_frame_id
        if path_meters is None:
            return message

        for index, (forward, left) in enumerate(path_meters):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(forward)
            pose.pose.position.y = float(left)
            if len(path_meters) == 1:
                yaw = 0.0
            elif index < len(path_meters) - 1:
                delta = path_meters[index + 1] - path_meters[index]
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            else:
                delta = path_meters[index] - path_meters[index - 1]
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.poses.append(pose)
        return message

    def _publish_compressed_image(
        self,
        image: np.ndarray,
        source: CompressedImage,
        publisher: Any,
        image_name: str,
    ) -> None:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.debug_jpeg_quality],
        )
        if not success:
            self.get_logger().warning(
                f"Could not encode {image_name} image"
            )
            return
        message = CompressedImage()
        message.header = source.header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        publisher.publish(message)

    def _publish_command(self, output: LaneResult) -> None:
        if not self.enable_drive or output.path_meters is None:
            self.publish_stop()
            return
        speed = output.target_speed_mps
        if output.used_fallback:
            speed *= self.hold_speed_scale
        turn_ratio = min(
            1.0,
            abs(math.degrees(output.steering_rad))
            / self.processor.config.max_steering_deg,
        )
        speed *= max(0.25, 1.0 - 0.70 * turn_ratio)
        self.cmd_publisher.publish(
            make_twist(
                speed,
                output.steering_rad,
                self.processor.config.wheelbase_m,
            )
        )

    def _publish_status(self, output: LaneResult) -> None:
        message = {
            "path_valid": output.path_meters is not None,
            "reason": output.reason,
            "fallback": output.used_fallback,
            "lane_groups": output.group_count,
            "dashed_region_count": output.dashed_region_count,
            "selection_mode": output.selection_mode,
            "which_lane": output.which_lane,
            "steering_deg": round(
                math.degrees(output.steering_rad), 2
            ),
            "lookahead_m": round(output.lookahead_m, 3),
            "lookahead_target_m": round(
                output.lookahead_target_m, 3
            ),
            "bev_reference_forward_offset_m": round(
                self.processor.config.bev_reference_forward_offset_m,
                3,
            ),
            "inference_ms": round(output.inference_ms, 1),
            "drive_enabled": self.enable_drive,
        }
        self.status_publisher.publish(
            String(data=json.dumps(message, ensure_ascii=False))
        )

    def _publish_which_lane(self, output: LaneResult) -> None:
        if output.which_lane is None:
            return
        self.which_lane_publisher.publish(
            String(data=output.which_lane)
        )

    def publish_stop(self) -> None:
        self.cmd_publisher.publish(Twist())

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.publish_stop()
        if self.display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a lane path from usb_cam compressed images and "
            "best1.pt segmentation masks."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--bev-params",
        type=Path,
        default=DEFAULT_BEV_PARAMS,
        help="NPZ containing src_points, dst_points, warp_w, and warp_h",
    )
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--path-topic", default=DEFAULT_PATH_TOPIC)
    parser.add_argument(
        "--segmentation-topic", default=DEFAULT_SEGMENTATION_TOPIC
    )
    parser.add_argument("--debug-topic", default=DEFAULT_DEBUG_TOPIC)
    parser.add_argument("--status-topic", default=DEFAULT_STATUS_TOPIC)
    parser.add_argument(
        "--which-lane-topic", default=DEFAULT_WHICH_LANE_TOPIC
    )
    parser.add_argument("--cmd-vel-topic", default=DEFAULT_CMD_VEL_TOPIC)
    parser.add_argument("--path-frame-id", default="base_link")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, or a CUDA index such as 0",
    )
    parser.add_argument("--calibration-width", type=int, default=640)
    parser.add_argument("--calibration-height", type=int, default=480)
    parser.add_argument("--pixels-per-meter", type=float, default=600.0)
    parser.add_argument("--lane-width-m", type=float, default=0.90)
    parser.add_argument("--min-component-area", type=int, default=250)
    parser.add_argument("--center-sample-step", type=int, default=5)
    parser.add_argument(
        "--same-line-threshold-px", type=float, default=75.0
    )
    parser.add_argument("--min-group-span-px", type=float, default=80.0)
    parser.add_argument("--min-group-area", type=int, default=500)
    parser.add_argument("--dashed-piece-threshold", type=int, default=2)
    parser.add_argument(
        "--prefer-solid-when-dashed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lane-track-max-age-frames", type=int, default=7)
    parser.add_argument(
        "--lane-track-match-threshold-px",
        type=float,
        default=90.0,
    )
    parser.add_argument("--solid-enter-frames", type=int, default=4)
    parser.add_argument("--solid-exit-frames", type=int, default=6)
    parser.add_argument("--path-top-y", type=int, default=180)
    parser.add_argument("--path-bottom-margin", type=int, default=30)
    parser.add_argument("--path-step-px", type=int, default=10)
    parser.add_argument("--path-resample-step-px", type=int, default=5)
    parser.add_argument("--max-missing-frames", type=int, default=8)
    parser.add_argument("--path-ema-alpha", type=float, default=0.35)
    parser.add_argument(
        "--path-transition-blend-frames",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-path-lateral-step-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--bev-reference-forward-offset-m",
        type=float,
        default=1.04,
        help=(
            "Forward distance in meters from rear-axle center to the "
            "BEV bottom reference point."
        ),
    )
    parser.add_argument(
        "--lookahead-min-m",
        type=float,
        default=1.1,
        help="Dynamic LD used at maximum steering.",
    )
    parser.add_argument(
        "--lookahead-max-m",
        type=float,
        default=2.5,
        help="Dynamic LD used near straight driving.",
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=None,
        help=(
            "Optional fixed LD for backward compatibility; when given, "
            "it disables the dynamic LD range."
        ),
    )
    parser.add_argument("--wheelbase-m", type=float, default=0.545)
    parser.add_argument("--max-steer-deg", type=float, default=20.0)
    parser.add_argument("--steering-ema-alpha", type=float, default=0.35)
    parser.add_argument(
        "--steering-deadband-deg", type=float, default=0.8
    )
    parser.add_argument(
        "--max-steering-change-deg", type=float, default=3.0
    )
    parser.add_argument("--speed-mps", type=float, default=0.20)
    parser.add_argument("--hold-speed-scale", type=float, default=0.35)
    parser.add_argument("--process-every-nth-frame", type=int, default=1)
    parser.add_argument("--debug-jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--enable-drive",
        action="store_true",
        help="Allow non-zero cmd_vel output; default is safe dry-run.",
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-scale", type=float, default=1.0)
    return parser.parse_known_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"YOLO model not found: {args.model}")
    if not args.bev_params.is_file():
        raise FileNotFoundError(
            f"BEV parameter file not found: {args.bev_params}"
        )
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.process_every_nth_frame <= 0:
        raise ValueError("--process-every-nth-frame must be positive")
    if (
        args.path_step_px <= 0
        or args.path_resample_step_px <= 0
    ):
        raise ValueError("Path sampling steps must be positive")
    if args.lane_track_max_age_frames < 0:
        raise ValueError(
            "--lane-track-max-age-frames cannot be negative"
        )
    if args.lane_track_match_threshold_px <= 0.0:
        raise ValueError(
            "--lane-track-match-threshold-px must be positive"
        )
    if args.solid_enter_frames <= 0 or args.solid_exit_frames <= 0:
        raise ValueError(
            "Solid preference hysteresis frames must be positive"
        )
    if args.path_transition_blend_frames <= 0:
        raise ValueError(
            "--path-transition-blend-frames must be positive"
        )
    if args.max_path_lateral_step_m <= 0.0:
        raise ValueError(
            "--max-path-lateral-step-m must be positive"
        )
    if args.speed_mps < 0.0:
        raise ValueError("--speed-mps cannot be negative")
    if args.wheelbase_m <= 0.0:
        raise ValueError("--wheelbase-m must be positive")
    if not math.isfinite(args.bev_reference_forward_offset_m):
        raise ValueError(
            "--bev-reference-forward-offset-m must be finite"
        )
    if args.lookahead_m is not None and args.lookahead_m <= 0.0:
        raise ValueError("--lookahead-m must be positive")
    if (
        args.lookahead_min_m <= 0.0
        or args.lookahead_max_m < args.lookahead_min_m
    ):
        raise ValueError(
            "Lookahead range must satisfy "
            "0 < --lookahead-min-m <= --lookahead-max-m"
        )
    if args.max_steer_deg <= 0.0:
        raise ValueError("--max-steer-deg must be positive")


def main(argv: Optional[Sequence[str]] = None) -> None:
    cli_args, ros_args = parse_args(argv)
    validate_args(cli_args)
    rclpy.init(args=ros_args)
    node: Optional[YoloLaneDriver] = None
    try:
        node = YoloLaneDriver(cli_args)
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
