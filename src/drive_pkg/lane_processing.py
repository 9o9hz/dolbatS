#!/usr/bin/env python3
"""BEV lane-mask path planner core, shared by path_plan.py and drive_main.py.

The best1.pt model was trained on 640x640 BEV images with one ``lane`` class.
Incoming 640x480 usb_cam frames are therefore warped to that training view
before inference. Per-detection segmentation components are extracted,
tracked frame to frame as left/right boundaries, and fused into a smoothed
metric path in ``base_link`` coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage


def _default_bev_params_path() -> Path:
    source_path = (
        Path(__file__).resolve().parent / "resource" / "bev_params_0729.npz"
    )
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("drive_pkg"))
            / "resource"
            / "bev_params_0729.npz"
        )
    except (ImportError, LookupError):
        return source_path


def _default_model_path() -> Path:
    source_path = Path(__file__).resolve().parent / "resource" / "best1.pt"
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("drive_pkg"))
            / "resource"
            / "best1.pt"
        )
    except (ImportError, LookupError):
        return source_path


DEFAULT_MODEL = _default_model_path()
DEFAULT_BEV_PARAMS = _default_bev_params_path()

PointArray = Optional[np.ndarray]
LaneGroup = Dict[str, Any]


def resolve_data_path(value: str, fallback: Path) -> Path:
    """Resolve absolute, current-working-dir, or workspace-relative data."""
    normalized = str(value).strip()
    if not normalized:
        return fallback
    requested = Path(normalized).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(
            Path(__file__).resolve().parents[2] / requested
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return requested


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
    pixels_per_meter: float = 254.0
    lane_width_m: float = 0.85
    min_boundary_spacing_m: float = 0.80
    max_boundary_spacing_m: float = 0.90
    min_component_area: int = 250
    center_sample_step: int = 5
    same_line_threshold_px: float = 75.0
    min_group_span_px: float = 80.0
    min_group_area: int = 500
    dashed_piece_threshold: int = 2
    prefer_solid_when_dashed: bool = True
    initial_lane: str = "auto"
    lane_state_confirm_frames: int = 3
    lane_track_max_age_frames: int = 7
    lane_track_match_threshold_px: float = 90.0
    solid_enter_frames: int = 4
    solid_exit_frames: int = 6
    path_top_y: int = 180
    path_bottom_margin: int = 30
    path_step_px: int = 10
    path_resample_step_px: int = 5
    path_polynomial_degree: int = 3
    path_ema_alpha: float = 0.35
    path_transition_blend_frames: int = 6
    max_path_lateral_step_m: float = 0.04
    max_missing_frames: int = 8
    # Morphological close kernel size used when isolating one detection's
    # largest mask component within its own bounding box.
    bbox_close_ksize: int = 5
    # Forward distance from the rear-axle center to the BEV bottom reference.
    bev_reference_forward_offset_m: float = 1.04


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
    detected_lines: List[Dict[str, Any]]
    left_boundary_pixels: PointArray
    right_boundary_pixels: PointArray


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
        required = {"src_points", "dst_points"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"BEV parameter keys missing: {sorted(missing)}")
        width_key = (
            "warp_width" if "warp_width" in data.files else "warp_w"
        )
        height_key = (
            "warp_height" if "warp_height" in data.files else "warp_h"
        )
        size_missing = [
            key
            for key in (width_key, height_key)
            if key not in data.files
        ]
        if size_missing:
            raise KeyError(
                "BEV parameter keys missing: expected warp_w/warp_h "
                "or warp_width/warp_height"
            )

        source_points = np.asarray(data["src_points"], dtype=np.float32)
        destination_points = np.asarray(
            data["dst_points"], dtype=np.float32
        )
        width = _scalar_int(data, width_key)
        height = _scalar_int(data, height_key)

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


@dataclass(frozen=True)
class CameraCalibration:
    """Precomputed undistortion map, as produced by ``cv2.calibrateCamera``."""

    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray


def load_calibration(path: Path) -> CameraCalibration:
    """Load a ``camera_calibration.npz`` file with keys ``camera_matrix``,
    ``distortion_coefficients``, ``new_camera_matrix``, ``image_width``,
    ``image_height``."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "camera_matrix",
            "distortion_coefficients",
            "new_camera_matrix",
            "image_width",
            "image_height",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Calibration keys missing: {sorted(missing)}")

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(
            data["distortion_coefficients"],
            dtype=np.float64,
        ).reshape(-1)
        new_camera_matrix = np.asarray(
            data["new_camera_matrix"],
            dtype=np.float64,
        )
        width = _scalar_int(data, "image_width")
        height = _scalar_int(data, "image_height")

    if (
        camera_matrix.shape != (3, 3)
        or new_camera_matrix.shape != (3, 3)
        or distortion.size not in (4, 5, 8, 12, 14)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"Invalid calibration data: {path}")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return CameraCalibration(width, height, map_x, map_y)


def undistort_frame(
    calibration: CameraCalibration,
    frame: np.ndarray,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if (width, height) != (calibration.width, calibration.height):
        raise ValueError(
            "Frame resolution does not match calibration resolution: "
            f"{width}x{height} != "
            f"{calibration.width}x{calibration.height}"
        )
    return cv2.remap(
        frame,
        calibration.map_x,
        calibration.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def decode_compressed_image(message: CompressedImage) -> np.ndarray:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode compressed camera image")
    return frame


class SegmentationLaneProcessor:
    """Turn YOLO lane masks into a smoothed local metric path."""

    # A bbox-restricted component whose area is implausibly larger than the
    # owning instance's own mask footprint has likely bled into a
    # neighboring detection's pixels via the combined mask; see
    # ``_extract_bbox_pieces``.
    _BBOX_CONTAMINATION_FACTOR = 3.0

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
        configured_lane = str(config.initial_lane).strip().lower()
        self._current_lane: Optional[str] = (
            configured_lane
            if configured_lane in ("lane_1", "lane_2")
            else None
        )
        self._lane_candidate: Optional[str] = None
        self._lane_candidate_streak = 0

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
        if (
            cfg.min_boundary_spacing_m <= 0.0
            or cfg.min_boundary_spacing_m > cfg.lane_width_m
        ):
            raise ValueError(
                "min_boundary_spacing_m must satisfy "
                "0 < minimum <= lane_width_m"
            )
        if cfg.max_boundary_spacing_m < cfg.lane_width_m:
            raise ValueError(
                "max_boundary_spacing_m must be >= lane_width_m"
            )
        if not math.isfinite(cfg.bev_reference_forward_offset_m):
            raise ValueError(
                "bev_reference_forward_offset_m must be finite"
            )
        if (
            cfg.center_sample_step <= 0
            or cfg.path_step_px <= 0
            or cfg.path_resample_step_px <= 0
        ):
            raise ValueError("Path sampling steps must be positive")
        if cfg.dashed_piece_threshold < 2:
            raise ValueError("dashed_piece_threshold must be at least 2")
        if str(cfg.initial_lane).strip().lower() not in (
            "auto",
            "lane_1",
            "lane_2",
        ):
            raise ValueError(
                "initial_lane must be auto, lane_1, or lane_2"
            )
        if cfg.lane_state_confirm_frames <= 0:
            raise ValueError(
                "lane_state_confirm_frames must be positive"
            )
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
        if cfg.path_polynomial_degree not in (2, 3):
            raise ValueError(
                "path_polynomial_degree must be 2 or 3"
            )
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
        if cfg.bbox_close_ksize < 1:
            raise ValueError("bbox_close_ksize must be at least 1")

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

    @staticmethod
    def _extract_largest_component_in_bbox(
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        min_area: int,
        close_ksize: int,
    ) -> Optional[np.ndarray]:
        """Isolate the single largest connected component within one
        detection's bounding box, discarding everything else in the ROI
        (noise, and any bleed-through from a neighboring detection's
        pixels in the combined mask)."""

        height, width = mask.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None

        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_ksize, close_ksize)
        )
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        component_count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(closed, connectivity=8)
        )
        if component_count <= 1:
            return None

        largest_label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
        if largest_area < min_area:
            return None

        component = np.zeros_like(mask, dtype=np.uint8)
        component[y1:y2, x1:x2][labels == largest_label] = 1
        return component

    def _extract_bbox_pieces(
        self,
        mask: np.ndarray,
        instances: Sequence[Dict[str, Any]],
    ) -> Tuple[List[LaneGroup], np.ndarray]:
        """Extract one piece per YOLO detection, restricted to that
        detection's own bounding box, instead of connected-components over
        the whole combined mask (see ``_extract_mask_pieces``)."""

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("Lane mask must be a non-empty mono image")

        total_mask = (mask > 0).astype(np.uint8)
        reference_y = total_mask.shape[0] * 0.88
        pieces: List[LaneGroup] = []
        for instance in instances:
            bbox = (
                int(instance["x_min"]),
                int(instance["y_min"]),
                int(instance["x_max"]),
                int(instance["y_max"]),
            )
            component = self._extract_largest_component_in_bbox(
                total_mask,
                bbox,
                self.config.min_component_area,
                self.config.bbox_close_ksize,
            )
            if component is None:
                continue
            area = int(np.count_nonzero(component))
            pixel_count = instance.get("pixel_count")
            if (
                pixel_count is not None
                and area > self._BBOX_CONTAMINATION_FACTOR * pixel_count
            ):
                continue
            points = self._component_center_points(
                component, self.config.center_sample_step
            )
            curve = self._fit_curve(points)
            if points is None or curve is None:
                continue
            y_min = float(np.min(points[:, 1]))
            y_max = float(np.max(points[:, 1]))
            x_reference_y = float(np.clip(reference_y, y_min, y_max))
            pieces.append(
                {
                    "points": points,
                    "curve": curve,
                    "y_min": y_min,
                    "y_max": y_max,
                    "area": area,
                    "confidence": float(instance.get("confidence", 1.0)),
                    "x_ref": self._curve_x(curve, x_reference_y),
                }
            )
        return pieces, total_mask

    def _extract_mask_pieces(
        self,
        mask: np.ndarray,
        instances: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[LaneGroup], np.ndarray]:
        """Extract mask components and attach overlapping YOLO semantics."""

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
            semantic_type: Optional[str] = None
            semantic_confidence = 0.0
            best_semantic_score = 0.0
            if instances:
                for instance in instances:
                    class_name = str(
                        instance.get("class_name", "")
                    ).strip().lower()
                    if "dashed" in class_name:
                        candidate_type = "DASHED"
                    elif "solid" in class_name:
                        candidate_type = "SOLID"
                    else:
                        continue
                    x1 = max(0, int(instance.get("x_min", 0)))
                    y1 = max(0, int(instance.get("y_min", 0)))
                    x2 = min(
                        total_mask.shape[1],
                        int(instance.get("x_max", 0)) + 1,
                    )
                    y2 = min(
                        total_mask.shape[0],
                        int(instance.get("y_max", 0)) + 1,
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue
                    overlap = int(
                        np.count_nonzero(component[y1:y2, x1:x2])
                    )
                    confidence = float(instance.get("confidence", 0.0))
                    score = overlap * max(confidence, 0.0)
                    if score > best_semantic_score:
                        best_semantic_score = score
                        semantic_type = candidate_type
                        semantic_confidence = confidence
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
                    "confidence": semantic_confidence,
                    "semantic_type": semantic_type,
                    "semantic_confidence": semantic_confidence,
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

    def _group_type(
        self, group: LaneGroup
    ) -> Tuple[str, Optional[float], str]:
        votes = {"DASHED": 0.0, "SOLID": 0.0}
        confidences: Dict[str, List[float]] = {
            "DASHED": [],
            "SOLID": [],
        }
        for piece in group["pieces"]:
            semantic_type = piece.get("semantic_type")
            confidence = float(
                piece.get("semantic_confidence", 0.0)
            )
            if semantic_type in votes and confidence > 0.0:
                votes[semantic_type] += confidence
                confidences[semantic_type].append(confidence)

        if votes["DASHED"] > 0.0 or votes["SOLID"] > 0.0:
            lane_type = max(votes, key=votes.get)
            return (
                lane_type,
                float(np.mean(confidences[lane_type])),
                "yolo",
            )

        fallback_type = (
            "DASHED"
            if len(group["pieces"])
            >= self.config.dashed_piece_threshold
            else "SOLID"
        )
        return fallback_type, None, "piece_count"

    def _group_is_dashed(self, group: LaneGroup) -> bool:
        lane_type, _, _ = self._group_type(group)
        return lane_type == "DASHED"

    def _filter_boundary_spacing(
        self,
        left: Optional[LaneGroup],
        right: Optional[LaneGroup],
    ) -> Tuple[Optional[LaneGroup], Optional[LaneGroup]]:
        """Keep one boundary when a detected pair is physically too close."""
        if left is None or right is None:
            return left, right
        spacing_px = float(right["x_ref"]) - float(left["x_ref"])
        minimum_px = (
            self.config.min_boundary_spacing_m
            * self.config.pixels_per_meter
        )
        maximum_px = (
            self.config.max_boundary_spacing_m
            * self.config.pixels_per_meter
        )
        if minimum_px <= spacing_px <= maximum_px:
            return left, right

        if self._group_reliability(left) >= self._group_reliability(right):
            return left, None
        return None, right

    def _group_reliability(
        self, group: LaneGroup
    ) -> Tuple[float, float, float]:
        _, confidence, source = self._group_type(group)
        semantic_score = (
            float(confidence)
            if source == "yolo" and confidence is not None
            else 0.0
        )
        return (
            semantic_score,
            float(group.get("span", 0.0)),
            float(group.get("area", 0.0)),
        )

    def _solid_dashed_solid_topology(
        self, groups: List[LaneGroup]
    ) -> Tuple[
        Optional[LaneGroup],
        Optional[LaneGroup],
        Optional[LaneGroup],
    ]:
        """Find the most plausible SOLID-DASHED-SOLID road topology."""
        usable = [
            group
            for group in groups
            if self._group_has_path_overlap(group)
        ]
        dashed_groups = [
            group
            for group in usable
            if self._group_is_dashed(group)
        ]
        solid_groups = [
            group
            for group in usable
            if not self._group_is_dashed(group)
        ]
        if not dashed_groups or not solid_groups:
            return None, None, None

        expected_spacing = (
            self.config.lane_width_m
            * self.config.pixels_per_meter
        )
        minimum_spacing = (
            self.config.min_boundary_spacing_m
            * self.config.pixels_per_meter
        )
        maximum_spacing = (
            self.config.max_boundary_spacing_m
            * self.config.pixels_per_meter
        )
        candidates = []
        for dashed in dashed_groups:
            dashed_x = float(dashed["x_ref"])
            left_options = [
                group
                for group in solid_groups
                if dashed_x - float(group["x_ref"])
                >= minimum_spacing
                and dashed_x - float(group["x_ref"])
                <= maximum_spacing
            ]
            right_options = [
                group
                for group in solid_groups
                if float(group["x_ref"]) - dashed_x
                >= minimum_spacing
                and float(group["x_ref"]) - dashed_x
                <= maximum_spacing
            ]
            left = (
                min(
                    left_options,
                    key=lambda group: dashed_x
                    - float(group["x_ref"]),
                )
                if left_options
                else None
            )
            right = (
                min(
                    right_options,
                    key=lambda group: float(group["x_ref"])
                    - dashed_x,
                )
                if right_options
                else None
            )
            if left is None and right is None:
                continue
            spacing_error = 0.0
            if left is not None:
                spacing_error += abs(
                    dashed_x
                    - float(left["x_ref"])
                    - expected_spacing
                )
            if right is not None:
                spacing_error += abs(
                    float(right["x_ref"])
                    - dashed_x
                    - expected_spacing
                )
            completeness = int(left is not None) + int(right is not None)
            candidates.append(
                (-completeness, spacing_error, dashed_x, left, dashed, right)
            )
        if not candidates:
            return None, None, None
        _, _, _, left, dashed, right = min(
            candidates, key=lambda item: item[:3]
        )
        return left, dashed, right

    def _infer_lane_from_groups(
        self, groups: List[LaneGroup]
    ) -> Optional[str]:
        solid_left, dashed, solid_right = (
            self._solid_dashed_solid_topology(groups)
        )
        if dashed is not None and (
            solid_left is not None or solid_right is not None
        ):
            center_x = self.config.warp_width * 0.5
            if center_x < float(dashed["x_ref"]):
                return "lane_1"
            return "lane_2"

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

        left_is_dashed = self._group_is_dashed(
            nearest_groups["left"]
        )
        right_is_dashed = self._group_is_dashed(
            nearest_groups["right"]
        )
        if not left_is_dashed and right_is_dashed:
            return "lane_1"
        if left_is_dashed and not right_is_dashed:
            return "lane_2"
        return None

    def _classify_which_lane(
        self, groups: List[LaneGroup]
    ) -> Optional[str]:
        """Latch the starting lane; never switch without an explicit command."""
        if self._current_lane is not None:
            return self._current_lane

        inferred = self._infer_lane_from_groups(groups)
        if inferred is None:
            self._lane_candidate = None
            self._lane_candidate_streak = 0
            return None
        if inferred == self._lane_candidate:
            self._lane_candidate_streak += 1
        else:
            self._lane_candidate = inferred
            self._lane_candidate_streak = 1
        if (
            self._lane_candidate_streak
            >= self.config.lane_state_confirm_frames
        ):
            self._current_lane = inferred
        return self._current_lane

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
        """Use local curve-distance association without stale geometry."""
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
            assignments[side] = initialized
            used_group_ids.add(id(initialized))

        return assignments.get("left"), assignments.get("right")

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
            if not self._group_is_dashed(group)
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
                return candidate, candidate_side
            else:
                self._solid_missing_frames += 1

            if (
                self._solid_missing_frames
                < self.config.solid_exit_frames
            ):
                # Preserve only the preference latch. Never feed a stale
                # boundary into a path with current-frame geometry.
                return None, None
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
            (
                max(
                    len(group["pieces"]),
                    self.config.dashed_piece_threshold,
                )
                for group in groups
                if self._group_is_dashed(group)
            ),
            default=0,
        )
        solid_left, dashed, solid_right = (
            self._solid_dashed_solid_topology(groups)
        )
        if dashed is not None:
            lane_state = (
                self._current_lane
                or self._infer_lane_from_groups(groups)
            )
            if lane_state == "lane_1":
                self._using_tracked_boundary = False
                return (
                    solid_left,
                    dashed,
                    (
                        "lane1_solid_dashed_hold"
                        if solid_left is not None
                        else "lane1_dashed_right_boundary_hold"
                    ),
                    dashed_region_count,
                )
            if lane_state == "lane_2":
                self._using_tracked_boundary = False
                return (
                    dashed,
                    solid_right,
                    (
                        "lane2_dashed_solid_hold"
                        if solid_right is not None
                        else "lane2_dashed_left_boundary_hold"
                    ),
                    dashed_region_count,
                )

        # Both semantic types are visible but no pair satisfies the
        # 800--900 mm geometry. Treat one as a likely false detection and
        # follow only the group with the stronger original YOLO confidence.
        semantic_dashed = [
            group
            for group in groups
            if self._group_is_dashed(group)
            and self._group_has_path_overlap(group)
        ]
        semantic_solid = [
            group
            for group in groups
            if not self._group_is_dashed(group)
            and self._group_has_path_overlap(group)
        ]
        lane_state = self._current_lane or self._infer_lane_from_groups(groups)
        if semantic_dashed and semantic_solid and lane_state is not None:
            strongest = max(
                semantic_dashed + semantic_solid,
                key=self._group_reliability,
            )
            strongest_is_dashed = self._group_is_dashed(strongest)
            self._using_tracked_boundary = False
            if lane_state == "lane_2":
                return (
                    strongest if strongest_is_dashed else None,
                    strongest if not strongest_is_dashed else None,
                    "invalid_spacing_yolo_confidence",
                    dashed_region_count,
                )
            return (
                strongest if not strongest_is_dashed else None,
                strongest if strongest_is_dashed else None,
                "invalid_spacing_yolo_confidence",
                dashed_region_count,
            )

        if self._current_lane in ("lane_1", "lane_2"):
            center_x = self.config.warp_width * 0.5
            dashed_candidates = [
                group
                for group in groups
                if self._group_is_dashed(group)
                and self._group_has_path_overlap(group)
            ]
            if self._current_lane == "lane_2":
                left_dashed = [
                    group
                    for group in dashed_candidates
                    if float(group["x_ref"]) < center_x
                ]
                if left_dashed:
                    self._using_tracked_boundary = False
                    return (
                        max(
                            left_dashed,
                            key=lambda group: float(group["x_ref"]),
                        ),
                        None,
                        "lane2_dashed_left_boundary_hold",
                        dashed_region_count,
                    )
            else:
                right_dashed = [
                    group
                    for group in dashed_candidates
                    if float(group["x_ref"]) >= center_x
                ]
                if right_dashed:
                    self._using_tracked_boundary = False
                    return (
                        None,
                        min(
                            right_dashed,
                            key=lambda group: float(group["x_ref"]),
                        ),
                        "lane1_dashed_right_boundary_hold",
                        dashed_region_count,
                    )

        left, right = self._filter_boundary_spacing(
            *self._update_lane_tracks(groups)
        )
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

        # No current boundary means _build_path() returns no path and
        # _smooth_temporal() alone may reuse the last complete path.
        self._using_tracked_boundary = False
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

    @staticmethod
    def _resample_polyline_by_y(
        xs: np.ndarray,
        ys: np.ndarray,
        y_min: int,
        y_max: int,
        y_step: int = 1,
    ) -> Tuple[PointArray, PointArray]:
        """Interpolate a (possibly sparse) polyline onto a uniform y-step
        grid, ported from yolotl_ros2's ``resample_polyline_by_y``."""

        if xs is None or ys is None or len(xs) < 2:
            return None, None

        ys = np.asarray(ys, dtype=np.float32)
        xs = np.asarray(xs, dtype=np.float32)
        order = np.argsort(ys)
        ys = ys[order]
        xs = xs[order]

        y_query = np.arange(y_min, y_max + 1, y_step, dtype=np.float32)
        valid = (y_query >= ys[0]) & (y_query <= ys[-1])
        y_query = y_query[valid]
        if len(y_query) < 2:
            return None, None

        x_query = np.interp(y_query, ys, xs).astype(np.float32)
        return x_query, y_query

    def _densify_group_points(
        self, group: Optional[LaneGroup]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if group is None:
            return None
        points = group["points"]
        y_min = int(np.floor(np.min(points[:, 1])))
        y_max = int(np.ceil(np.max(points[:, 1])))
        dense_xs, dense_ys = self._resample_polyline_by_y(
            points[:, 0], points[:, 1], y_min, y_max, y_step=1
        )
        if dense_xs is None or len(dense_xs) < 2:
            return None
        return dense_xs, dense_ys

    @staticmethod
    def _offset_polyline_points(
        xs: np.ndarray,
        ys: np.ndarray,
        offset_px: float,
        direction: float,
        sample_step: int = 20,
        window: int = 30,
    ) -> Tuple[PointArray, PointArray]:
        """Offset a polyline by its local normal direction, ported from
        yolotl_ros2's ``offset_polyline_points``. Works off point-window
        tangents rather than a fitted curve derivative, so it stays robust
        where a quadratic fit would badly extrapolate."""

        if xs is None or ys is None or len(xs) < 2:
            return None, None

        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        n = len(xs)
        if n <= sample_step:
            sample_step = max(1, n // 2)

        indices = np.arange(0, n, sample_step)
        if indices[-1] != n - 1:
            indices = np.append(indices, n - 1)

        sparse_xo: List[float] = []
        sparse_yo: List[float] = []
        for i in indices:
            idx_prev = max(0, i - window)
            idx_next = min(n - 1, i + window)
            if idx_prev == idx_next:
                dx, dy = 0.0, 1.0
            else:
                dx = float(xs[idx_next] - xs[idx_prev])
                dy = float(ys[idx_next] - ys[idx_prev])
            norm = math.sqrt(dx * dx + dy * dy) + 1e-6
            nx = dy / norm
            ny = -dx / norm
            sparse_xo.append(float(xs[i]) + direction * offset_px * nx)
            sparse_yo.append(float(ys[i]) + direction * offset_px * ny)

        all_indices = np.arange(n)
        out_xs = np.interp(all_indices, indices, sparse_xo).astype(
            np.float32
        )
        out_ys = np.interp(all_indices, indices, sparse_yo).astype(
            np.float32
        )

        if len(out_xs) >= 3:
            k_size = 5
            padded = np.pad(
                out_xs, (k_size // 2, k_size // 2), mode="edge"
            )
            out_xs = np.convolve(
                padded, np.ones(k_size) / k_size, mode="valid"
            ).astype(np.float32)
        return out_xs, out_ys

    def _compute_fusion_centerline(
        self,
        left_xs: np.ndarray,
        left_ys: np.ndarray,
        right_xs: np.ndarray,
        right_ys: np.ndarray,
        offset_px: float,
        y_step: int = 1,
    ) -> Tuple[PointArray, PointArray]:
        """Fuse two boundary polylines into a centerline, ported from
        yolotl_ros2's ``compute_fusion_centerline``: y-wise average where
        both boundaries are present, normal offset where only one is, and a
        moving-average smoothing pass over the seam between the two."""

        if left_xs is None or right_xs is None:
            return None, None

        y_min_l, y_max_l = int(np.min(left_ys)), int(np.max(left_ys))
        lx_dense, ly_dense = self._resample_polyline_by_y(
            left_xs, left_ys, y_min_l, y_max_l, y_step
        )
        y_min_r, y_max_r = int(np.min(right_ys)), int(np.max(right_ys))
        rx_dense, ry_dense = self._resample_polyline_by_y(
            right_xs, right_ys, y_min_r, y_max_r, y_step
        )

        lx_off, ly_off = self._offset_polyline_points(
            left_xs, left_ys, offset_px, direction=1.0
        )
        lx_off_dense, ly_off_dense = None, None
        if lx_off is not None and len(lx_off) > 1:
            lx_off_dense, ly_off_dense = self._resample_polyline_by_y(
                lx_off,
                ly_off,
                int(np.min(ly_off)),
                int(np.max(ly_off)),
                y_step,
            )

        rx_off, ry_off = self._offset_polyline_points(
            right_xs, right_ys, offset_px, direction=-1.0
        )
        rx_off_dense, ry_off_dense = None, None
        if rx_off is not None and len(rx_off) > 1:
            rx_off_dense, ry_off_dense = self._resample_polyline_by_y(
                rx_off,
                ry_off,
                int(np.min(ry_off)),
                int(np.max(ry_off)),
                y_step,
            )

        dict_lx = dict(zip(ly_dense, lx_dense)) if ly_dense is not None else {}
        dict_rx = dict(zip(ry_dense, rx_dense)) if ry_dense is not None else {}
        dict_lx_off = (
            dict(zip(ly_off_dense, lx_off_dense))
            if ly_off_dense is not None
            else {}
        )
        dict_rx_off = (
            dict(zip(ry_off_dense, rx_off_dense))
            if ry_off_dense is not None
            else {}
        )

        all_y = set(dict_lx.keys()) | set(dict_rx.keys())
        if not all_y:
            return None, None
        min_y, max_y = int(min(all_y)), int(max(all_y))

        merged_xs: List[float] = []
        merged_ys: List[float] = []
        for y_val in range(min_y, max_y + 1, y_step):
            y_f = float(y_val)
            has_left = y_f in dict_lx
            has_right = y_f in dict_rx
            center_x: Optional[float] = None
            if has_left and has_right:
                center_x = (dict_lx[y_f] + dict_rx[y_f]) / 2.0
            elif has_left and y_f in dict_lx_off:
                center_x = dict_lx_off[y_f]
            elif has_right and y_f in dict_rx_off:
                center_x = dict_rx_off[y_f]
            if center_x is not None:
                merged_xs.append(center_x)
                merged_ys.append(y_f)

        if len(merged_xs) < 2:
            return None, None

        center_xs = np.array(merged_xs, dtype=np.float32)
        center_ys = np.array(merged_ys, dtype=np.float32)

        k_size = 15
        if len(center_xs) >= k_size:
            padded = np.pad(
                center_xs, (k_size // 2, k_size // 2), mode="edge"
            )
            center_xs = np.convolve(
                padded, np.ones(k_size) / k_size, mode="valid"
            ).astype(np.float32)

        return center_xs, center_ys

    def _build_path_spline(
        self,
        left: Optional[LaneGroup],
        right: Optional[LaneGroup],
    ) -> Tuple[PointArray, str]:
        if left is None and right is None:
            return None, "no_boundary"

        offset_px = (
            0.5 * self.config.lane_width_m * self.config.pixels_per_meter
        )
        left_dense = self._densify_group_points(left)
        right_dense = self._densify_group_points(right)

        if left_dense is not None and right_dense is not None:
            center_xs, center_ys = self._compute_fusion_centerline(
                left_dense[0],
                left_dense[1],
                right_dense[0],
                right_dense[1],
                offset_px,
            )
            reason = "fused_boundaries"
        elif left_dense is not None:
            center_xs, center_ys = self._offset_polyline_points(
                left_dense[0], left_dense[1], offset_px, direction=1.0
            )
            reason = "left_boundary_normal_offset"
        elif right_dense is not None:
            center_xs, center_ys = self._offset_polyline_points(
                right_dense[0], right_dense[1], offset_px, direction=-1.0
            )
            reason = "right_boundary_normal_offset"
        else:
            return None, "insufficient_path"

        if center_xs is None or len(center_xs) < 3:
            return None, "insufficient_path"

        path_array = np.column_stack(
            (center_xs, center_ys)
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
        return path_array, reason

    def _smooth_and_interpolate_path(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """B-spline spatial smoothing, ported from yolotl_ros2's
        ``smooth_and_interpolate_path``. A single quadratic can't follow an
        S-curve/chicane; the spline follows the actual point-window shape
        instead. Falls back to the raw input when there are too few
        distinct points for a cubic fit."""

        from scipy.interpolate import splev, splprep

        if len(xs) < 4:
            return xs, ys

        points = np.column_stack((xs, ys))
        _, unique_indices = np.unique(points, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        xs_u, ys_u = xs[unique_indices], ys[unique_indices]
        if len(xs_u) < 4:
            return xs, ys

        try:
            tck, _ = splprep(
                [xs_u, ys_u],
                s=self.config.path_spline_smooth_factor,
                k=3,
            )
            u_new = np.linspace(0.0, 1.0, self.config.path_spline_points)
            x_new, y_new = splev(u_new, tck)
            return (
                np.asarray(x_new, dtype=np.float32),
                np.asarray(y_new, dtype=np.float32),
            )
        except Exception:
            return xs, ys

    def _smooth_spatial_spline(self, path: PointArray) -> PointArray:
        if path is None or len(path) < 3:
            return None

        spline_xs, spline_ys = self._smooth_and_interpolate_path(
            path[:, 0].astype(np.float32), path[:, 1].astype(np.float32)
        )

        # The spline is arc-length parametrized, not evenly spaced in y,
        # and can be locally non-monotonic in y on a sharp curve -- sort
        # and dedupe before treating it as a y -> x function.
        order = np.argsort(spline_ys)
        sorted_ys = spline_ys[order]
        sorted_xs = spline_xs[order]
        unique_ys, unique_indices = np.unique(
            sorted_ys, return_index=True
        )
        if len(unique_ys) < 2:
            return None
        unique_xs = sorted_xs[unique_indices]

        # Resample onto the same fixed y-grid every frame so
        # `_smooth_temporal`'s frame-to-frame EMA keeps comparing points at
        # matching y positions. Grid points outside the spline's observed
        # range are dropped rather than extrapolated.
        y_values = self._path_y_values()
        valid = (
            (y_values >= unique_ys[0]) & (y_values <= unique_ys[-1])
        )
        if np.count_nonzero(valid) < 3:
            return None
        x_values = np.interp(
            y_values[valid], unique_ys, unique_xs
        ).astype(np.float32)
        if not np.all(np.isfinite(x_values)):
            return None
        margin = self.config.warp_width * 0.15
        x_values = np.clip(
            x_values,
            -margin,
            self.config.warp_width + margin,
        )
        return np.column_stack(
            (x_values, y_values[valid])
        ).astype(np.float32)

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
        offset_x = source_x + direction * offset_px / normal_scale
        offset_y = source_y - direction * offset_px * slope / normal_scale
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

        left_only = ~np.isfinite(path_x) & np.isfinite(left_offset)
        path_x[left_only] = left_offset[left_only]
        right_only = ~np.isfinite(path_x) & np.isfinite(right_offset)
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
            & (path_array[:, 0] < self.config.warp_width + margin)
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
        degree = self.config.path_polynomial_degree
        if degree == 3 and len(np.unique(path[:, 1])) < 4:
            degree = 2
        try:
            coefficients = np.polyfit(
                path[:, 1], path[:, 0], degree
            )
        except (TypeError, ValueError, np.linalg.LinAlgError):
            if degree != 3:
                return path
            try:
                coefficients = np.polyfit(
                    path[:, 1], path[:, 0], 2
                )
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

    def _anchor_path_to_vehicle_center(
        self, path: PointArray
    ) -> PointArray:
        """Pin the nearest path end to the vehicle center without a kink."""
        if path is None or len(path) == 0:
            return None

        anchored = np.asarray(path, dtype=np.float32).copy()
        nearest_index = int(np.argmax(anchored[:, 1]))
        farthest_y = float(np.min(anchored[:, 1]))
        nearest_y = float(anchored[nearest_index, 1])
        vehicle_x = float(self.config.warp_width) * 0.5
        vehicle_y = float(self.config.warp_height - 1)
        correction = vehicle_x - float(anchored[nearest_index, 0])

        y_span = max(nearest_y - farthest_y, 1.0)
        blend = np.clip(
            (anchored[:, 1] - farthest_y) / y_span, 0.0, 1.0
        )
        blend = blend * blend * (3.0 - 2.0 * blend)
        anchored[:, 0] += correction * blend.astype(np.float32)
        anchored[nearest_index, 0] = vehicle_x

        if nearest_y < vehicle_y:
            vehicle_point = np.asarray(
                [[vehicle_x, vehicle_y]], dtype=np.float32
            )
            anchored = np.insert(
                anchored, nearest_index, vehicle_point, axis=0
            )
        return anchored

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
                lane_type, type_confidence, type_source = (
                    self._group_type(group)
                )
                confidence_text = (
                    f" {type_confidence * 100.0:.0f}%"
                    if type_confidence is not None
                    else ""
                )
                line_type = (
                    f"{lane_type} x{piece_count}{confidence_text} "
                    f"({type_source})"
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

    def plan_mask(
        self,
        mask: np.ndarray,
        instances: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> PathPlanResult:
        """Plan only; detection and vehicle control belong to other nodes.

        Local connected-component extraction remains authoritative for
        geometry. YOLO instance boxes optionally attach dashed/solid class
        confidence to each overlapping component.
        """

        pieces, _ = self._extract_mask_pieces(mask, instances)
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
        spatial_path = self._anchor_path_to_vehicle_center(
            self._smooth_spatial(raw_path)
        )
        final_path, fallback = self._smooth_temporal(
            spatial_path,
            f"{selection_mode}:{reason}",
        )
        fallback = fallback or self._using_tracked_boundary
        detected_lines = []
        for group in groups:
            points = np.asarray(group["points"], dtype=np.float32)
            if len(points) == 0:
                continue
            label_point = points[len(points) // 2]
            piece_count = len(group["pieces"])
            lane_type, type_confidence, type_source = (
                self._group_type(group)
            )
            box_padding = 15.0
            detected_lines.append(
                {
                    "type": lane_type,
                    "type_confidence": (
                        None
                        if type_confidence is None
                        else round(type_confidence, 4)
                    ),
                    "type_source": type_source,
                    "x": round(float(label_point[0]), 1),
                    "y": round(float(label_point[1]), 1),
                    "x_min": round(
                        float(np.min(points[:, 0]) - box_padding),
                        1,
                    ),
                    "y_min": round(
                        float(np.min(points[:, 1]) - box_padding),
                        1,
                    ),
                    "x_max": round(
                        float(np.max(points[:, 0]) + box_padding),
                        1,
                    ),
                    "y_max": round(
                        float(np.max(points[:, 1]) + box_padding),
                        1,
                    ),
                    "piece_count": piece_count,
                }
            )
        return PathPlanResult(
            path_pixels=final_path,
            path_meters=self.pixels_to_meters(final_path),
            reason=reason,
            used_fallback=fallback,
            group_count=len(groups),
            dashed_region_count=dashed_region_count,
            selection_mode=selection_mode,
            which_lane=which_lane,
            detected_lines=detected_lines,
            left_boundary_pixels=(
                None
                if left is None
                else np.asarray(left["points"], dtype=np.float32)
            ),
            right_boundary_pixels=(
                None
                if right is None
                else np.asarray(right["points"], dtype=np.float32)
            ),
        )
