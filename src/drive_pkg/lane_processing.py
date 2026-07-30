#!/usr/bin/env python3
"""BEV lane-mask path planner core, shared by path_plan.py and drive_main.py.

The best.pt model was trained on 640x640 BEV images with ``dashed``/``solid``
lane classes. Incoming 640x480 usb_cam frames are therefore warped to that
training view before inference. Per-detection segmentation components are
extracted, tracked frame to frame as left/right boundaries, and fused into a
smoothed metric path in ``base_link`` coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from sensor_msgs.msg import CompressedImage


def _default_bev_params_path() -> Path:
    source_path = (
        Path(__file__).resolve().parent / "resource" / "bev(0729).npz"
    )
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("drive_pkg"))
            / "resource"
            / "bev(0729).npz"
        )
    except (ImportError, LookupError):
        return source_path


def _default_model_path() -> Path:
    """Find the shared lane-segmentation checkpoint in source or installed
    layouts."""
    source_path = (
        Path(__file__).resolve().parent / "resource" / "best.pt"
    )
    if source_path.is_file():
        return source_path

    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory("drive_pkg"))
            / "resource"
            / "best.pt"
        )
    except (ImportError, LookupError):
        return source_path


DEFAULT_MODEL = _default_model_path()
DEFAULT_BEV_PARAMS = _default_bev_params_path()

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
    # Morphological close kernel size used when isolating one detection's
    # largest mask component within its own bounding box.
    bbox_close_ksize: int = 5
    # B-spline smoothing factor and output point count for spatial smoothing
    # (see scipy.interpolate.splprep's ``s`` parameter).
    path_spline_smooth_factor: float = 10.0
    path_spline_points: int = 100
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
        if cfg.bbox_close_ksize < 1:
            raise ValueError("bbox_close_ksize must be at least 1")
        if cfg.path_spline_smooth_factor < 0.0:
            raise ValueError(
                "path_spline_smooth_factor cannot be negative"
            )
        if cfg.path_spline_points < 4:
            raise ValueError("path_spline_points must be at least 4")

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
        (noise)."""

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
        instances: Sequence[Dict[str, Any]],
    ) -> List[LaneGroup]:
        """Extract one piece per YOLO detection, using that detection's own
        mask (already isolated per detection, not shared with any other
        detection)."""

        pieces: List[LaneGroup] = []
        for instance in instances:
            detection_mask = instance["mask"]
            reference_y = detection_mask.shape[0] * 0.88
            bbox = (
                int(instance["x_min"]),
                int(instance["y_min"]),
                int(instance["x_max"]),
                int(instance["y_max"]),
            )
            component = self._extract_largest_component_in_bbox(
                detection_mask,
                bbox,
                self.config.min_component_area,
                self.config.bbox_close_ksize,
            )
            if component is None:
                continue
            area = int(np.count_nonzero(component))
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
        return pieces

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
        """Frame-to-frame left/right tracking, ported from yolotl_ros2
        main5.py's ``process_image`` step 5 (``tracked_lanes``).

        Candidates are sorted by ``x_ref`` (dolbatS's stand-in for main5's
        ``x_bottom``). With exactly two candidates, they're assigned
        left/right directly by that order regardless of tracking state --
        no distance threshold, unlike the curve-matching this replaces.
        With exactly one candidate, it's assigned to whichever side's last
        position it's closer to (or by BEV-center split on a cold start).
        Faithfully keeps main5's own limitation: 0 or 3+ candidates in a
        frame are ignored entirely and existing tracks just age.
        """

        candidates = sorted(
            (
                group
                for group in groups
                if self._group_has_path_overlap(group)
            ),
            key=lambda group: float(group["x_ref"]),
        )

        left_track = self._tracked_lanes["left"]
        right_track = self._tracked_lanes["right"]
        current_left: Optional[LaneGroup] = None
        current_right: Optional[LaneGroup] = None

        if len(candidates) == 2:
            current_left, current_right = candidates[0], candidates[1]
        elif len(candidates) == 1:
            detected = candidates[0]
            left_previous = left_track["group"]
            right_previous = right_track["group"]
            distance_to_left = (
                abs(
                    float(detected["x_ref"])
                    - float(left_previous["x_ref"])
                )
                if left_previous is not None
                else float("inf")
            )
            distance_to_right = (
                abs(
                    float(detected["x_ref"])
                    - float(right_previous["x_ref"])
                )
                if right_previous is not None
                else float("inf")
            )
            if (
                distance_to_left < distance_to_right
                and left_previous is not None
            ):
                current_left = detected
            elif (
                distance_to_right < distance_to_left
                and right_previous is not None
            ):
                current_right = detected
            else:
                center_x = self.config.warp_width * 0.5
                if float(detected["x_ref"]) < center_x:
                    current_left = detected
                else:
                    current_right = detected

        for side, current in (
            ("left", current_left),
            ("right", current_right),
        ):
            track = self._tracked_lanes[side]
            if current is not None:
                track["group"] = current
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

    def _build_path(
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

    def _smooth_spatial(self, path: PointArray) -> PointArray:
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

    def plan_mask(
        self,
        instances: Sequence[Dict[str, Any]],
    ) -> PathPlanResult:
        """Plan only; detection and vehicle control belong to other nodes.

        ``instances`` is the per-detection list from
        ``DetectionOutput.instances`` (lane_detect.py), each carrying its
        own already-isolated mask. One piece is extracted per detection
        (``_extract_bbox_pieces``).
        """

        pieces = self._extract_bbox_pieces(instances)
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
        detected_lines = []
        for group in groups:
            points = np.asarray(group["points"], dtype=np.float32)
            if len(points) == 0:
                continue
            label_point = points[len(points) // 2]
            piece_count = len(group["pieces"])
            box_padding = 15.0
            detected_lines.append(
                {
                    "type": (
                        "DASHED"
                        if piece_count
                        >= self.config.dashed_piece_threshold
                        else "SOLID"
                    ),
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
