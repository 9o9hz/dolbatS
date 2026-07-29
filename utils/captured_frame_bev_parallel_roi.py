#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""캡처 프레임에서 평행 가이드를 이용해 BEV ROI를 고르는 간단한 도구."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = PROJECT_ROOT / "camera_calibration.npz"
DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "rosbag2_2026_07_23-14_55_06_captured_frames"
    / "frame_000639_t000038969ms_ts1784786145048977827.jpg"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "bev(0729).npz"

SOURCE_WINDOW = "Captured frame - parallel BEV ROI"
BEV_WINDOW = "Temporary BEV - marker distortion"

LB, RB, LT, RT = 0, 1, 2, 3
POINT_NAMES = ("LB", "RB", "LT", "RT")
MIN_EDGE_PX = 25.0
MIN_AREA_PX2 = 1000.0
MIN_OUTPUT_SIDE = 64
MAX_OUTPUT_SIDE = 1400
SOURCE_EXTENSION_PX = 100
# 원본 밖 좌표 선택 여백과 BEV 결과물 여백은 별개다.
# 좌우와 위쪽은 확장하되, 원본 정보가 없는 아래쪽은 강제 확장하지 않는다.
BEV_PADDING_LEFT_PX = 100
BEV_PADDING_RIGHT_PX = 100
BEV_PADDING_TOP_PX = 100
BEV_PADDING_BOTTOM_PX = 0
AUTO_CROP_EMPTY_BOTTOM = True

# 검출하는 흰 선 상자의 실제 내부 치수는 560x490mm이다.
MARKER_INNER_WIDTH_MM = 560.0
MARKER_INNER_HEIGHT_MM = 490.0
EXPECTED_CELL_RATIO = MARKER_INNER_WIDTH_MM / MARKER_INNER_HEIGHT_MM


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    new_camera_matrix: np.ndarray
    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray


@dataclass(frozen=True)
class MarkerDetection:
    points: np.ndarray
    threshold_spread_px: float
    minimum_fit_points: int


@dataclass(frozen=True)
class DistortionMetrics:
    raw_cell_ratios: tuple[float, ...]
    raw_mean_square_error_percent: float
    cell_ratios: tuple[float, ...]
    mean_square_error_percent: float
    max_square_error_percent: float
    cell_size_cv_percent: float
    horizontal_spacing_cv_percent: float
    vertical_spacing_cv_percent: float
    max_orthogonality_error_deg: float
    applied_x_scale: float
    marker_inside_roi: bool


@dataclass(frozen=True)
class BevResult:
    src_points: np.ndarray
    dst_points: np.ndarray
    homography: np.ndarray
    width: int
    height: int
    pure_bev: np.ndarray
    marker_points_bev: np.ndarray
    metrics: DistortionMetrics


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "캡처한 ROS bag 프레임에서 평행 가이드로 BEV ROI를 선택하고 "
            "560x490mm 흰 선 기준 상자의 비율과 직각도를 검사합니다."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=(
            "입력 캡처 이미지. 기본값은 14:55:06 bag의 "
            "39초(실제 38.968722초) 프레임"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="기존 약 688px 초점거리의 640x480 calibration NPZ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="S 키로 저장할 BEV NPZ",
    )
    arguments = parser.parse_args(argv)
    for name in ("image", "calibration", "output"):
        path = Path(getattr(arguments, name)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        setattr(arguments, name, path.resolve())
    if arguments.output.suffix.lower() != ".npz":
        parser.error("--output 확장자는 .npz여야 합니다.")
    return arguments


def scalar_int(data: object, key: str) -> int:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key}는 스칼라여야 합니다.")
    result = int(value.reshape(-1)[0])
    return result


def load_calibration(path: Path) -> Calibration:
    if not path.is_file():
        raise FileNotFoundError(f"캘리브레이션 파일이 없습니다: {path}")
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
            raise KeyError(f"캘리브레이션 키 누락: {sorted(missing)}")
        camera_matrix = np.asarray(
            data["camera_matrix"],
            dtype=np.float64,
        )
        distortion = np.asarray(
            data["distortion_coefficients"],
            dtype=np.float64,
        ).reshape(-1)
        new_camera_matrix = np.asarray(
            data["new_camera_matrix"],
            dtype=np.float64,
        )
        width = scalar_int(data, "image_width")
        height = scalar_int(data, "image_height")
    if (
        camera_matrix.shape != (3, 3)
        or new_camera_matrix.shape != (3, 3)
        or distortion.size not in (4, 5, 8, 12, 14)
        or not np.all(np.isfinite(camera_matrix))
        or not np.all(np.isfinite(new_camera_matrix))
        or not np.all(np.isfinite(distortion))
    ):
        raise ValueError("카메라 캘리브레이션 행렬이 유효하지 않습니다.")
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return Calibration(
        camera_matrix=camera_matrix,
        distortion=distortion,
        new_camera_matrix=new_camera_matrix,
        width=width,
        height=height,
        map_x=map_x,
        map_y=map_y,
    )


def normalized_line(
    gray: np.ndarray,
    seed_start: tuple[float, float],
    seed_end: tuple[float, float],
    *,
    limit_axis: str,
    limit_low: float,
    limit_high: float,
    threshold: int,
    band_px: float = 10.0,
) -> tuple[np.ndarray, int]:
    yy, xx = np.where(gray >= threshold)
    x1, y1 = seed_start
    x2, y2 = seed_end
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = math.hypot(a, b)
    if norm <= 1.0e-9:
        raise ValueError("기준 표식 seed 선의 길이가 0입니다.")
    a, b, c = a / norm, b / norm, c / norm
    keep = np.abs(a * xx + b * yy + c) <= band_px
    if limit_axis == "x":
        keep &= (xx >= limit_low) & (xx <= limit_high)
    else:
        keep &= (yy >= limit_low) & (yy <= limit_high)
    points = np.column_stack((xx[keep], yy[keep])).astype(np.float32)
    if len(points) < 80:
        raise ValueError(
            f"기준 표식 선의 밝은 점이 부족합니다: {len(points)}"
        )
    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    line = np.asarray(
        [-float(vy), float(vx), float(vy * x0 - vx * y0)],
        dtype=np.float64,
    )
    line_norm = math.hypot(float(line[0]), float(line[1]))
    if line_norm <= 1.0e-9 or not np.all(np.isfinite(line)):
        raise ValueError("기준 표식 선 fitting 결과가 유효하지 않습니다.")
    return line / line_norm, len(points)


def line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    homogeneous = np.cross(first, second)
    if abs(float(homogeneous[2])) <= 1.0e-9:
        raise ValueError("기준 표식의 두 선이 평행합니다.")
    point = homogeneous[:2] / homogeneous[2]
    if not np.all(np.isfinite(point)):
        raise ValueError("기준 표식 교점이 유효하지 않습니다.")
    return point


def detect_reference_marker(
    undistorted: np.ndarray,
) -> MarkerDetection:
    """현재 고정 카메라에서 완전히 보이는 흰 선 상자 한 칸을 검출한다."""
    height, width = undistorted.shape[:2]
    sx = width / 640.0
    sy = height / 480.0
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    def scaled(point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * sx, point[1] * sy

    detections: list[np.ndarray] = []
    minimum_points = 1_000_000
    for threshold in (80, 100, 120):
        top, top_count = normalized_line(
            gray,
            scaled((249.0, 253.0)),
            scaled((440.0, 257.0)),
            limit_axis="x",
            limit_low=265.0 * sx,
            limit_high=425.0 * sx,
            threshold=threshold,
            band_px=6.0 * max(sx, sy),
        )
        bottom, bottom_count = normalized_line(
            gray,
            scaled((216.0, 320.0)),
            scaled((469.0, 326.0)),
            limit_axis="x",
            limit_low=235.0 * sx,
            limit_high=450.0 * sx,
            threshold=threshold,
            band_px=6.0 * max(sx, sy),
        )
        left, left_count = normalized_line(
            gray,
            scaled((248.0, 253.0)),
            scaled((215.0, 321.0)),
            limit_axis="y",
            limit_low=266.0 * sy,
            limit_high=307.0 * sy,
            threshold=threshold,
            band_px=6.0 * max(sx, sy),
        )
        right, right_count = normalized_line(
            gray,
            scaled((440.0, 257.0)),
            scaled((469.0, 326.0)),
            limit_axis="y",
            limit_low=270.0 * sy,
            limit_high=312.0 * sy,
            threshold=threshold,
            band_px=6.0 * max(sx, sy),
        )
        minimum_points = min(
            minimum_points,
            top_count,
            bottom_count,
            left_count,
            right_count,
        )
        detections.append(np.asarray([
            [
                line_intersection(top, left),
                line_intersection(top, right),
            ],
            [
                line_intersection(bottom, left),
                line_intersection(bottom, right),
            ],
        ], dtype=np.float64))

    stacked = np.stack(detections)
    mean_points = np.mean(stacked, axis=0)
    spread = float(np.max(np.linalg.norm(
        stacked - mean_points[None, :, :, :],
        axis=3,
    )))
    if spread > 2.0 * max(sx, sy):
        raise ValueError(
            f"흰 선 상자 반복 검출 흔들림이 큽니다: {spread:.2f}px"
        )
    flat_points = mean_points.reshape(-1, 2)
    if (
        np.any(flat_points[:, 0] < 0.0)
        or np.any(flat_points[:, 0] > width - 1)
        or np.any(flat_points[:, 1] < 0.0)
        or np.any(flat_points[:, 1] > height - 1)
    ):
        raise ValueError("흰 선 상자 검출점이 영상 밖입니다.")
    polygon = np.float32([
        mean_points[0, 0],
        mean_points[0, 1],
        mean_points[1, 1],
        mean_points[1, 0],
    ])
    if not cv2.isContourConvex(polygon):
        raise ValueError("흰 선 상자 검출 사각형이 볼록하지 않습니다.")
    return MarkerDetection(
        points=np.asarray(mean_points, dtype=np.float32),
        threshold_spread_px=spread,
        minimum_fit_points=minimum_points,
    )


def transform_points(
    homography: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    result = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(homography, dtype=np.float64),
    ).reshape(-1, 2)
    if not np.all(np.isfinite(result)):
        raise ValueError("perspective 변환 결과가 유효하지 않습니다.")
    return result


def parallel_angle_error(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-9:
        raise ValueError("길이가 0인 표식 변이 있습니다.")
    cosine = float(np.clip(
        abs(np.dot(first, second)) / denominator,
        -1.0,
        1.0,
    ))
    return float(np.degrees(np.arccos(cosine)))


def corner_angle_error(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1.0e-9:
        raise ValueError("길이가 0인 표식 변이 있습니다.")
    cosine = float(np.clip(
        np.dot(first, second) / denominator,
        -1.0,
        1.0,
    ))
    angle = float(np.degrees(np.arccos(cosine)))
    return abs(90.0 - angle)


def grid_cell_measurements(
    grid_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return cell ratios, sizes, horizontal/vertical edge lengths and angles."""
    grid = np.asarray(grid_points, dtype=np.float64)
    if grid.ndim != 3 or grid.shape[2] != 2 or min(grid.shape[:2]) < 2:
        raise ValueError("격자점 배열은 (행, 열, 2) 형태여야 합니다.")
    horizontal_vectors = grid[:, 1:] - grid[:, :-1]
    vertical_vectors = grid[1:] - grid[:-1]
    horizontal = np.linalg.norm(horizontal_vectors, axis=2)
    vertical = np.linalg.norm(vertical_vectors, axis=2)
    cell_width = (horizontal[:-1] + horizontal[1:]) * 0.5
    cell_height = (vertical[:, :-1] + vertical[:, 1:]) * 0.5
    ratios = cell_width / np.maximum(cell_height, 1.0e-9)
    sizes = np.sqrt(np.maximum(cell_width * cell_height, 0.0))
    angles = []
    for row in range(grid.shape[0] - 1):
        for column in range(grid.shape[1] - 1):
            top = grid[row, column + 1] - grid[row, column]
            bottom = grid[row + 1, column + 1] - grid[row + 1, column]
            left = grid[row + 1, column] - grid[row, column]
            right = grid[row + 1, column + 1] - grid[row, column + 1]
            angles.extend(
                [
                    corner_angle_error(top, left),
                    corner_angle_error(top, right),
                    corner_angle_error(bottom, left),
                    corner_angle_error(bottom, right),
                ]
            )
    return ratios, sizes, horizontal, vertical, np.asarray(angles)


def variation_percent(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return (
        float(np.std(values)) / mean * 100.0
        if mean > 1.0e-9
        else float("inf")
    )


def atomic_write_npz(path: Path, arrays: dict[str, object]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    backup_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o644)
        if path.exists():
            backup_path = path.with_name(
                f".{path.stem}.previous{path.suffix}"
            )
            shutil.copy2(path, backup_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return backup_path


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"이미지 저장 실패: {path}")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class ParallelRoiTool:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.calibration = load_calibration(arguments.calibration)
        raw = cv2.imread(str(arguments.image), cv2.IMREAD_COLOR)
        if raw is None or raw.size == 0:
            raise ValueError(f"입력 이미지를 읽지 못했습니다: {arguments.image}")
        if (
            raw.shape[1] != self.calibration.width
            or raw.shape[0] != self.calibration.height
        ):
            raise ValueError(
                "입력 이미지와 캘리브레이션 해상도가 다릅니다: "
                f"{raw.shape[1]}x{raw.shape[0]} != "
                f"{self.calibration.width}x{self.calibration.height}"
            )
        self.raw = raw
        self.frame = cv2.remap(
            raw,
            self.calibration.map_x,
            self.calibration.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.display_frame = cv2.copyMakeBorder(
            self.frame,
            SOURCE_EXTENSION_PX,
            SOURCE_EXTENSION_PX,
            SOURCE_EXTENSION_PX,
            SOURCE_EXTENSION_PX,
            cv2.BORDER_CONSTANT,
            value=(18, 18, 18),
        )
        self.marker = detect_reference_marker(self.frame)
        self.output_path: Path = arguments.output
        self.points: list[np.ndarray] = []
        self.cursor = np.asarray([0.0, 0.0], dtype=np.float64)
        self.result: BevResult | None = None
        self.status_message = "Click LEFT-BOTTOM"
        self.dirty = True
        self.bev_window_open = False

        cv2.namedWindow(SOURCE_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(SOURCE_WINDOW, self.mouse_callback)

    def bottom_unit(self) -> np.ndarray:
        if len(self.points) < 2:
            raise ValueError("밑변 두 점이 아직 없습니다.")
        vector = self.points[RB] - self.points[LB]
        length = float(np.linalg.norm(vector))
        if length < MIN_EDGE_PX:
            raise ValueError("밑변이 너무 짧습니다.")
        return vector / length

    def upward_normal(self) -> np.ndarray:
        unit = self.bottom_unit()
        normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
        if normal[1] > 0.0:
            normal *= -1.0
        return normal

    def snapped_right_top(self, point: np.ndarray) -> np.ndarray:
        if len(self.points) < 3:
            raise ValueError("좌상단 점이 아직 없습니다.")
        unit = self.bottom_unit()
        distance = float(np.dot(point - self.points[LT], unit))
        return self.points[LT] + max(distance, MIN_EDGE_PX) * unit

    def validate_completed_points(
        self,
        points: np.ndarray,
    ) -> None:
        width = self.calibration.width
        height = self.calibration.height
        if (
            np.any(points[:, 0] < -SOURCE_EXTENSION_PX)
            or np.any(
                points[:, 0] > width - 1 + SOURCE_EXTENSION_PX
            )
            or np.any(points[:, 1] < -SOURCE_EXTENSION_PX)
            or np.any(
                points[:, 1] > height - 1 + SOURCE_EXTENSION_PX
            )
        ):
            raise ValueError(
                f"스냅된 ROI 점이 {SOURCE_EXTENSION_PX}px 확장 범위 밖입니다."
            )
        for first, second in (
            (LB, RB),
            (LT, RT),
            (LB, LT),
            (RB, RT),
        ):
            if np.linalg.norm(points[second] - points[first]) < MIN_EDGE_PX:
                raise ValueError("ROI 변 길이가 너무 짧습니다.")
        polygon = np.float32([
            points[LT],
            points[RT],
            points[RB],
            points[LB],
        ])
        if not cv2.isContourConvex(polygon):
            raise ValueError("ROI 네 점이 교차하거나 오목합니다.")
        area = float(abs(cv2.contourArea(polygon)))
        if area < MIN_AREA_PX2:
            raise ValueError(f"ROI 면적이 너무 작습니다: {area:.1f}px²")

    def marker_inside_roi(self, src_points: np.ndarray) -> bool:
        polygon = np.float32([
            src_points[LT],
            src_points[RT],
            src_points[RB],
            src_points[LB],
        ])
        return all(
            cv2.pointPolygonTest(
                polygon,
                tuple(map(float, point)),
                False,
            ) >= 0.0
            for point in self.marker.points.reshape(-1, 2)
        )

    def compute_result(self) -> BevResult:
        src = np.asarray(self.points, dtype=np.float64)
        self.validate_completed_points(src)
        bottom_width = float(np.linalg.norm(src[RB] - src[LB]))
        top_width = float(np.linalg.norm(src[RT] - src[LT]))
        left_height = float(np.linalg.norm(src[LT] - src[LB]))
        right_height = float(np.linalg.norm(src[RT] - src[RB]))
        output_width = max(64, int(math.ceil(
            max(bottom_width, top_width)
        )) + 1)
        output_height = max(64, int(math.ceil(
            max(left_height, right_height)
        )) + 1)
        scale = min(
            1.0,
            MAX_OUTPUT_SIDE / output_width,
            MAX_OUTPUT_SIDE / output_height,
        )
        output_width = max(64, int(round(output_width * scale)))
        output_height = max(64, int(round(output_height * scale)))
        base_dst = np.float32([
            [0.0, output_height - 1.0],
            [output_width - 1.0, output_height - 1.0],
            [0.0, 0.0],
            [output_width - 1.0, 0.0],
        ])
        base_homography = cv2.getPerspectiveTransform(
            np.asarray(src, dtype=np.float32),
            base_dst,
        )
        if (
            base_homography.shape != (3, 3)
            or not np.all(np.isfinite(base_homography))
            or np.linalg.matrix_rank(base_homography) < 3
            or np.linalg.cond(base_homography) > 1.0e10
        ):
            raise ValueError("BEV homography가 유효하지 않습니다.")

        grid_shape = self.marker.points.shape
        raw_marker_bev = transform_points(
            base_homography,
            self.marker.points,
        ).reshape(grid_shape)

        def mean_cell_ratio_at_x_scale(x_scale: float) -> float:
            scaled = raw_marker_bev.copy()
            scaled[:, :, 0] *= x_scale
            ratios, _, _, _, _ = grid_cell_measurements(scaled)
            return float(np.mean(ratios))

        raw_cell_ratios, _, _, _, _ = grid_cell_measurements(raw_marker_bev)
        low_scale = 1.0e-3
        high_scale = 1.0e3
        low_ratio = mean_cell_ratio_at_x_scale(low_scale)
        high_ratio = mean_cell_ratio_at_x_scale(high_scale)
        if not (
            low_ratio <= EXPECTED_CELL_RATIO <= high_ratio
            or high_ratio <= EXPECTED_CELL_RATIO <= low_ratio
        ):
            raise ValueError(
                "이 ROI에서는 기준 상자의 목표 비율을 맞출 수 없습니다. "
                "네 번째 점을 다시 찍으세요."
            )
        increasing = high_ratio > low_ratio
        for _ in range(70):
            middle_scale = (low_scale + high_scale) / 2.0
            middle_ratio = mean_cell_ratio_at_x_scale(middle_scale)
            if (
                middle_ratio < EXPECTED_CELL_RATIO
            ) == increasing:
                low_scale = middle_scale
            else:
                high_scale = middle_scale
        applied_x_scale = (low_scale + high_scale) / 2.0

        corrected_content_width = (
            (output_width - 1.0) * applied_x_scale
        )
        corrected_content_height = output_height - 1.0
        uniform_scale = min(
            1.0,
            (MAX_OUTPUT_SIDE - 1.0) / corrected_content_width,
            (MAX_OUTPUT_SIDE - 1.0) / corrected_content_height,
        )
        corrected_content_width *= uniform_scale
        corrected_content_height *= uniform_scale
        corrected_content_canvas_width = max(
            2,
            int(math.ceil(corrected_content_width)) + 1,
        )
        corrected_content_canvas_height = max(
            2,
            int(math.ceil(corrected_content_height)) + 1,
        )
        if (
            corrected_content_canvas_width < MIN_OUTPUT_SIDE
            or corrected_content_canvas_height < MIN_OUTPUT_SIDE
        ):
            raise ValueError(
                "BEV 출력이 지나치게 가늘어집니다. "
                "네 번째 점을 다시 찍으세요."
            )
        correction = np.asarray([
            [applied_x_scale * uniform_scale, 0.0, 0.0],
            [0.0, uniform_scale, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        extension_translation = np.asarray([
            [1.0, 0.0, float(BEV_PADDING_LEFT_PX)],
            [0.0, 1.0, float(BEV_PADDING_TOP_PX)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        homography = extension_translation @ correction @ base_homography
        corrected_width = (
            corrected_content_canvas_width
            + BEV_PADDING_LEFT_PX
            + BEV_PADDING_RIGHT_PX
        )
        corrected_height = (
            corrected_content_canvas_height
            + BEV_PADDING_TOP_PX
            + BEV_PADDING_BOTTOM_PX
        )
        if (
            not np.all(np.isfinite(homography))
            or np.linalg.matrix_rank(homography) < 3
            or np.linalg.cond(homography) > 1.0e10
        ):
            raise ValueError("비율 보정 후 BEV homography가 불안정합니다.")
        dst = np.asarray(
            transform_points(homography, src),
            dtype=np.float32,
        )
        if (
            np.any(dst[:, 0] < -1.0e-4)
            or np.any(dst[:, 0] > corrected_width - 1 + 1.0e-4)
            or np.any(dst[:, 1] < -1.0e-4)
            or np.any(dst[:, 1] > corrected_height - 1 + 1.0e-4)
        ):
            raise ValueError("비율 보정 destination이 출력 영상 밖입니다.")
        bev = cv2.warpPerspective(
            self.frame,
            homography,
            (corrected_width, corrected_height),
            flags=cv2.INTER_LINEAR,
        )
        if AUTO_CROP_EMPTY_BOTTOM:
            source_valid = np.full(
                self.frame.shape[:2], 255, dtype=np.uint8
            )
            valid_mask = cv2.warpPerspective(
                source_valid,
                homography,
                (corrected_width, corrected_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            valid_rows = np.flatnonzero(np.any(valid_mask > 0, axis=1))
            if valid_rows.size:
                valid_height = int(valid_rows[-1]) + 1
                if valid_height < corrected_height:
                    bev = bev[:valid_height].copy()
                    corrected_height = valid_height
        marker_bev = transform_points(
            homography, self.marker.points
        ).reshape(grid_shape)
        cell_ratios, cell_sizes, horizontal, vertical, angle_errors = (
            grid_cell_measurements(marker_bev)
        )
        raw_errors = np.abs(
            raw_cell_ratios / EXPECTED_CELL_RATIO - 1.0
        ) * 100.0
        square_errors = np.abs(
            cell_ratios / EXPECTED_CELL_RATIO - 1.0
        ) * 100.0
        metrics = DistortionMetrics(
            raw_cell_ratios=tuple(map(float, raw_cell_ratios.flat)),
            raw_mean_square_error_percent=float(np.mean(raw_errors)),
            cell_ratios=tuple(map(float, cell_ratios.flat)),
            mean_square_error_percent=float(np.mean(square_errors)),
            max_square_error_percent=float(np.max(square_errors)),
            cell_size_cv_percent=variation_percent(cell_sizes),
            horizontal_spacing_cv_percent=variation_percent(horizontal),
            vertical_spacing_cv_percent=variation_percent(vertical),
            max_orthogonality_error_deg=float(np.max(angle_errors)),
            applied_x_scale=applied_x_scale,
            marker_inside_roi=self.marker_inside_roi(src),
        )
        return BevResult(
            src_points=np.asarray(src, dtype=np.float32),
            dst_points=dst,
            homography=np.asarray(homography, dtype=np.float64),
            width=corrected_width,
            height=corrected_height,
            pure_bev=bev,
            marker_points_bev=np.asarray(marker_bev, dtype=np.float32),
            metrics=metrics,
        )

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        del flags, param
        point = np.asarray([
            float(np.clip(
                x,
                0,
                self.calibration.width + 2 * SOURCE_EXTENSION_PX - 1,
            ) - SOURCE_EXTENSION_PX),
            float(np.clip(
                y,
                0,
                self.calibration.height + 2 * SOURCE_EXTENSION_PX - 1,
            ) - SOURCE_EXTENSION_PX),
        ])
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = point
            if len(self.points) == 3:
                self.dirty = True
            return
        if event != cv2.EVENT_LBUTTONDOWN or len(self.points) >= 4:
            return
        appended_right_top = False
        try:
            if len(self.points) == 0:
                self.points.append(point)
                self.status_message = "Click RIGHT-BOTTOM"
            elif len(self.points) == 1:
                if np.linalg.norm(point - self.points[LB]) < MIN_EDGE_PX:
                    raise ValueError("밑변이 너무 짧습니다.")
                self.points.append(point)
                self.status_message = "Click LEFT-TOP"
            elif len(self.points) == 2:
                height_on_normal = float(np.dot(
                    point - self.points[LB],
                    self.upward_normal(),
                ))
                if height_on_normal < MIN_EDGE_PX:
                    raise ValueError("좌상단은 밑변보다 충분히 위에 찍으세요.")
                self.points.append(point)
                self.status_message = (
                    "Click RIGHT-TOP on cyan parallel ray"
                )
            else:
                snapped = self.snapped_right_top(point)
                candidate = np.asarray(
                    [*self.points, snapped],
                    dtype=np.float64,
                )
                self.validate_completed_points(candidate)
                self.points.append(snapped)
                appended_right_top = True
                self.result = self.compute_result()
                self.status_message = (
                    "BEV ready | R reset | S save"
                )
                self.print_metrics()
        except (ValueError, cv2.error) as error:
            if appended_right_top and len(self.points) == 4:
                self.points.pop()
            self.status_message = f"INVALID: {error}"
            print(f"[점 선택 거부] {error}")
        self.dirty = True

    def reset(self) -> None:
        self.points.clear()
        self.result = None
        self.status_message = "Click LEFT-BOTTOM"
        if self.bev_window_open:
            try:
                cv2.destroyWindow(BEV_WINDOW)
            except cv2.error:
                pass
            self.bev_window_open = False
        self.dirty = True

    def draw_source(self) -> np.ndarray:
        display = self.display_frame.copy()
        shift = np.asarray(
            [SOURCE_EXTENSION_PX, SOURCE_EXTENSION_PX],
            dtype=np.float64,
        )
        cv2.rectangle(
            display,
            (SOURCE_EXTENSION_PX, SOURCE_EXTENSION_PX),
            (
                SOURCE_EXTENSION_PX + self.calibration.width - 1,
                SOURCE_EXTENSION_PX + self.calibration.height - 1,
            ),
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
        marker = np.asarray(
            self.marker.points + shift, dtype=np.float32
        )
        marker_polygon = np.rint(np.asarray([
            marker[0, 0],
            marker[0, 1],
            marker[1, 1],
            marker[1, 0],
        ])).astype(np.int32)
        cv2.polylines(
            display, [marker_polygon], True, (255, 255, 0), 2, cv2.LINE_AA
        )
        for marker_point in marker_polygon:
            cv2.circle(
                display, tuple(marker_point), 4, (0, 255, 255), -1, cv2.LINE_AA
            )
        label_point = tuple(np.rint(marker[0, 0]).astype(int))
        cv2.putText(
            display,
            "detected white-line box",
            (label_point[0] + 5, label_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        for index, point in enumerate(self.points):
            center = tuple(np.rint(point + shift).astype(int))
            cv2.circle(display, center, 6, (0, 255, 255), -1)
            cv2.putText(
                display,
                POINT_NAMES[index],
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if len(self.points) >= 2:
            cv2.line(
                display,
                tuple(np.rint(self.points[LB] + shift).astype(int)),
                tuple(np.rint(self.points[RB] + shift).astype(int)),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(self.points) >= 3:
            unit = self.bottom_unit()
            ray_end = self.points[LT] + unit * 2000.0
            cv2.line(
                display,
                tuple(np.rint(self.points[LT] + shift).astype(int)),
                tuple(np.rint(ray_end + shift).astype(int)),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if len(self.points) == 3:
                snapped = self.snapped_right_top(self.cursor)
                cv2.circle(
                    display,
                    tuple(np.rint(snapped + shift).astype(int)),
                    6,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        if len(self.points) == 4:
            roi = np.rint(np.asarray([
                self.points[LT],
                self.points[RT],
                self.points[RB],
                self.points[LB],
            ]) + shift).astype(np.int32)
            cv2.polylines(
                display,
                [roi],
                True,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (display.shape[1] - 1, 58),
            (0, 0, 0),
            -1,
        )
        display = cv2.addWeighted(overlay, 0.68, display, 0.32, 0.0)
        cv2.putText(
            display,
            self.status_message[:82],
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Order LB -> RB -> LT -> RT(snapped) | R reset | S save | Q quit",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return display

    def metric_color(self) -> tuple[int, int, int]:
        if self.result is None:
            return 0, 255, 255
        metrics = self.result.metrics
        worst_percent = max(
            metrics.max_square_error_percent,
            metrics.cell_size_cv_percent,
            metrics.horizontal_spacing_cv_percent,
            metrics.vertical_spacing_cv_percent,
        )
        worst_angle = metrics.max_orthogonality_error_deg
        if (
            metrics.marker_inside_roi
            and worst_percent <= 2.0
            and worst_angle <= 2.0
        ):
            return 0, 255, 0
        if (
            metrics.marker_inside_roi
            and worst_percent <= 5.0
            and worst_angle <= 5.0
        ):
            return 0, 220, 255
        return 0, 0, 255

    def draw_bev(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("표시할 BEV 결과가 없습니다.")
        bev = self.result.pure_bev
        panel_height = 132
        canvas_width = max(760, bev.shape[1])
        display = np.full(
            (panel_height + bev.shape[0], canvas_width, 3),
            24,
            dtype=np.uint8,
        )
        bev_x = (canvas_width - bev.shape[1]) // 2
        display[
            panel_height:panel_height + bev.shape[0],
            bev_x:bev_x + bev.shape[1],
        ] = bev
        marker = np.asarray(
            self.result.marker_points_bev, dtype=np.float32
        ).copy()
        marker[:, :, 0] += bev_x
        marker[:, :, 1] += panel_height
        color = self.metric_color()
        marker_polygon = np.rint(np.asarray([
            marker[0, 0],
            marker[0, 1],
            marker[1, 1],
            marker[1, 0],
        ])).astype(np.int32)
        cv2.polylines(
            display, [marker_polygon], True, color, 2, cv2.LINE_AA
        )
        for marker_point in marker_polygon:
            cv2.circle(
                display, tuple(marker_point), 4, color, -1, cv2.LINE_AA
            )
        metrics = self.result.metrics
        measured_ratio = float(np.mean(metrics.cell_ratios))
        ratio_error = abs(
            measured_ratio / EXPECTED_CELL_RATIO - 1.0
        ) * 100.0
        passed = (
            metrics.marker_inside_roi
            and ratio_error <= 2.0
            and metrics.max_orthogonality_error_deg <= 2.0
        )
        lines = [
            (
                f"REFERENCE BOX: {'PASS' if passed else 'CHECK'} "
                f"| target W/H {EXPECTED_CELL_RATIO:.6f}"
            ),
            (
                f"measured W/H {measured_ratio:.6f} "
                f"| ratio error {ratio_error:.3f}%"
            ),
            (
                f"max corner angle error "
                f"{metrics.max_orthogonality_error_deg:.3f}deg "
                f"| inside ROI {'YES' if metrics.marker_inside_roi else 'NO'}"
            ),
            (
                f"applied X scale {metrics.applied_x_scale:.6f} "
                f"| padding L/R/T/B {BEV_PADDING_LEFT_PX}/"
                f"{BEV_PADDING_RIGHT_PX}/{BEV_PADDING_TOP_PX}/"
                f"{BEV_PADDING_BOTTOM_PX}px"
            ),
            (
                f"expanded BEV {self.result.width}x{self.result.height} "
                "| empty bottom auto-cropped | R reset | S save"
            ),
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (10, 22 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color if index == 0 else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return display

    def show_bev(self) -> None:
        if self.result is None:
            return
        if self.bev_window_open:
            try:
                if cv2.getWindowProperty(
                    BEV_WINDOW,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    self.bev_window_open = False
            except cv2.error:
                self.bev_window_open = False
        if not self.bev_window_open:
            cv2.namedWindow(
                BEV_WINDOW,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            self.bev_window_open = True
        display = self.draw_bev()
        scale = min(
            1.0,
            900.0 / display.shape[1],
            720.0 / display.shape[0],
        )
        cv2.resizeWindow(
            BEV_WINDOW,
            max(1, int(round(display.shape[1] * scale))),
            max(1, int(round(display.shape[0] * scale))),
        )
        cv2.imshow(BEV_WINDOW, display)

    def print_metrics(self) -> None:
        if self.result is None:
            return
        metrics = self.result.metrics
        print()
        print("=" * 72)
        print("임시 BEV 흰 선 기준 상자 검사")
        print("=" * 72)
        print(
            "보정 전 상자 W/H: "
            + ", ".join(f"{value:.4f}" for value in metrics.raw_cell_ratios)
        )
        print(f"적용 X 배율  : {metrics.applied_x_scale:.6f}")
        print(
            "보정 후 상자 W/H: "
            + ", ".join(f"{value:.4f}" for value in metrics.cell_ratios)
        )
        print(
            f"목표 비율 오차: 평균 {metrics.mean_square_error_percent:.3f}%, "
            f"최대 {metrics.max_square_error_percent:.3f}%"
        )
        print(
            f"상자 크기 CV : {metrics.cell_size_cv_percent:.3f}%"
        )
        print(
            "간격 CV      : "
            f"가로 {metrics.horizontal_spacing_cv_percent:.3f}%, "
            f"세로 {metrics.vertical_spacing_cv_percent:.3f}%"
        )
        print(
            f"최대 직각오차: {metrics.max_orthogonality_error_deg:.3f}°"
        )
        print(
            "기준표식 포함: "
            f"{'예' if metrics.marker_inside_roi else '아니오'}"
        )
        print("=" * 72)

    def save(self) -> None:
        if self.result is None:
            self.status_message = "Nothing to save: select 4 points"
            self.dirty = True
            return
        result = self.result
        metrics = result.metrics
        arrays: dict[str, object] = {
            "src_points": result.src_points,
            "dst_points": result.dst_points,
            "homography": result.homography,
            "warp_width": np.int32(result.width),
            "warp_height": np.int32(result.height),
            "calibration_width": np.int32(self.calibration.width),
            "calibration_height": np.int32(self.calibration.height),
            "coordinate_space": np.asarray(
                "full_undistorted_camera_image"
            ),
            "source_extension_px": np.int32(SOURCE_EXTENSION_PX),
            "bev_padding_left_px": np.int32(BEV_PADDING_LEFT_PX),
            "bev_padding_right_px": np.int32(BEV_PADDING_RIGHT_PX),
            "bev_padding_top_px": np.int32(BEV_PADDING_TOP_PX),
            "bev_padding_bottom_px": np.int32(BEV_PADDING_BOTTOM_PX),
            "auto_crop_empty_bottom": np.bool_(AUTO_CROP_EMPTY_BOTTOM),
            "selection_canvas_width": np.int32(
                self.calibration.width + 2 * SOURCE_EXTENSION_PX
            ),
            "selection_canvas_height": np.int32(
                self.calibration.height + 2 * SOURCE_EXTENSION_PX
            ),
            "selection_mode": np.asarray(
                "four_point_parallel_marker_scaled"
            ),
            "top_edge_parallel_snapped": np.bool_(True),
            "input_image": np.asarray(str(self.arguments.image)),
            "calibration_file": np.asarray(
                str(self.arguments.calibration)
            ),
            "reference_marker_src_points": self.marker.points,
            "reference_marker_bev_points": result.marker_points_bev,
            "reference_box_inner_width_mm": np.float64(
                MARKER_INNER_WIDTH_MM
            ),
            "reference_box_inner_height_mm": np.float64(
                MARKER_INNER_HEIGHT_MM
            ),
            "reference_box_target_ratio": np.float64(
                EXPECTED_CELL_RATIO
            ),
            "reference_box_raw_ratios": np.asarray(
                metrics.raw_cell_ratios, dtype=np.float64
            ),
            "reference_box_ratios": np.asarray(
                metrics.cell_ratios, dtype=np.float64
            ),
            "reference_box_mean_ratio_error_percent": np.float64(
                metrics.mean_square_error_percent
            ),
            "reference_box_max_ratio_error_percent": np.float64(
                metrics.max_square_error_percent
            ),
            "reference_box_size_cv_percent": np.float64(
                metrics.cell_size_cv_percent
            ),
            "reference_box_horizontal_length_cv_percent": np.float64(
                metrics.horizontal_spacing_cv_percent
            ),
            "reference_box_vertical_length_cv_percent": np.float64(
                metrics.vertical_spacing_cv_percent
            ),
            "reference_box_max_corner_angle_error_deg": np.float64(
                metrics.max_orthogonality_error_deg
            ),
            "applied_x_scale": np.float64(
                metrics.applied_x_scale
            ),
            "marker_inside_roi": np.bool_(metrics.marker_inside_roi),
        }
        stem = self.output_path.stem
        preview_path = self.output_path.with_name(f"{stem}_preview.jpg")
        analysis_path = self.output_path.with_name(f"{stem}_analysis.jpg")
        source_path = self.output_path.with_name(f"{stem}_source.jpg")
        report_path = self.output_path.with_name(f"{stem}_report.json")
        atomic_write_image(preview_path, result.pure_bev)
        atomic_write_image(analysis_path, self.draw_bev())
        atomic_write_image(source_path, self.draw_source())
        atomic_write_json(
            report_path,
            {
                "input_image": str(self.arguments.image),
                "calibration": str(self.arguments.calibration),
                "src_points_order": list(POINT_NAMES),
                "src_points": result.src_points.tolist(),
                "source_extension_px": SOURCE_EXTENSION_PX,
                "bev_padding_px": {
                    "left": BEV_PADDING_LEFT_PX,
                    "right": BEV_PADDING_RIGHT_PX,
                    "top": BEV_PADDING_TOP_PX,
                    "bottom": BEV_PADDING_BOTTOM_PX,
                },
                "auto_crop_empty_bottom": AUTO_CROP_EMPTY_BOTTOM,
                "output_size": [result.width, result.height],
                "reference_box_inner_mm": [
                    MARKER_INNER_WIDTH_MM,
                    MARKER_INNER_HEIGHT_MM,
                ],
                "reference_box_target_ratio": EXPECTED_CELL_RATIO,
                "reference_box_raw_ratios": list(metrics.raw_cell_ratios),
                "applied_x_scale": metrics.applied_x_scale,
                "reference_box_ratios": list(metrics.cell_ratios),
                "mean_ratio_error_percent": (
                    metrics.mean_square_error_percent
                ),
                "max_ratio_error_percent": (
                    metrics.max_square_error_percent
                ),
                "reference_box_size_cv_percent": (
                    metrics.cell_size_cv_percent
                ),
                "horizontal_length_cv_percent": (
                    metrics.horizontal_spacing_cv_percent
                ),
                "vertical_length_cv_percent": (
                    metrics.vertical_spacing_cv_percent
                ),
                "max_corner_angle_error_deg": (
                    metrics.max_orthogonality_error_deg
                ),
                "reference_box_inside_roi": metrics.marker_inside_roi,
            },
        )
        # 보조 결과가 모두 성공한 뒤 실제 설정 NPZ를 마지막에 교체한다.
        backup = atomic_write_npz(self.output_path, arrays)
        self.status_message = f"SAVED {self.output_path.name}"
        self.dirty = True
        print()
        print(f"[저장 완료] {self.output_path}")
        if backup is not None:
            print(f"[이전 파일 백업] {backup}")
        print(f"[순수 BEV] {preview_path}")
        print(f"[왜곡 분석] {analysis_path}")
        print(f"[검사 보고서] {report_path}")

    def print_summary(self) -> None:
        print("=" * 78)
        print("캡처 프레임 평행 ROI + 흰 선 상자 BEV 검사")
        print("=" * 78)
        print(f"입력 이미지     : {self.arguments.image}")
        print(
            f"캘리브레이션    : {self.arguments.calibration} "
            f"({self.calibration.width}x{self.calibration.height}, "
            f"fx={self.calibration.camera_matrix[0, 0]:.3f})"
        )
        print(
            "기준 상자       : "
            f"내부 {MARKER_INNER_WIDTH_MM:.0f}x"
            f"{MARKER_INNER_HEIGHT_MM:.0f}mm, "
            f"목표 W/H={EXPECTED_CELL_RATIO:.6f}"
        )
        print(
            "코너 자동검출   : "
            f"{self.marker.minimum_fit_points}개"
        )
        print(
            "선택 가능 범위  : 원본 영상 상하좌우 "
            f"{SOURCE_EXTENSION_PX}px 확장"
        )
        print(f"저장 NPZ        : {self.output_path}")
        print()
        print("클릭 순서: 좌하단 -> 우하단 -> 좌상단 -> 우상단")
        print("우상단은 좌상단에서 시작하는 평행 반직선에 자동 스냅됩니다.")
        print("R 초기화 | S 저장 | Q/ESC 종료")
        print("=" * 78)

    def run(self) -> None:
        self.print_summary()
        try:
            while True:
                if self.dirty:
                    cv2.imshow(SOURCE_WINDOW, self.draw_source())
                    self.show_bev()
                    self.dirty = False
                key = cv2.waitKeyEx(20)
                if key >= 0:
                    ascii_key = key & 0xFF
                    if ascii_key in (27, ord("q"), ord("Q")):
                        break
                    if ascii_key in (ord("r"), ord("R")):
                        self.reset()
                    elif ascii_key in (ord("s"), ord("S")):
                        self.save()
                try:
                    if cv2.getWindowProperty(
                        SOURCE_WINDOW,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1.0:
                        break
                except cv2.error:
                    break
        finally:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        tool = ParallelRoiTool(arguments)
        tool.run()
        return 0
    except KeyboardInterrupt:
        print("\n[중단] BEV ROI 도구를 종료했습니다.")
        return 130
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        OSError,
        cv2.error,
    ) as error:
        print()
        print("=" * 72)
        print("캡처 프레임 BEV ROI 도구를 시작하지 못했습니다.")
        print("=" * 72)
        print(error)
        print("=" * 72)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
