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
DEFAULT_OUTPUT = PROJECT_ROOT / "bev_params_parallel_marker.npz"

SOURCE_WINDOW = "Captured frame - parallel BEV ROI"
BEV_WINDOW = "Temporary BEV - marker distortion"

LB, RB, LT, RT = 0, 1, 2, 3
POINT_NAMES = ("LB", "RB", "LT", "RT")
MIN_EDGE_PX = 25.0
MIN_AREA_PX2 = 1000.0
MIN_OUTPUT_SIDE = 64
MAX_OUTPUT_SIDE = 1400
SOURCE_EXTENSION_PX = 10

# 흰 선 격자의 각 칸은 BEV에서 정사각형(가로/세로=1)이어야 한다.
EXPECTED_CELL_RATIO = 1.0


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
            "흰 선 격자의 각 칸이 정사각형에 가까운지 검사합니다."
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
    parser.add_argument(
        "--checkerboard",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        default=(10, 7),
        help="흑백 체커보드 내부 코너 수 (기본값: 10 7)",
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
    pattern: tuple[int, int],
) -> MarkerDetection:
    """표준 흑백 체커보드의 모든 내부 코너를 검출한다."""
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        ),
    )
    if not found or corners is None:
        raise ValueError(
            f"흑백 체커보드 내부 코너 {pattern[0]}x{pattern[1]}를 "
            "검출하지 못했습니다."
        )
    columns, rows = pattern
    grid = np.asarray(corners, np.float32).reshape(rows, columns, 2)
    return MarkerDetection(
        points=grid,
        threshold_spread_px=0.0,
        minimum_fit_points=rows * columns,
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
        self.checkerboard_pattern = tuple(arguments.checkerboard)
        self.marker = detect_reference_marker(
            self.frame, self.checkerboard_pattern
        )
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
                "이 ROI에서는 격자 칸을 정사각형으로 맞출 수 없습니다. "
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
        corrected_width = max(
            2,
            int(math.ceil(corrected_content_width)) + 1,
        )
        corrected_height = max(
            2,
            int(math.ceil(corrected_content_height)) + 1,
        )
        if (
            corrected_width < MIN_OUTPUT_SIDE
            or corrected_height < MIN_OUTPUT_SIDE
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
        homography = correction @ base_homography
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
        marker_bev = transform_points(
            homography, self.marker.points
        ).reshape(grid_shape)
        cell_ratios, cell_sizes, horizontal, vertical, angle_errors = (
            grid_cell_measurements(marker_bev)
        )
        raw_errors = np.abs(raw_cell_ratios - 1.0) * 100.0
        square_errors = np.abs(cell_ratios - 1.0) * 100.0
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
        cv2.drawChessboardCorners(
            display,
            self.checkerboard_pattern,
            marker.reshape(-1, 1, 2),
            True,
        )
        label_point = tuple(np.rint(marker[0, 0]).astype(int))
        cv2.putText(
            display,
            f"checkerboard {self.checkerboard_pattern[0]}x"
            f"{self.checkerboard_pattern[1]} inner corners",
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
        cv2.drawChessboardCorners(
            display,
            self.checkerboard_pattern,
            marker.reshape(-1, 1, 2),
            True,
        )
        metrics = self.result.metrics
        lines = [
            (
                f"raw mean square error "
                f"{metrics.raw_mean_square_error_percent:.2f}% "
                f"| applied X {metrics.applied_x_scale:.4f}"
            ),
            (
                f"cell square error mean {metrics.mean_square_error_percent:.2f}% "
                f"| max {metrics.max_square_error_percent:.2f}%"
            ),
            (
                f"cell size CV {metrics.cell_size_cv_percent:.2f}% "
                f"| spacing CV X {metrics.horizontal_spacing_cv_percent:.2f}% "
                f"Y {metrics.vertical_spacing_cv_percent:.2f}%"
            ),
            (
                f"max orthogonality error "
                f"{metrics.max_orthogonality_error_deg:.2f}deg "
                f"| cells {len(metrics.cell_ratios)}"
            ),
            (
                f"marker inside ROI: {'YES' if metrics.marker_inside_roi else 'NO'} "
                "| R reset | S save"
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
        print("임시 BEV 체커보드 칸 정사각형 검사")
        print("=" * 72)
        print(
            "보정 전 칸 비율: "
            + ", ".join(f"{value:.4f}" for value in metrics.raw_cell_ratios)
        )
        print(f"적용 X 배율  : {metrics.applied_x_scale:.6f}")
        print(
            "보정 후 칸 비율: "
            + ", ".join(f"{value:.4f}" for value in metrics.cell_ratios)
        )
        print(
            f"정사각형 오차: 평균 {metrics.mean_square_error_percent:.3f}%, "
            f"최대 {metrics.max_square_error_percent:.3f}%"
        )
        print(
            f"칸 크기 CV   : {metrics.cell_size_cv_percent:.3f}%"
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
            "selection_canvas_width": np.int32(
                self.calibration.width + 2 * SOURCE_EXTENSION_PX
            ),
            "selection_canvas_height": np.int32(
                self.calibration.height + 2 * SOURCE_EXTENSION_PX
            ),
            "selection_mode": np.asarray(
                "four_point_parallel_checkerboard_square_scaled"
            ),
            "top_edge_parallel_snapped": np.bool_(True),
            "input_image": np.asarray(str(self.arguments.image)),
            "calibration_file": np.asarray(
                str(self.arguments.calibration)
            ),
            "reference_marker_src_points": self.marker.points,
            "reference_marker_bev_points": result.marker_points_bev,
            "checkerboard_size": np.asarray(
                self.checkerboard_pattern, dtype=np.int32
            ),
            "checkerboard_raw_cell_ratios": np.asarray(
                metrics.raw_cell_ratios, dtype=np.float64
            ),
            "checkerboard_cell_ratios": np.asarray(
                metrics.cell_ratios, dtype=np.float64
            ),
            "checkerboard_mean_square_error_percent": np.float64(
                metrics.mean_square_error_percent
            ),
            "checkerboard_max_square_error_percent": np.float64(
                metrics.max_square_error_percent
            ),
            "checkerboard_cell_size_cv_percent": np.float64(
                metrics.cell_size_cv_percent
            ),
            "checkerboard_horizontal_spacing_cv_percent": np.float64(
                metrics.horizontal_spacing_cv_percent
            ),
            "checkerboard_vertical_spacing_cv_percent": np.float64(
                metrics.vertical_spacing_cv_percent
            ),
            "checkerboard_max_orthogonality_error_deg": np.float64(
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
                "output_size": [result.width, result.height],
                "checkerboard_inner_corners": list(
                    self.checkerboard_pattern
                ),
                "raw_cell_ratios": list(metrics.raw_cell_ratios),
                "applied_x_scale": metrics.applied_x_scale,
                "cell_ratios": list(metrics.cell_ratios),
                "mean_square_error_percent": (
                    metrics.mean_square_error_percent
                ),
                "max_square_error_percent": (
                    metrics.max_square_error_percent
                ),
                "cell_size_cv_percent": metrics.cell_size_cv_percent,
                "horizontal_spacing_cv_percent": (
                    metrics.horizontal_spacing_cv_percent
                ),
                "vertical_spacing_cv_percent": (
                    metrics.vertical_spacing_cv_percent
                ),
                "max_orthogonality_error_deg": (
                    metrics.max_orthogonality_error_deg
                ),
                "marker_inside_roi": metrics.marker_inside_roi,
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
        print("캡처 프레임 평행 ROI + 체커보드 셀 BEV 검사")
        print("=" * 78)
        print(f"입력 이미지     : {self.arguments.image}")
        print(
            f"캘리브레이션    : {self.arguments.calibration} "
            f"({self.calibration.width}x{self.calibration.height}, "
            f"fx={self.calibration.camera_matrix[0, 0]:.3f})"
        )
        print(
            "체커보드        : "
            f"내부 코너 {self.checkerboard_pattern[0]}x"
            f"{self.checkerboard_pattern[1]}, 목표 셀 W/H=1.0"
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
