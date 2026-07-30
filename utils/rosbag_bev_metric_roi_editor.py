#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""체커보드 metric 평면을 고정한 ROS 2 bag용 BEV ROI 편집기.

먼저 webcam_bev_drag_checkerboard_live.py에서 새 metric BEV NPZ를 저장해야
한다. 이 편집기는 그 NPZ의 image<->ground-plane(mm) homography를 변경하지
않고, 같은 카메라 자세로 녹화한 rosbag 영상 위에서 ROI 직사각형만 편집한다.
"""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

try:
    import rosbag2_py
    from rclpy.logging import LoggingSeverity, set_logger_level
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as ros_import_error:
    rosbag2_py = None
    LoggingSeverity = None
    set_logger_level = None
    deserialize_message = None
    get_message = None
    ROS_IMPORT_ERROR: ImportError | None = ros_import_error
else:
    ROS_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = PROJECT_ROOT / "camera_calibration.npz"
DEFAULT_METRIC_BEV = PROJECT_ROOT / "bev_params_y_auto.npz"
DEFAULT_OUTPUT = PROJECT_ROOT / "bev_params_road_roi.npz"

SUPPORTED_TOPIC_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
}
PREFERRED_TOPICS = (
    "/camera/lane/raw/compressed",
    "/image_raw/compressed",
    "/camera/lane/raw",
    "/image_raw",
)

DEFAULT_MAX_OUTPUT_SIDE = 4096
DEFAULT_MAX_OUTPUT_PIXELS = 8_000_000
MIN_OUTPUT_SIDE = 32
MIN_SOURCE_EDGE_PX = 8.0
MIN_SOURCE_AREA_PX2 = 500.0
HANDLE_RADIUS_PX = 15
MAX_BEV_PREVIEW_WIDTH = 900
MAX_BEV_PREVIEW_HEIGHT = 720
GRID_INTERVAL_MM = 250.0

SOURCE_WINDOW = "Rosbag - Undistorted Metric ROI"
BEV_WINDOW = "Metric BEV Road ROI Preview"

LB, RB, LT, RT = 0, 1, 2, 3
POINT_NAMES = ("LB", "RB", "LT", "RT")
HANDLE_ORDER = (
    "LB",
    "RB",
    "LT",
    "RT",
    "LEFT",
    "RIGHT",
    "BOTTOM",
    "TOP",
    "CENTER",
)

# waitKeyEx()가 반환하는 확장 키 코드만 둔다. 81~84는 일부 예전
# waitKey() 구현의 방향키 값이지만 Q/R/S/T와 구별할 수 없어 사용하지 않는다.
LEFT_KEY_CODES = {2424832, 65361}
UP_KEY_CODES = {2490368, 65362}
RIGHT_KEY_CODES = {2555904, 65363}
DOWN_KEY_CODES = {2621440, 65364}


@dataclass(frozen=True)
class Calibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    new_camera_matrix: np.ndarray
    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray
    sha256: str


@dataclass(frozen=True)
class MetricPlane:
    plane_to_image: np.ndarray
    image_to_plane: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    seed_u_min: float
    seed_u_max: float
    seed_v_bottom: float
    seed_v_top: float
    pixels_per_mm: float
    square_size_mm: float
    calibration_width: int
    calibration_height: int
    source_arrays: dict[str, np.ndarray]
    source_path: Path


@dataclass(frozen=True)
class RoiState:
    u_min: float
    u_max: float
    v_bottom: float
    v_top: float
    pixels_per_mm: float
    aspect_locked: bool = False

    @property
    def width_mm(self) -> float:
        return self.u_max - self.u_min

    @property
    def height_mm(self) -> float:
        return self.v_top - self.v_bottom

    @property
    def center_u(self) -> float:
        return (self.u_min + self.u_max) / 2.0

    @property
    def center_v(self) -> float:
        return (self.v_bottom + self.v_top) / 2.0


@dataclass(frozen=True)
class BevGeometry:
    metric_points_mm: np.ndarray
    src_points: np.ndarray
    dst_points: np.ndarray
    homography: np.ndarray
    warp_width: int
    warp_height: int
    content_width_px: float
    content_height_px: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def npz_scalar(arrays: dict[str, np.ndarray], key: str) -> object:
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError(f"{key}는 스칼라여야 합니다: {value.shape}")
    scalar = value.reshape(-1)[0]
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def npz_float(arrays: dict[str, np.ndarray], key: str) -> float:
    try:
        value = float(npz_scalar(arrays, key))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key}는 실수여야 합니다.") from error
    if not np.isfinite(value):
        raise ValueError(f"{key}에 NaN 또는 inf가 있습니다.")
    return value


def npz_int(arrays: dict[str, np.ndarray], key: str) -> int:
    value = npz_float(arrays, key)
    if not value.is_integer():
        raise ValueError(f"{key}는 정확한 정수여야 합니다: {value}")
    return int(value)


def perspective_points(
    homography: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(homography, dtype=np.float64)
    source = np.asarray(points, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"homography shape가 잘못됐습니다: {matrix.shape}")
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"point shape가 잘못됐습니다: {source.shape}")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(source)):
        raise ValueError("homography 또는 좌표에 NaN/inf가 있습니다.")
    result = cv2.perspectiveTransform(
        source.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)
    if not np.all(np.isfinite(result)):
        raise ValueError("homography 투영 결과가 유효하지 않습니다.")
    return result


def vector_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(
        np.linalg.norm(first) * np.linalg.norm(second)
    )
    if denominator <= 1.0e-12:
        raise ValueError("길이가 0인 축 벡터가 있습니다.")
    cosine = float(np.clip(
        np.dot(first, second) / denominator,
        -1.0,
        1.0,
    ))
    return float(np.degrees(np.arccos(cosine)))


def mouse_wheel_delta(flags: int) -> int:
    """OpenCV Python 빌드와 무관하게 signed wheel delta를 꺼낸다."""
    helper = getattr(cv2, "getMouseWheelDelta", None)
    if callable(helper):
        return int(helper(flags))
    high_word = (int(flags) >> 16) & 0xFFFF
    return high_word - 0x10000 if high_word & 0x8000 else high_word


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            encoding="utf-8",
            newline="",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}_",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"이미지를 저장하지 못했습니다: {path}")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_npz(
    path: Path,
    arrays: dict[str, object],
) -> Path | None:
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


def discover_default_bag() -> Path | None:
    candidates = [
        metadata.parent
        for metadata in PROJECT_ROOT.rglob("metadata.yaml")
        if not any(
            part in {"build", "install", "log"}
            for part in metadata.parts
        )
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (
            (path / "metadata.yaml").stat().st_mtime_ns,
            path.name,
        ),
    )


def normalize_bag_path(path: Path) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if resolved.is_file():
        if resolved.name == "metadata.yaml" or resolved.suffix == ".db3":
            resolved = resolved.parent
        else:
            raise ValueError(
                "--bag은 rosbag 디렉터리, metadata.yaml 또는 db3여야 합니다."
            )
    if not resolved.is_dir() or not (resolved / "metadata.yaml").is_file():
        raise FileNotFoundError(
            f"ROS bag metadata.yaml을 찾지 못했습니다: {resolved}"
        )
    return resolved


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "체커보드로 고정한 실제 mm 평면을 ROS 2 bag 영상에 적용하고 "
            "도로 ROI를 실제 비율로 편집합니다."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=None,
        help="ROS 2 bag 디렉터리. 생략하면 프로젝트의 최신 bag 자동 선택",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="bag 영상 토픽. 생략하면 지원되는 영상 토픽 자동 선택",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="동일 카메라의 640x480 camera_calibration.npz",
    )
    parser.add_argument(
        "--metric-bev",
        type=Path,
        default=DEFAULT_METRIC_BEV,
        help="새 체커보드 코드에서 저장한 metric BEV NPZ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="최종 도로 ROI BEV NPZ",
    )
    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        default=None,
        help=(
            "렌더링 해상도(px/mm). 생략하면 metric BEV의 값을 유지. "
            "실제 가로/세로 비율에는 영향을 주지 않음"
        ),
    )
    parser.add_argument(
        "--max-output-side",
        type=int,
        default=DEFAULT_MAX_OUTPUT_SIDE,
        help=f"출력 한 변 상한(기본값: {DEFAULT_MAX_OUTPUT_SIDE})",
    )
    parser.add_argument(
        "--max-output-pixels",
        type=int,
        default=DEFAULT_MAX_OUTPUT_PIXELS,
        help=f"출력 총 픽셀 상한(기본값: {DEFAULT_MAX_OUTPUT_PIXELS})",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="처음 표시할 bag 프레임 번호",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="재생 배속(기본값: 1.0)",
    )
    parser.add_argument(
        "--aspect-lock",
        action="store_true",
        help="ROI 코너/변 조절 시 시작 W:H를 고정",
    )
    arguments = parser.parse_args(argv)

    if arguments.bag is None:
        arguments.bag = discover_default_bag()
        if arguments.bag is None:
            parser.error("--bag을 지정하거나 프로젝트에 rosbag을 넣으세요.")
    arguments.bag = normalize_bag_path(arguments.bag)

    for name in ("calibration", "metric_bev", "output"):
        path = Path(getattr(arguments, name)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        setattr(arguments, name, path.resolve())

    if arguments.topic is not None:
        arguments.topic = arguments.topic.strip()
        if not arguments.topic:
            parser.error("--topic은 빈 문자열일 수 없습니다.")
    if (
        arguments.pixels_per_mm is not None
        and (
            not np.isfinite(arguments.pixels_per_mm)
            or arguments.pixels_per_mm <= 0.0
        )
    ):
        parser.error("--pixels-per-mm은 양수여야 합니다.")
    if not (64 <= arguments.max_output_side <= 8192):
        parser.error("--max-output-side는 64 이상 8192 이하여야 합니다.")
    if not (4096 <= arguments.max_output_pixels <= 64_000_000):
        parser.error(
            "--max-output-pixels는 4096 이상 64000000 이하여야 합니다."
        )
    if arguments.start_frame < 0:
        parser.error("--start-frame은 0 이상이어야 합니다.")
    if (
        not np.isfinite(arguments.playback_rate)
        or arguments.playback_rate <= 0.0
        or arguments.playback_rate > 16.0
    ):
        parser.error("--playback-rate는 0보다 크고 16 이하여야 합니다.")
    for name in ("calibration", "metric_bev", "output"):
        if Path(getattr(arguments, name)).suffix.lower() != ".npz":
            parser.error(f"--{name.replace('_', '-')} 확장자는 .npz여야 합니다.")
    if arguments.output == arguments.calibration:
        parser.error(
            "--output은 카메라 캘리브레이션 파일과 같을 수 없습니다."
        )
    return arguments


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
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}

    camera_matrix = np.asarray(arrays["camera_matrix"], dtype=np.float64)
    new_camera_matrix = np.asarray(
        arrays["new_camera_matrix"],
        dtype=np.float64,
    )
    distortion_raw = np.asarray(
        arrays["distortion_coefficients"],
        dtype=np.float64,
    )
    if camera_matrix.shape != (3, 3) or new_camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix와 new_camera_matrix는 3x3이어야 합니다.")
    if (
        distortion_raw.ndim not in (1, 2)
        or (distortion_raw.ndim == 2 and 1 not in distortion_raw.shape)
        or distortion_raw.size not in (4, 5, 8, 12, 14)
    ):
        raise ValueError(
            "distortion_coefficients는 길이 4/5/8/12/14 벡터여야 합니다."
        )
    distortion = distortion_raw.reshape(-1)
    if not (
        np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(new_camera_matrix))
        and np.all(np.isfinite(distortion))
    ):
        raise ValueError("캘리브레이션 행렬에 NaN/inf가 있습니다.")
    if (
        camera_matrix[0, 0] <= 0.0
        or camera_matrix[1, 1] <= 0.0
        or new_camera_matrix[0, 0] <= 0.0
        or new_camera_matrix[1, 1] <= 0.0
        or np.linalg.matrix_rank(camera_matrix) < 3
        or np.linalg.matrix_rank(new_camera_matrix) < 3
    ):
        raise ValueError("캘리브레이션 카메라 행렬이 유효하지 않습니다.")
    width = npz_int(arrays, "image_width")
    height = npz_int(arrays, "image_height")
    if width <= 0 or height <= 0:
        raise ValueError("캘리브레이션 해상도가 유효하지 않습니다.")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    if (
        map_x.shape != (height, width)
        or map_y.shape != (height, width)
        or not np.all(np.isfinite(map_x))
        or not np.all(np.isfinite(map_y))
    ):
        raise ValueError("유효한 왜곡 보정 맵을 생성하지 못했습니다.")
    return Calibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        new_camera_matrix=new_camera_matrix,
        width=width,
        height=height,
        map_x=map_x,
        map_y=map_y,
        sha256=sha256_file(path),
    )


def load_metric_plane(
    path: Path,
    calibration: Calibration,
    pixels_per_mm_override: float | None,
) -> MetricPlane:
    if not path.is_file():
        raise FileNotFoundError(f"metric BEV 파일이 없습니다: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files}

    required = {
        "plane_to_image_homography",
        "image_to_plane_homography",
        "effective_metric_points_mm",
        "pixels_per_mm",
        "square_size_mm",
        "calibration_width",
        "calibration_height",
        "coordinate_space",
    }
    missing = required - set(arrays)
    if missing:
        selection_mode = (
            str(npz_scalar(arrays, "selection_mode"))
            if "selection_mode" in arrays
            else "unknown"
        )
        raise ValueError(
            "현재 BEV NPZ는 실제 바닥 평면 정보가 없는 구형 파일입니다.\n"
            f"selection_mode: {selection_mode}\n"
            f"누락 키: {sorted(missing)}\n"
            "먼저 업데이트된 webcam_bev_drag_checkerboard_live.py에서 "
            "체커보드가 보이는 상태로 S를 눌러 새 metric NPZ를 저장하세요."
        )

    coordinate_space = str(npz_scalar(arrays, "coordinate_space"))
    if coordinate_space != "full_undistorted_camera_image":
        raise ValueError(
            "metric BEV 좌표계가 현재 파이프라인과 다릅니다: "
            f"{coordinate_space!r}. "
            "'full_undistorted_camera_image' 좌표만 사용할 수 있습니다."
        )

    plane_to_image = np.asarray(
        arrays["plane_to_image_homography"],
        dtype=np.float64,
    )
    image_to_plane = np.asarray(
        arrays["image_to_plane_homography"],
        dtype=np.float64,
    )
    for name, matrix in (
        ("plane_to_image_homography", plane_to_image),
        ("image_to_plane_homography", image_to_plane),
    ):
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or np.linalg.matrix_rank(matrix) < 3
        ):
            raise ValueError(f"{name}가 유효한 3x3 행렬이 아닙니다.")

    product = plane_to_image @ image_to_plane
    if abs(float(product[2, 2])) < 1.0e-12:
        raise ValueError("두 평면 homography의 곱을 정규화할 수 없습니다.")
    product /= product[2, 2]
    if not np.allclose(product, np.eye(3), rtol=1.0e-5, atol=1.0e-5):
        raise ValueError("image/plane homography가 서로 역행렬이 아닙니다.")

    calibration_width = npz_int(arrays, "calibration_width")
    calibration_height = npz_int(arrays, "calibration_height")
    if (
        calibration_width != calibration.width
        or calibration_height != calibration.height
    ):
        raise ValueError(
            "metric BEV와 카메라 캘리브레이션 해상도가 다릅니다: "
            f"{calibration_width}x{calibration_height} != "
            f"{calibration.width}x{calibration.height}"
        )
    if "calibration_sha256" in arrays:
        expected_hash = str(npz_scalar(arrays, "calibration_sha256"))
        if expected_hash and expected_hash != calibration.sha256:
            raise ValueError(
                "metric BEV를 만든 카메라 캘리브레이션과 현재 파일이 다릅니다."
            )

    metric_points = np.asarray(
        arrays["effective_metric_points_mm"],
        dtype=np.float64,
    )
    if metric_points.shape != (4, 2) or not np.all(np.isfinite(metric_points)):
        raise ValueError("effective_metric_points_mm shape가 잘못됐습니다.")
    left_bottom, right_bottom, left_top, right_top = metric_points
    horizontal_bottom = right_bottom - left_bottom
    horizontal_top = right_top - left_top
    vertical_left = left_top - left_bottom
    vertical_right = right_top - right_bottom
    lengths = [
        float(np.linalg.norm(vector))
        for vector in (
            horizontal_bottom,
            horizontal_top,
            vertical_left,
            vertical_right,
        )
    ]
    if min(lengths) <= 1.0e-6:
        raise ValueError("metric ROI에 길이가 0인 변이 있습니다.")
    u_axis = (
        horizontal_bottom / lengths[0]
        + horizontal_top / lengths[1]
    )
    v_axis = (
        vertical_left / lengths[2]
        + vertical_right / lengths[3]
    )
    u_axis /= np.linalg.norm(u_axis)
    v_axis /= np.linalg.norm(v_axis)
    orthogonality_error = abs(
        90.0 - vector_angle_degrees(u_axis, v_axis)
    )
    if orthogonality_error > 0.1:
        raise ValueError(
            "저장된 metric ROI 축이 직각이 아닙니다: "
            f"오차 {orthogonality_error:.4f}°"
        )
    # 부동소수점 잔차를 없애기 위해 v를 u에 정확히 직교화한다.
    v_axis = v_axis - np.dot(v_axis, u_axis) * u_axis
    v_axis /= np.linalg.norm(v_axis)
    if np.dot(v_axis, vertical_left) < 0.0:
        v_axis *= -1.0

    u_min = float(np.mean(metric_points[[LB, LT]] @ u_axis))
    u_max = float(np.mean(metric_points[[RB, RT]] @ u_axis))
    v_bottom = float(np.mean(metric_points[[LB, RB]] @ v_axis))
    v_top = float(np.mean(metric_points[[LT, RT]] @ v_axis))
    if u_max <= u_min or v_top <= v_bottom:
        raise ValueError("metric ROI의 LB/RB/LT/RT 순서가 잘못됐습니다.")

    reconstructed = np.asarray([
        u_min * u_axis + v_bottom * v_axis,
        u_max * u_axis + v_bottom * v_axis,
        u_min * u_axis + v_top * v_axis,
        u_max * u_axis + v_top * v_axis,
    ])
    square_size_mm = npz_float(arrays, "square_size_mm")
    if square_size_mm <= 0.0:
        raise ValueError("square_size_mm은 양수여야 합니다.")
    reconstruction_error = float(np.max(np.linalg.norm(
        reconstructed - metric_points,
        axis=1,
    )))
    if reconstruction_error > max(0.5, square_size_mm * 0.05):
        raise ValueError(
            "저장된 metric ROI가 직사각형과 일치하지 않습니다: "
            f"최대 {reconstruction_error:.3f}mm"
        )
    if "src_points" in arrays:
        stored_source = np.asarray(arrays["src_points"], dtype=np.float64)
        if (
            stored_source.shape != (4, 2)
            or not np.all(np.isfinite(stored_source))
        ):
            raise ValueError("저장된 src_points shape가 잘못됐습니다.")
        projected_source = perspective_points(
            plane_to_image,
            metric_points,
        )
        source_error = float(np.max(np.linalg.norm(
            projected_source - stored_source,
            axis=1,
        )))
        if source_error > 0.1:
            raise ValueError(
                "metric 평면과 저장된 BEV 사다리꼴이 서로 다릅니다: "
                f"최대 {source_error:.3f}px"
            )

    pixels_per_mm = (
        float(pixels_per_mm_override)
        if pixels_per_mm_override is not None
        else npz_float(arrays, "pixels_per_mm")
    )
    if pixels_per_mm <= 0.0:
        raise ValueError("pixels_per_mm은 양수여야 합니다.")

    return MetricPlane(
        plane_to_image=plane_to_image,
        image_to_plane=image_to_plane,
        u_axis=u_axis,
        v_axis=v_axis,
        seed_u_min=u_min,
        seed_u_max=u_max,
        seed_v_bottom=v_bottom,
        seed_v_top=v_top,
        pixels_per_mm=pixels_per_mm,
        square_size_mm=square_size_mm,
        calibration_width=calibration_width,
        calibration_height=calibration_height,
        source_arrays=arrays,
        source_path=path,
    )


def ros_raw_image_to_bgr(message: object) -> np.ndarray:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).strip().lower()
    if width <= 0 or height <= 0:
        raise ValueError(f"ROS Image 크기가 잘못됐습니다: {width}x{height}")
    if encoding in ("bgr8", "rgb8", "8uc3"):
        channels = 3
    elif encoding in ("bgra8", "rgba8", "8uc4"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    elif encoding in ("yuv422_yuy2", "yuv422", "yuyv"):
        channels = 2
    else:
        raise ValueError(f"지원하지 않는 ROS Image encoding: {encoding}")
    packed_bytes = width * channels
    if step < packed_bytes:
        raise ValueError(f"ROS Image step이 너무 작습니다: {step}")
    data = np.frombuffer(message.data, dtype=np.uint8)
    required = step * height
    if data.size < required:
        raise ValueError("ROS Image data 길이가 부족합니다.")
    packed = data[:required].reshape(height, step)[:, :packed_bytes].copy()
    if channels == 1:
        return cv2.cvtColor(
            packed.reshape(height, width),
            cv2.COLOR_GRAY2BGR,
        )
    image = packed.reshape(height, width, channels)
    if encoding in ("bgr8", "8uc3"):
        return image
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "yuv422":
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)


class RosbagFrameSource:
    def __init__(
        self,
        bag_path: Path,
        requested_topic: str | None,
    ) -> None:
        if rosbag2_py is None:
            raise RuntimeError(
                "ROS 2 rosbag Python 모듈을 불러오지 못했습니다.\n"
                "다음 환경에서 실행하세요:\n"
                "  source /opt/ros/humble/setup.bash\n"
                f"원인: {ROS_IMPORT_ERROR}"
            )
        if set_logger_level is not None and LoggingSeverity is not None:
            try:
                set_logger_level(
                    "rosbag2_storage",
                    LoggingSeverity.WARN,
                )
            except Exception:
                pass

        self.bag_path = bag_path
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(
            rosbag2_py.StorageOptions(
                uri=str(bag_path),
                storage_id="",
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        topic_types = {
            item.name: item.type
            for item in self.reader.get_all_topics_and_types()
        }
        if requested_topic is not None:
            if requested_topic not in topic_types:
                raise ValueError(
                    f"bag에 토픽이 없습니다: {requested_topic}\n"
                    f"사용 가능: {sorted(topic_types)}"
                )
            if topic_types[requested_topic] not in SUPPORTED_TOPIC_TYPES:
                raise ValueError(
                    "지원하지 않는 영상 메시지 타입입니다: "
                    f"{topic_types[requested_topic]}"
                )
            self.topic = requested_topic
        else:
            supported = [
                name
                for name, type_name in topic_types.items()
                if type_name in SUPPORTED_TOPIC_TYPES
            ]
            if not supported:
                raise ValueError(
                    "bag에 sensor_msgs/Image 또는 CompressedImage가 없습니다."
                )
            preferred = [
                name for name in PREFERRED_TOPICS if name in supported
            ]
            self.topic = preferred[0] if preferred else sorted(supported)[0]

        self.type_name = topic_types[self.topic]
        self.message_type = get_message(self.type_name)
        self.reader.set_filter(
            rosbag2_py.StorageFilter(topics=[self.topic])
        )
        self.timestamps: list[int] = []
        while self.reader.has_next():
            _, _, timestamp = self.reader.read_next()
            self.timestamps.append(int(timestamp))
        if not self.timestamps:
            raise ValueError(f"선택한 영상 토픽에 메시지가 없습니다: {self.topic}")
        if any(
            current < previous
            for previous, current in zip(
                self.timestamps,
                self.timestamps[1:],
            )
        ):
            raise ValueError("bag 영상 타임스탬프가 시간순이 아닙니다.")
        self.reader.seek(self.timestamps[0])

    def __len__(self) -> int:
        return len(self.timestamps)

    def read_frame(self, index: int) -> tuple[np.ndarray, int, str, str]:
        if index < 0 or index >= len(self.timestamps):
            raise IndexError(f"bag frame index 범위 초과: {index}")
        target_timestamp = self.timestamps[index]
        first_duplicate = bisect.bisect_left(
            self.timestamps,
            target_timestamp,
        )
        duplicate_offset = index - first_duplicate
        self.reader.seek(target_timestamp)
        record = None
        for _ in range(duplicate_offset + 1):
            if not self.reader.has_next():
                raise RuntimeError("bag seek 후 프레임을 읽지 못했습니다.")
            record = self.reader.read_next()
        assert record is not None
        topic, serialized, timestamp = record
        if topic != self.topic or int(timestamp) != target_timestamp:
            raise RuntimeError(
                "bag seek 결과가 timestamp index와 일치하지 않습니다."
            )
        message = deserialize_message(serialized, self.message_type)
        if self.type_name == "sensor_msgs/msg/CompressedImage":
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None or frame.size == 0:
                raise ValueError("CompressedImage JPEG/PNG 디코딩에 실패했습니다.")
            encoding = str(message.format)
        else:
            frame = ros_raw_image_to_bgr(message)
            encoding = str(message.encoding)
        frame_id = str(message.header.frame_id)
        return frame, target_timestamp, encoding, frame_id

    def index_near_time(self, timestamp: int) -> int:
        index = bisect.bisect_left(self.timestamps, timestamp)
        if index <= 0:
            return 0
        if index >= len(self.timestamps):
            return len(self.timestamps) - 1
        before = self.timestamps[index - 1]
        after = self.timestamps[index]
        return index - 1 if timestamp - before <= after - timestamp else index


class MetricRosbagRoiEditor:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.calibration = load_calibration(arguments.calibration)
        self.metric_plane = load_metric_plane(
            arguments.metric_bev,
            self.calibration,
            arguments.pixels_per_mm,
        )
        self.frame_source = RosbagFrameSource(
            arguments.bag,
            arguments.topic,
        )
        self.output_path: Path = arguments.output
        self.max_output_side = int(arguments.max_output_side)
        self.max_output_pixels = int(arguments.max_output_pixels)
        self.playback_rate = float(arguments.playback_rate)

        initial_index = min(
            int(arguments.start_frame),
            len(self.frame_source) - 1,
        )
        plane = self.metric_plane
        self.seed_state = RoiState(
            u_min=plane.seed_u_min,
            u_max=plane.seed_u_max,
            v_bottom=plane.seed_v_bottom,
            v_top=plane.seed_v_top,
            pixels_per_mm=plane.pixels_per_mm,
            aspect_locked=bool(arguments.aspect_lock),
        )
        self.state = self.seed_state
        self.geometry: BevGeometry | None = None
        self.geometry_error: str | None = None
        self.undo_stack: list[RoiState] = []
        self.redo_stack: list[RoiState] = []

        self.current_index = -1
        self.requested_index = initial_index
        self.current_raw_frame: np.ndarray | None = None
        self.current_frame: np.ndarray | None = None
        self.current_timestamp = self.frame_source.timestamps[initial_index]
        self.current_encoding = ""
        self.current_frame_id = ""
        self.current_bev: np.ndarray | None = None

        self.playing = False
        self.next_play_deadline = 0.0
        self.dirty = True
        self.status_message = ""

        self.drag_mode: str | None = None
        self.drag_start_uv: tuple[float, float] | None = None
        self.drag_start_state: RoiState | None = None
        self.drag_changed = False

        self.updating_trackbar = False
        self.bev_window_open = False
        self.bev_window_shape: tuple[int, int] | None = None

        cv2.namedWindow(SOURCE_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(SOURCE_WINDOW, self.mouse_callback)
        self.has_frame_trackbar = len(self.frame_source) > 1
        if self.has_frame_trackbar:
            cv2.createTrackbar(
                "Frame",
                SOURCE_WINDOW,
                initial_index,
                len(self.frame_source) - 1,
                self.trackbar_callback,
            )

        self.recompute_geometry(strict=True)

    def metric_points_from_state(
        self,
        state: RoiState,
    ) -> np.ndarray:
        u = self.metric_plane.u_axis
        v = self.metric_plane.v_axis
        return np.asarray([
            state.u_min * u + state.v_bottom * v,
            state.u_max * u + state.v_bottom * v,
            state.u_min * u + state.v_top * v,
            state.u_max * u + state.v_top * v,
        ], dtype=np.float64)

    @staticmethod
    def snap_near_integer(value: float) -> float:
        tolerance = max(1.0e-6, abs(value) * 1.0e-6)
        nearest = float(round(value))
        return nearest if abs(value - nearest) <= tolerance else value

    def output_canvas_size(
        self,
        width_mm: float,
        height_mm: float,
        pixels_per_mm: float,
    ) -> tuple[int, int]:
        content_width = self.snap_near_integer(width_mm * pixels_per_mm)
        content_height = self.snap_near_integer(height_mm * pixels_per_mm)
        return (
            int(math.ceil(content_width)) + 1,
            int(math.ceil(content_height)) + 1,
        )

    def output_fits_limits(self, width: int, height: int) -> bool:
        return (
            width <= self.max_output_side
            and height <= self.max_output_side
            and width * height <= self.max_output_pixels
        )

    def maximum_fitting_pixels_per_mm(
        self,
        width_mm: float,
        height_mm: float,
    ) -> float:
        if (
            not np.isfinite(width_mm)
            or not np.isfinite(height_mm)
            or width_mm <= 0.0
            or height_mm <= 0.0
        ):
            return 0.0
        high = min(
            (self.max_output_side - 1) / width_mm,
            (self.max_output_side - 1) / height_mm,
        )
        if not np.isfinite(high) or high <= 0.0:
            return 0.0
        low = 0.0
        for _ in range(60):
            middle = (low + high) / 2.0
            width, height = self.output_canvas_size(
                width_mm,
                height_mm,
                middle,
            )
            if self.output_fits_limits(width, height):
                low = middle
            else:
                high = middle
        return low * 0.999999

    def compute_geometry(self, state: RoiState) -> BevGeometry:
        minimum_size = 2.0 * self.metric_plane.square_size_mm
        if (
            not np.isfinite(state.width_mm)
            or not np.isfinite(state.height_mm)
            or state.width_mm < minimum_size
            or state.height_mm < minimum_size
        ):
            raise ValueError(
                "ROI 실제 크기가 너무 작습니다: "
                f"{state.width_mm:.1f}x{state.height_mm:.1f}mm "
                f"(각 변 최소 {minimum_size:.0f}mm)"
            )
        if not np.isfinite(state.pixels_per_mm) or state.pixels_per_mm <= 0.0:
            raise ValueError("ROI pixels_per_mm이 유효하지 않습니다.")

        metric_points = self.metric_points_from_state(state)
        source_points = perspective_points(
            self.metric_plane.plane_to_image,
            metric_points,
        )
        width = self.calibration.width
        height = self.calibration.height
        if (
            np.any(source_points[:, 0] < 0.0)
            or np.any(source_points[:, 0] > width - 1)
            or np.any(source_points[:, 1] < 0.0)
            or np.any(source_points[:, 1] > height - 1)
        ):
            raise ValueError(
                "ROI가 보정 영상 밖으로 나갔습니다. "
                "검은 영역이 생기므로 저장할 수 없습니다."
            )

        left_center_x = float(np.mean(source_points[[LB, LT], 0]))
        right_center_x = float(np.mean(source_points[[RB, RT], 0]))
        bottom_center_y = float(np.mean(source_points[[LB, RB], 1]))
        top_center_y = float(np.mean(source_points[[LT, RT], 1]))
        if right_center_x - left_center_x < MIN_SOURCE_EDGE_PX:
            raise ValueError("ROI 좌우 방향이 화면에서 뒤집혔습니다.")
        if bottom_center_y - top_center_y < MIN_SOURCE_EDGE_PX:
            raise ValueError("ROI 상하 방향이 화면에서 뒤집혔습니다.")

        polygon = np.asarray([
            source_points[LT],
            source_points[RT],
            source_points[RB],
            source_points[LB],
        ], dtype=np.float32)
        if not cv2.isContourConvex(polygon):
            raise ValueError("ROI 원본 사다리꼴이 오목하거나 교차합니다.")
        area = float(abs(cv2.contourArea(polygon)))
        if area < MIN_SOURCE_AREA_PX2:
            raise ValueError(f"ROI 영상 면적이 너무 작습니다: {area:.1f}px²")

        content_width = self.snap_near_integer(
            state.width_mm * state.pixels_per_mm
        )
        content_height = self.snap_near_integer(
            state.height_mm * state.pixels_per_mm
        )
        warp_width, warp_height = self.output_canvas_size(
            state.width_mm,
            state.height_mm,
            state.pixels_per_mm,
        )
        if warp_width < MIN_OUTPUT_SIDE or warp_height < MIN_OUTPUT_SIDE:
            raise ValueError(
                "출력 해상도가 너무 작습니다: "
                f"{warp_width}x{warp_height}px"
            )
        if not self.output_fits_limits(warp_width, warp_height):
            raise ValueError(
                "출력 해상도 안전 상한 초과: "
                f"{warp_width}x{warp_height}px. "
                "px/mm를 낮추거나 F로 허용 최대값을 적용하세요."
            )
        destination_points = np.float32([
            [0.0, content_height],
            [content_width, content_height],
            [0.0, 0.0],
            [content_width, 0.0],
        ])
        if (
            float(np.max(destination_points[:, 0])) > warp_width - 1
            or float(np.max(destination_points[:, 1])) > warp_height - 1
        ):
            raise ValueError("destination point가 출력 canvas 밖입니다.")

        homography = cv2.getPerspectiveTransform(
            np.asarray(source_points, dtype=np.float32),
            destination_points,
        )
        if (
            homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or np.linalg.matrix_rank(homography) < 3
        ):
            raise ValueError("유효한 BEV homography를 계산하지 못했습니다.")
        condition = float(np.linalg.cond(homography))
        if not np.isfinite(condition) or condition > 1.0e10:
            raise ValueError(
                f"BEV homography가 불안정합니다: condition={condition:.3e}"
            )
        mapped = perspective_points(homography, source_points)
        if float(np.max(np.abs(mapped - destination_points))) > 0.05:
            raise ValueError("BEV source/destination 변환 검증 실패")

        return BevGeometry(
            metric_points_mm=metric_points,
            src_points=np.asarray(source_points, dtype=np.float32),
            dst_points=destination_points,
            homography=np.asarray(homography, dtype=np.float64),
            warp_width=warp_width,
            warp_height=warp_height,
            content_width_px=content_width,
            content_height_px=content_height,
        )

    def recompute_geometry(self, *, strict: bool = False) -> bool:
        try:
            geometry = self.compute_geometry(self.state)
        except (ValueError, cv2.error) as error:
            self.geometry_error = str(error)
            if strict:
                raise
            return False
        self.geometry = geometry
        self.geometry_error = None
        self.dirty = True
        return True

    def try_state(
        self,
        candidate: RoiState,
        *,
        message: str = "",
    ) -> bool:
        old_state = self.state
        old_geometry = self.geometry
        physical_roi_changed = any((
            candidate.u_min != old_state.u_min,
            candidate.u_max != old_state.u_max,
            candidate.v_bottom != old_state.v_bottom,
            candidate.v_top != old_state.v_top,
        ))
        if (
            physical_roi_changed
            and candidate.width_mm > 0.0
            and candidate.height_mm > 0.0
        ):
            output_width, output_height = self.output_canvas_size(
                candidate.width_mm,
                candidate.height_mm,
                candidate.pixels_per_mm,
            )
            if not self.output_fits_limits(output_width, output_height):
                fitted_scale = self.maximum_fitting_pixels_per_mm(
                    candidate.width_mm,
                    candidate.height_mm,
                )
                if 0.0 < fitted_scale < candidate.pixels_per_mm:
                    candidate = replace(
                        candidate,
                        pixels_per_mm=fitted_scale,
                    )
                    suffix = (
                        f"출력 상한에 맞춰 {fitted_scale:.6f}px/mm 자동 조정"
                    )
                    message = f"{message} | {suffix}" if message else suffix
        self.state = candidate
        if not self.recompute_geometry(strict=False):
            error = self.geometry_error or "유효하지 않은 ROI"
            self.state = old_state
            self.geometry = old_geometry
            self.geometry_error = None
            self.status_message = error
            self.dirty = True
            return False
        self.status_message = message
        return True

    def commit_state_change(
        self,
        candidate: RoiState,
        message: str,
    ) -> bool:
        previous = self.state
        if not self.try_state(candidate, message=message):
            return False
        if self.state != previous:
            self.undo_stack.append(previous)
            self.redo_stack.clear()
        return True

    def fit_pixels_per_mm(self) -> None:
        new_scale = self.maximum_fitting_pixels_per_mm(
            self.state.width_mm,
            self.state.height_mm,
        )
        if new_scale <= 0.0:
            self.status_message = "현재 ROI에 맞는 렌더링 축척이 없습니다."
            self.dirty = True
            return
        candidate = replace(self.state, pixels_per_mm=new_scale)
        self.commit_state_change(
            candidate,
            f"렌더링 축척 맞춤: {new_scale:.6f}px/mm",
        )

    def state_with_scaled_roi(
        self,
        factor: float,
    ) -> RoiState:
        half_width = self.state.width_mm * factor / 2.0
        half_height = self.state.height_mm * factor / 2.0
        return replace(
            self.state,
            u_min=self.state.center_u - half_width,
            u_max=self.state.center_u + half_width,
            v_bottom=self.state.center_v - half_height,
            v_top=self.state.center_v + half_height,
        )

    def state_with_translation(
        self,
        delta_u: float,
        delta_v: float,
    ) -> RoiState:
        return replace(
            self.state,
            u_min=self.state.u_min + delta_u,
            u_max=self.state.u_max + delta_u,
            v_bottom=self.state.v_bottom + delta_v,
            v_top=self.state.v_top + delta_v,
        )

    def image_to_uv(self, x: int, y: int) -> tuple[float, float]:
        plane_point = perspective_points(
            self.metric_plane.image_to_plane,
            np.asarray([[x, y]], dtype=np.float64),
        )[0]
        return (
            float(np.dot(plane_point, self.metric_plane.u_axis)),
            float(np.dot(plane_point, self.metric_plane.v_axis)),
        )

    def handle_positions(self) -> dict[str, tuple[float, float]]:
        if self.geometry is None:
            return {}
        points = self.geometry.src_points
        return {
            "LB": tuple(points[LB]),
            "RB": tuple(points[RB]),
            "LT": tuple(points[LT]),
            "RT": tuple(points[RT]),
            "LEFT": tuple(np.mean(points[[LB, LT]], axis=0)),
            "RIGHT": tuple(np.mean(points[[RB, RT]], axis=0)),
            "BOTTOM": tuple(np.mean(points[[LB, RB]], axis=0)),
            "TOP": tuple(np.mean(points[[LT, RT]], axis=0)),
            "CENTER": tuple(np.mean(points, axis=0)),
        }

    def nearest_handle(self, x: int, y: int) -> str | None:
        handles = self.handle_positions()
        if not handles:
            return None
        point = np.asarray([x, y], dtype=np.float64)
        candidates = [
            (
                float(np.linalg.norm(
                    point - np.asarray(handles[name])
                )),
                name,
            )
            for name in HANDLE_ORDER
        ]
        distance, name = min(candidates)
        return name if distance <= HANDLE_RADIUS_PX else None

    def point_inside_roi(self, x: int, y: int) -> bool:
        if self.geometry is None:
            return False
        polygon = np.asarray([
            self.geometry.src_points[LT],
            self.geometry.src_points[RT],
            self.geometry.src_points[RB],
            self.geometry.src_points[LB],
        ], dtype=np.float32)
        return cv2.pointPolygonTest(
            polygon,
            (float(x), float(y)),
            False,
        ) >= 0.0

    def drag_candidate(
        self,
        mode: str,
        mouse_u: float,
        mouse_v: float,
    ) -> RoiState:
        if self.drag_start_state is None or self.drag_start_uv is None:
            return self.state
        base = self.drag_start_state
        if mode == "CENTER":
            return replace(
                base,
                u_min=base.u_min + mouse_u - self.drag_start_uv[0],
                u_max=base.u_max + mouse_u - self.drag_start_uv[0],
                v_bottom=base.v_bottom + mouse_v - self.drag_start_uv[1],
                v_top=base.v_top + mouse_v - self.drag_start_uv[1],
            )

        candidate = base
        if "LEFT" == mode:
            candidate = replace(candidate, u_min=mouse_u)
        elif "RIGHT" == mode:
            candidate = replace(candidate, u_max=mouse_u)
        elif "BOTTOM" == mode:
            candidate = replace(candidate, v_bottom=mouse_v)
        elif "TOP" == mode:
            candidate = replace(candidate, v_top=mouse_v)
        elif mode == "LB":
            candidate = replace(
                candidate,
                u_min=mouse_u,
                v_bottom=mouse_v,
            )
        elif mode == "RB":
            candidate = replace(
                candidate,
                u_max=mouse_u,
                v_bottom=mouse_v,
            )
        elif mode == "LT":
            candidate = replace(
                candidate,
                u_min=mouse_u,
                v_top=mouse_v,
            )
        elif mode == "RT":
            candidate = replace(
                candidate,
                u_max=mouse_u,
                v_top=mouse_v,
            )

        if not base.aspect_locked:
            return candidate

        ratio = base.width_mm / base.height_mm
        if mode in ("LEFT", "RIGHT"):
            width = candidate.width_mm
            height = width / ratio
            return replace(
                candidate,
                v_bottom=base.center_v - height / 2.0,
                v_top=base.center_v + height / 2.0,
            )
        if mode in ("BOTTOM", "TOP"):
            height = candidate.height_mm
            width = height * ratio
            return replace(
                candidate,
                u_min=base.center_u - width / 2.0,
                u_max=base.center_u + width / 2.0,
            )

        opposite = {
            "LB": (base.u_max, base.v_top, -1.0, -1.0),
            "RB": (base.u_min, base.v_top, 1.0, -1.0),
            "LT": (base.u_max, base.v_bottom, -1.0, 1.0),
            "RT": (base.u_min, base.v_bottom, 1.0, 1.0),
        }
        if mode in opposite:
            anchor_u, anchor_v, sign_u, sign_v = opposite[mode]
            raw_width = sign_u * (mouse_u - anchor_u)
            raw_height = sign_v * (mouse_v - anchor_v)
            width = (
                raw_width + raw_height / ratio
            ) / (1.0 + 1.0 / (ratio * ratio))
            height = width / ratio
            dragged_u = anchor_u + sign_u * width
            dragged_v = anchor_v + sign_v * height
            values = {
                "u_min": min(anchor_u, dragged_u),
                "u_max": max(anchor_u, dragged_u),
                "v_bottom": min(anchor_v, dragged_v),
                "v_top": max(anchor_v, dragged_v),
            }
            return replace(base, **values)
        return candidate

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        del param
        x = int(np.clip(x, 0, self.calibration.width - 1))
        y = int(np.clip(y, 0, self.calibration.height - 1))

        if event == cv2.EVENT_MOUSEWHEEL:
            if self.drag_mode is not None:
                self.status_message = (
                    "현재 드래그를 놓거나 우클릭으로 취소한 뒤 휠을 사용하세요."
                )
                self.dirty = True
                return
            delta = mouse_wheel_delta(flags)
            if delta == 0:
                return
            factor = 1.05 if delta > 0 else 1.0 / 1.05
            self.commit_state_change(
                self.state_with_scaled_roi(factor),
                "ROI 크기 조절",
            )
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            if self.drag_start_state is not None:
                self.state = self.drag_start_state
                self.recompute_geometry(strict=False)
            self.drag_mode = None
            self.drag_start_uv = None
            self.drag_start_state = None
            self.drag_changed = False
            self.status_message = "드래그 취소"
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            mode = self.nearest_handle(x, y)
            if mode is None and self.point_inside_roi(x, y):
                mode = "CENTER"
            if mode is None:
                self.status_message = "ROI 핸들이나 내부를 누르세요."
                self.dirty = True
                return
            try:
                uv = self.image_to_uv(x, y)
            except (ValueError, cv2.error) as error:
                self.status_message = str(error)
                self.dirty = True
                return
            self.playing = False
            self.drag_mode = mode
            self.drag_start_uv = uv
            self.drag_start_state = self.state
            self.drag_changed = False
            return

        if (
            event == cv2.EVENT_MOUSEMOVE
            and self.drag_mode is not None
            and (flags & cv2.EVENT_FLAG_LBUTTON)
        ):
            try:
                mouse_u, mouse_v = self.image_to_uv(x, y)
                candidate = self.drag_candidate(
                    self.drag_mode,
                    mouse_u,
                    mouse_v,
                )
                if self.try_state(candidate, message="ROI 편집 중"):
                    self.drag_changed = True
            except (ValueError, cv2.error) as error:
                self.status_message = str(error)
                self.dirty = True
            return

        if event == cv2.EVENT_LBUTTONUP and self.drag_mode is not None:
            if (
                self.drag_changed
                and self.drag_start_state is not None
                and self.state != self.drag_start_state
            ):
                self.undo_stack.append(self.drag_start_state)
                self.redo_stack.clear()
                self.status_message = "ROI 편집 확정"
            self.drag_mode = None
            self.drag_start_uv = None
            self.drag_start_state = None
            self.drag_changed = False

    def trackbar_callback(self, value: int) -> None:
        if self.updating_trackbar:
            return
        self.playing = False
        self.requested_index = int(value)

    def set_requested_frame(
        self,
        index: int,
        *,
        pause: bool = True,
    ) -> None:
        self.requested_index = int(np.clip(
            index,
            0,
            len(self.frame_source) - 1,
        ))
        if pause:
            self.playing = False

    def load_requested_frame(self) -> None:
        if self.current_index == self.requested_index:
            return
        raw, timestamp, encoding, frame_id = self.frame_source.read_frame(
            self.requested_index
        )
        if (
            raw.shape[1] != self.calibration.width
            or raw.shape[0] != self.calibration.height
        ):
            raise ValueError(
                "bag 영상과 캘리브레이션 해상도가 다릅니다. "
                "리사이즈하면 metric 평면이 깨지므로 중단합니다: "
                f"bag={raw.shape[1]}x{raw.shape[0]}, "
                f"calibration={self.calibration.width}x"
                f"{self.calibration.height}"
            )
        self.current_raw_frame = raw
        self.current_frame = cv2.remap(
            raw,
            self.calibration.map_x,
            self.calibration.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.current_index = self.requested_index
        self.current_timestamp = timestamp
        self.current_encoding = encoding
        self.current_frame_id = frame_id
        if self.has_frame_trackbar:
            self.updating_trackbar = True
            cv2.setTrackbarPos("Frame", SOURCE_WINDOW, self.current_index)
            self.updating_trackbar = False
        self.dirty = True

    def draw_source_overlay(self) -> np.ndarray:
        if self.current_frame is None or self.geometry is None:
            raise RuntimeError("표시할 source frame 또는 geometry가 없습니다.")
        display = self.current_frame.copy()
        points = np.rint(self.geometry.src_points).astype(np.int32)
        polygon = np.asarray([
            points[LT],
            points[RT],
            points[RB],
            points[LB],
        ])
        fill = display.copy()
        cv2.fillConvexPoly(fill, polygon, (0, 180, 0))
        display = cv2.addWeighted(fill, 0.18, display, 0.82, 0.0)
        cv2.polylines(
            display,
            [polygon],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        handles = self.handle_positions()
        for name in ("LB", "RB", "LT", "RT"):
            center = tuple(np.rint(handles[name]).astype(int))
            cv2.circle(display, center, 7, (0, 255, 255), -1)
            cv2.putText(
                display,
                name,
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for name in ("LEFT", "RIGHT", "BOTTOM", "TOP"):
            center = tuple(np.rint(handles[name]).astype(int))
            cv2.rectangle(
                display,
                (center[0] - 5, center[1] - 5),
                (center[0] + 5, center[1] + 5),
                (255, 180, 0),
                -1,
            )
        center = tuple(np.rint(handles["CENTER"]).astype(int))
        cv2.drawMarker(
            display,
            center,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )

        relative_seconds = (
            self.current_timestamp - self.frame_source.timestamps[0]
        ) / 1.0e9
        geometry = self.geometry
        lines = [
            (
                f"Frame {self.current_index + 1}/{len(self.frame_source)} "
                f"| t={relative_seconds:.3f}s "
                f"| {'PLAY' if self.playing else 'PAUSE'}"
            ),
            (
                f"ROI {self.state.width_mm / 1000.0:.3f} x "
                f"{self.state.height_mm / 1000.0:.3f} m "
                f"| W:H={self.state.width_mm / self.state.height_mm:.4f}"
            ),
            (
                f"BEV {geometry.warp_width}x{geometry.warp_height}px "
                f"| {self.state.pixels_per_mm:.6f}px/mm "
                f"| scale X=Y"
            ),
            (
                f"Aspect lock: {'ON' if self.state.aspect_locked else 'OFF'} "
                "| Drag corners/edges; drag inside to move"
            ),
        ]
        overlay = display.copy()
        panel_height = 26 * len(lines) + 8
        cv2.rectangle(
            overlay,
            (0, 0),
            (display.shape[1] - 1, panel_height),
            (0, 0, 0),
            -1,
        )
        display = cv2.addWeighted(overlay, 0.62, display, 0.38, 0.0)
        for index, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (10, 23 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        bottom_lines = [
            "SPACE play | J/L frame | U/O 1sec | arrows move | wheel ROI size",
            "H aspect | +/- px/mm | F fit | Z/Y undo/redo | R reset | S save | Q quit",
        ]
        overlay = display.copy()
        y0 = display.shape[0] - 48
        cv2.rectangle(
            overlay,
            (0, y0),
            (display.shape[1] - 1, display.shape[0] - 1),
            (0, 0, 0),
            -1,
        )
        display = cv2.addWeighted(overlay, 0.65, display, 0.35, 0.0)
        for index, text in enumerate(bottom_lines):
            cv2.putText(
                display,
                text,
                (8, y0 + 19 + index * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.status_message:
            cv2.putText(
                display,
                self.status_message[:90],
                (10, panel_height + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return display

    def preview_layout(
        self,
    ) -> tuple[np.ndarray, int, int, float]:
        if self.geometry is None:
            raise RuntimeError("BEV preview geometry가 없습니다.")
        geometry = self.geometry
        scale = min(
            1.0,
            (MAX_BEV_PREVIEW_WIDTH - 1) / geometry.content_width_px,
            (MAX_BEV_PREVIEW_HEIGHT - 1) / geometry.content_height_px,
        )
        if scale < 1.0:
            scale *= 0.999999
        preview_content_width = self.snap_near_integer(
            geometry.content_width_px * scale
        )
        preview_content_height = self.snap_near_integer(
            geometry.content_height_px * scale
        )
        preview_width = int(math.ceil(preview_content_width)) + 1
        preview_height = int(math.ceil(preview_content_height)) + 1
        scale_matrix = np.asarray([
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return (
            scale_matrix @ geometry.homography,
            preview_width,
            preview_height,
            scale,
        )

    def draw_bev_overlay(
        self,
        bev: np.ndarray,
        preview_scale: float,
    ) -> np.ndarray:
        display = bev.copy()
        step_px = (
            GRID_INTERVAL_MM
            * self.state.pixels_per_mm
            * preview_scale
        )
        if step_px >= 8.0:
            x = 0.0
            preview_content_width = (
                self.geometry.content_width_px * preview_scale
            )
            preview_content_height = (
                self.geometry.content_height_px * preview_scale
            )
            while x <= preview_content_width + 0.5:
                pixel = int(round(x))
                cv2.line(
                    display,
                    (pixel, 0),
                    (pixel, display.shape[0] - 1),
                    (80, 80, 80),
                    1,
                )
                x += step_px
            y = 0.0
            while y <= preview_content_height + 0.5:
                pixel = int(round(y))
                cv2.line(
                    display,
                    (0, pixel),
                    (display.shape[1] - 1, pixel),
                    (80, 80, 80),
                    1,
                )
                y += step_px
        lines = [
            (
                f"REAL SCALE X=Y | grid {GRID_INTERVAL_MM / 1000.0:g}m "
                f"| {self.state.pixels_per_mm:.6f}px/mm"
            ),
            (
                f"physical {self.state.width_mm / 1000.0:.3f} x "
                f"{self.state.height_mm / 1000.0:.3f}m "
                f"| save {self.geometry.warp_width}x"
                f"{self.geometry.warp_height} "
                f"| preview {bev.shape[1]}x{bev.shape[0]}"
            ),
        ]
        overlay = display.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (display.shape[1] - 1, 58),
            (0, 0, 0),
            -1,
        )
        display = cv2.addWeighted(overlay, 0.62, display, 0.38, 0.0)
        for index, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (10, 23 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return display

    def show_bev(self, image: np.ndarray) -> None:
        if self.bev_window_open:
            try:
                if cv2.getWindowProperty(
                    BEV_WINDOW,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    self.bev_window_open = False
                    self.bev_window_shape = None
            except cv2.error:
                self.bev_window_open = False
                self.bev_window_shape = None
        if not self.bev_window_open:
            cv2.namedWindow(
                BEV_WINDOW,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            self.bev_window_open = True
        height, width = image.shape[:2]
        if self.bev_window_shape != (height, width):
            scale = min(
                1.0,
                MAX_BEV_PREVIEW_WIDTH / width,
                MAX_BEV_PREVIEW_HEIGHT / height,
            )
            cv2.resizeWindow(
                BEV_WINDOW,
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            self.bev_window_shape = (height, width)
        cv2.imshow(BEV_WINDOW, image)

    def render(self) -> None:
        self.load_requested_frame()
        if self.current_frame is None or self.geometry is None:
            raise RuntimeError("렌더링할 프레임 또는 geometry가 없습니다.")
        (
            preview_homography,
            preview_width,
            preview_height,
            preview_scale,
        ) = self.preview_layout()
        self.current_bev = cv2.warpPerspective(
            self.current_frame,
            preview_homography,
            (preview_width, preview_height),
            flags=cv2.INTER_LINEAR,
        )
        cv2.imshow(SOURCE_WINDOW, self.draw_source_overlay())
        self.show_bev(
            self.draw_bev_overlay(
                self.current_bev,
                preview_scale,
            )
        )
        self.dirty = False

    def reset_play_deadline(self) -> None:
        if self.current_index >= len(self.frame_source) - 1:
            self.playing = False
            return
        delta = (
            self.frame_source.timestamps[self.current_index + 1]
            - self.frame_source.timestamps[self.current_index]
        ) / 1.0e9
        self.next_play_deadline = (
            time.monotonic()
            + max(0.001, delta / self.playback_rate)
        )

    def update_playback(self) -> None:
        if not self.playing:
            return
        if self.current_index < 0:
            return
        now = time.monotonic()
        if now < self.next_play_deadline:
            return
        if self.current_index >= len(self.frame_source) - 1:
            self.playing = False
            return
        self.set_requested_frame(self.current_index + 1, pause=False)
        self.load_requested_frame()
        self.reset_play_deadline()

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_message = "되돌릴 ROI 변경이 없습니다."
            return
        previous = self.undo_stack.pop()
        self.redo_stack.append(self.state)
        self.state = previous
        self.recompute_geometry(strict=True)
        self.status_message = "ROI 변경 되돌림"

    def redo(self) -> None:
        if not self.redo_stack:
            self.status_message = "다시 적용할 ROI 변경이 없습니다."
            return
        next_state = self.redo_stack.pop()
        self.undo_stack.append(self.state)
        self.state = next_state
        self.recompute_geometry(strict=True)
        self.status_message = "ROI 변경 다시 적용"

    def save(self) -> None:
        if self.current_frame is None:
            raise RuntimeError("저장할 bag 프레임이 없습니다.")
        geometry = self.compute_geometry(self.state)
        pure_bev = cv2.warpPerspective(
            self.current_frame,
            geometry.homography,
            (geometry.warp_width, geometry.warp_height),
            flags=cv2.INTER_LINEAR,
        )
        source_overlay = self.draw_source_overlay()
        report = {
            "selection_mode": "metric_plane_rosbag_roi_editor",
            "source_metric_bev": str(self.metric_plane.source_path),
            "calibration_file": str(self.arguments.calibration),
            "calibration_sha256": self.calibration.sha256,
            "bag_uri": str(self.frame_source.bag_path),
            "bag_topic": self.frame_source.topic,
            "bag_message_type": self.frame_source.type_name,
            "bag_frame_index": self.current_index,
            "bag_frame_count": len(self.frame_source),
            "bag_timestamp_ns": self.current_timestamp,
            "camera_frame_id": self.current_frame_id,
            "camera_encoding": self.current_encoding,
            "physical_width_mm": self.state.width_mm,
            "physical_height_mm": self.state.height_mm,
            "physical_aspect_ratio": (
                self.state.width_mm / self.state.height_mm
            ),
            "pixels_per_mm": self.state.pixels_per_mm,
            "warp_width": geometry.warp_width,
            "warp_height": geometry.warp_height,
            "src_points_order": ["LB", "RB", "LT", "RT"],
            "src_points": geometry.src_points.tolist(),
            "dst_points": geometry.dst_points.tolist(),
            "effective_metric_points_mm": (
                geometry.metric_points_mm.tolist()
            ),
            "roi_u_axis": self.metric_plane.u_axis.tolist(),
            "roi_v_axis": self.metric_plane.v_axis.tolist(),
            "roi_bounds_mm": {
                "u_min": self.state.u_min,
                "u_max": self.state.u_max,
                "v_bottom": self.state.v_bottom,
                "v_top": self.state.v_top,
            },
            "aspect_locked_during_last_state": self.state.aspect_locked,
        }

        arrays: dict[str, object] = {
            "src_points": np.asarray(
                geometry.src_points,
                dtype=np.float32,
            ),
            "selected_src_points": np.asarray(
                geometry.src_points,
                dtype=np.float32,
            ),
            "dst_points": np.asarray(
                geometry.dst_points,
                dtype=np.float32,
            ),
            "homography": np.asarray(
                geometry.homography,
                dtype=np.float64,
            ),
            "warp_width": np.int32(geometry.warp_width),
            "warp_height": np.int32(geometry.warp_height),
            "calibration_width": np.int32(self.calibration.width),
            "calibration_height": np.int32(self.calibration.height),
            "square_size_mm": np.float64(
                self.metric_plane.square_size_mm
            ),
            "physical_width_mm": np.float64(self.state.width_mm),
            "physical_height_mm": np.float64(self.state.height_mm),
            "pixels_per_mm": np.float64(self.state.pixels_per_mm),
            "selected_metric_points_mm": np.asarray(
                geometry.metric_points_mm,
                dtype=np.float64,
            ),
            "effective_metric_points_mm": np.asarray(
                geometry.metric_points_mm,
                dtype=np.float64,
            ),
            "plane_to_image_homography": np.asarray(
                self.metric_plane.plane_to_image,
                dtype=np.float64,
            ),
            "image_to_plane_homography": np.asarray(
                self.metric_plane.image_to_plane,
                dtype=np.float64,
            ),
            "roi_u_axis": np.asarray(
                self.metric_plane.u_axis,
                dtype=np.float64,
            ),
            "roi_v_axis": np.asarray(
                self.metric_plane.v_axis,
                dtype=np.float64,
            ),
            "roi_bounds_mm": np.asarray([
                self.state.u_min,
                self.state.u_max,
                self.state.v_bottom,
                self.state.v_top,
            ], dtype=np.float64),
            "calibration_sha256": np.asarray(
                self.calibration.sha256
            ),
            "source_metric_bev": np.asarray(
                str(self.metric_plane.source_path)
            ),
            "input_bag": np.asarray(str(self.frame_source.bag_path)),
            "input_topic": np.asarray(self.frame_source.topic),
            "input_transport": np.asarray(
                "compressed"
                if self.frame_source.type_name.endswith("CompressedImage")
                else "raw"
            ),
            "camera_frame_id": np.asarray(self.current_frame_id),
            "camera_encoding": np.asarray(self.current_encoding),
            "bag_frame_index": np.int64(self.current_index),
            "bag_timestamp_ns": np.int64(self.current_timestamp),
            "coordinate_space": np.asarray(
                "full_undistorted_camera_image"
            ),
            "selection_mode": np.asarray(
                "metric_plane_rosbag_roi_editor"
            ),
        }
        for key in (
            "checkerboard_size",
            "plane_reprojection_rms_px",
            "plane_inlier_ratio",
            "source_checker_mean_spacing_px",
            "source_checker_area_px2",
        ):
            if key in self.metric_plane.source_arrays:
                arrays[key] = np.asarray(
                    self.metric_plane.source_arrays[key]
                ).copy()

        stem = self.output_path.stem
        output_dir = self.output_path.parent
        preview_path = output_dir / f"{stem}_preview.jpg"
        source_path = output_dir / f"{stem}_source_roi.jpg"
        report_path = output_dir / f"{stem}_report.json"
        text_path = output_dir / f"{stem}_points.txt"
        point_lines = [
            "# point order: LB, RB, LT, RT",
            "# coordinate: full undistorted camera image",
            (
                f"# physical: {self.state.width_mm:.6f} x "
                f"{self.state.height_mm:.6f} mm"
            ),
            (
                f"# output: {geometry.warp_width} x "
                f"{geometry.warp_height} px"
            ),
            f"# pixels_per_mm: {self.state.pixels_per_mm:.9f}",
            "",
        ]
        for name, point in zip(POINT_NAMES, geometry.src_points):
            point_lines.append(
                f"{float(point[0]):.6f}, "
                f"{float(point[1]):.6f}  # {name}"
            )

        atomic_write_image(preview_path, pure_bev)
        atomic_write_image(source_path, source_overlay)
        atomic_write_text(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        )
        atomic_write_text(text_path, "\n".join(point_lines) + "\n")
        backup = atomic_write_npz(self.output_path, arrays)

        print()
        print("=" * 72)
        print("ROS bag 도로 ROI BEV 저장 완료")
        print("=" * 72)
        print(f"BEV NPZ       : {self.output_path}")
        if backup is not None:
            print(f"이전 파일 백업: {backup}")
        print(f"순수 BEV      : {preview_path}")
        print(f"원본 ROI      : {source_path}")
        print(f"보고서        : {report_path}")
        print(
            "실제 크기      : "
            f"{self.state.width_mm / 1000.0:.3f} x "
            f"{self.state.height_mm / 1000.0:.3f} m"
        )
        print(
            "출력 크기      : "
            f"{geometry.warp_width} x {geometry.warp_height}px"
        )
        print(f"축척 X=Y      : {self.state.pixels_per_mm:.6f}px/mm")
        print("=" * 72)
        self.status_message = f"저장 완료: {self.output_path.name}"

    def handle_key(self, key: int) -> bool:
        if key < 0:
            return True
        if self.drag_mode is not None:
            ascii_key = key & 0xFF
            arrow_codes = (
                LEFT_KEY_CODES
                | UP_KEY_CODES
                | RIGHT_KEY_CODES
                | DOWN_KEY_CODES
            )
            if (
                key not in arrow_codes
                and ascii_key in (27, ord("q"), ord("Q"))
            ):
                return False
            self.status_message = (
                "현재 드래그를 놓거나 우클릭으로 취소한 뒤 키를 사용하세요."
            )
            self.dirty = True
            return True
        # Ubuntu Qt/X11의 방향키(65361~65364)는 하위 8비트가
        # Q/R/S/T와 겹친다. ASCII로 자르기 전에 확장 키를 먼저 처리한다.
        if key in LEFT_KEY_CODES:
            self.commit_state_change(
                self.state_with_translation(
                    -self.metric_plane.square_size_mm / 5.0,
                    0.0,
                ),
                "ROI 왼쪽 미세 이동",
            )
            self.dirty = True
            return True
        if key in RIGHT_KEY_CODES:
            self.commit_state_change(
                self.state_with_translation(
                    self.metric_plane.square_size_mm / 5.0,
                    0.0,
                ),
                "ROI 오른쪽 미세 이동",
            )
            self.dirty = True
            return True
        if key in UP_KEY_CODES:
            self.commit_state_change(
                self.state_with_translation(
                    0.0,
                    self.metric_plane.square_size_mm / 5.0,
                ),
                "ROI 위쪽 미세 이동",
            )
            self.dirty = True
            return True
        if key in DOWN_KEY_CODES:
            self.commit_state_change(
                self.state_with_translation(
                    0.0,
                    -self.metric_plane.square_size_mm / 5.0,
                ),
                "ROI 아래쪽 미세 이동",
            )
            self.dirty = True
            return True

        ascii_key = key & 0xFF
        if ascii_key in (27, ord("q"), ord("Q")):
            return False
        if ascii_key == 32:
            self.playing = not self.playing
            if self.playing:
                self.reset_play_deadline()
            self.status_message = "재생" if self.playing else "일시정지"
            self.dirty = True
            return True
        if ascii_key in (ord("j"), ord("J")):
            self.set_requested_frame(self.current_index - 1)
        elif ascii_key in (ord("l"), ord("L")):
            self.set_requested_frame(self.current_index + 1)
        elif ascii_key in (ord("u"), ord("U")):
            self.set_requested_frame(
                self.frame_source.index_near_time(
                    self.current_timestamp - 1_000_000_000
                )
            )
        elif ascii_key in (ord("o"), ord("O")):
            self.set_requested_frame(
                self.frame_source.index_near_time(
                    self.current_timestamp + 1_000_000_000
                )
            )
        elif ascii_key in (ord("h"), ord("H")):
            self.state = replace(
                self.state,
                aspect_locked=not self.state.aspect_locked,
            )
            self.status_message = (
                "W:H 잠금 ON"
                if self.state.aspect_locked
                else "W:H 잠금 OFF"
            )
            self.dirty = True
        elif ascii_key in (ord("z"), ord("Z")):
            self.undo()
        elif ascii_key in (ord("y"), ord("Y")):
            self.redo()
        elif ascii_key in (ord("r"), ord("R")):
            self.commit_state_change(
                replace(
                    self.seed_state,
                    aspect_locked=self.state.aspect_locked,
                ),
                "초기 metric ROI 복원",
            )
        elif ascii_key in (ord("s"), ord("S")):
            try:
                self.save()
            except (ValueError, RuntimeError, OSError, cv2.error) as error:
                self.status_message = f"저장 실패: {error}"
                print(f"[저장 실패] {error}")
        elif ascii_key in (ord("f"), ord("F")):
            self.fit_pixels_per_mm()
        elif ascii_key in (ord("-"), ord("_")):
            self.commit_state_change(
                replace(
                    self.state,
                    pixels_per_mm=self.state.pixels_per_mm / 1.1,
                ),
                "렌더링 px/mm 감소",
            )
        elif ascii_key in (ord("="), ord("+")):
            self.commit_state_change(
                replace(
                    self.state,
                    pixels_per_mm=self.state.pixels_per_mm * 1.1,
                ),
                "렌더링 px/mm 증가",
            )
        elif ascii_key == ord(","):
            self.commit_state_change(
                self.state_with_scaled_roi(1.0 / 1.05),
                "ROI 축소",
            )
        elif ascii_key == ord("."):
            self.commit_state_change(
                self.state_with_scaled_roi(1.05),
                "ROI 확대",
            )
        self.dirty = True
        return True

    def print_summary(self) -> None:
        print("=" * 78)
        print("ROS bag 실제 비율 BEV 도로 ROI 편집기")
        print("=" * 78)
        print(f"bag                 : {self.frame_source.bag_path}")
        print(f"영상 토픽           : {self.frame_source.topic}")
        print(f"메시지 타입         : {self.frame_source.type_name}")
        print(f"프레임 수           : {len(self.frame_source)}")
        print(
            f"캘리브레이션        : {self.arguments.calibration} "
            f"({self.calibration.width}x{self.calibration.height})"
        )
        print(f"metric 평면         : {self.metric_plane.source_path}")
        if "calibration_sha256" not in self.metric_plane.source_arrays:
            print(
                "[주의] 입력 metric NPZ에는 calibration hash가 없습니다. "
                "반드시 이 평면을 만든 동일 calibration 파일을 사용하세요."
            )
        print(f"출력                : {self.output_path}")
        print(
            "초기 실제 ROI       : "
            f"{self.seed_state.width_mm / 1000.0:.3f} x "
            f"{self.seed_state.height_mm / 1000.0:.3f} m"
        )
        print(f"렌더링 축척 X=Y     : {self.state.pixels_per_mm:.6f}px/mm")
        print()
        print("마우스")
        print("- 모서리 원 또는 변 사각형 드래그: ROI 경계 조절")
        print("- ROI 내부/가운데 십자 드래그      : ROI 전체 이동")
        print("- 휠                                : ROI 전체 확대/축소")
        print("- 우클릭                            : 현재 드래그 취소")
        print()
        print("키보드")
        print("SPACE 재생/정지 | J/L 한 프레임 | U/O 1초 | 트랙바 탐색")
        print("방향키 ROI 미세 이동 | H W:H 잠금 | ,/. ROI 축소/확대")
        print("-/+ 렌더링 축척 | F 출력 상한에 맞춤 | Z/Y 되돌리기/재실행")
        print("R 초기 ROI | S 저장 | Q/ESC 종료")
        print("=" * 78)

    def run(self) -> None:
        self.print_summary()
        try:
            while True:
                self.load_requested_frame()
                self.update_playback()
                if self.dirty:
                    self.render()
                key = cv2.waitKeyEx(10)
                if not self.handle_key(key):
                    break
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
        editor = MetricRosbagRoiEditor(arguments)
        editor.run()
        return 0
    except KeyboardInterrupt:
        print("\n[중단] 사용자가 편집을 종료했습니다.")
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
        print("ROS bag metric ROI 편집기를 시작하지 못했습니다.")
        print("=" * 72)
        print(error)
        print("=" * 72)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
