#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS usb_cam 실시간 BEV 설정 + 25 mm 체커보드 품질 검사 도구.

준비:
    1. Logitech C920은 보통 /dev/video2를 사용한다.
    2. usb_cam이 /camera/lane/raw/compressed 토픽을 발행해야 한다.
    3. 내부 코너 10x7, 한 칸 25 mm 체커보드를 평평한 바닥에 놓는다.

실행 - 터미널 1 (C920 카메라 켜기):
    cd /home/hanjingyu/dolbatS
    source /opt/ros/humble/setup.bash
    ros2 run usb_cam usb_cam_node_exe --ros-args \\
      -p video_device:=/dev/video2 \\
      -p image_width:=640 -p image_height:=480 -p framerate:=30.0 \\
      -p io_method:=mmap -p pixel_format:=mjpeg2rgb \\
      -p camera_name:=lane_camera -p frame_id:=lane_camera \\
      -r image_raw:=/camera/lane/raw \\
      -r image_raw/compressed:=/camera/lane/raw/compressed \\
      -r camera_info:=/camera/lane/camera_info

실행 - 터미널 2 (BEV 설정 도구):
    cd /home/hanjingyu/dolbatS
    source /opt/ros/humble/setup.bash
    python3 utils/webcam_bev_drag_checkerboard_live.py \\
      --topic /camera/lane/raw/compressed \\
      --transport compressed

사용 순서:
    1. 원본 실시간 화면에서 체커보드 전체가 보이게 배치한다.
    2. SPACE를 눌러 사진을 찍듯 현재 화면을 정지한다.
    3. LB(좌하), RB(우하), LT(좌상), RT(우상) 순서로 클릭한다.
       RB의 y는 LB에, RT의 y는 LT에 자동으로 맞춰진다.
       원본 영상 바깥의 회색 여백도 클릭할 수 있으며 좌표는 음수 또는
       640x480 범위 밖 값으로 저장되어 확장 BEV에 사용된다.
    4. 네 점이 완성되면 정지 선택 창과 정지 BEV 체커보드 검사 창이
       유지되고, 최신 원본 카메라와 실시간 BEV 창이 별도로 표시된다.
    5. X/Y 비율은 1.0, 간격 CV와 직각 오차는 0에 가까울수록 좋다.
    6. QUALITY PASS를 확인한 뒤 S를 눌러 NPZ와 보고서를 저장한다.

BEV 검사 화면의 글자:
    Checkerboard DETECTED
        10x7 내부 코너 검출 성공 여부다.
    Physical square
        캘리브레이션에 저장된 실제 한 칸 크기이며 반드시 25.0 mm여야 한다.
    X/Y px/square
        BEV에서 체커보드 한 칸의 평균 가로/세로 픽셀 길이다.
    X/Y CV
        칸 간격의 변동계수다. 0%에 가까울수록 모든 칸 간격이 균일하다.
    X/Y square difference
        평균 가로와 세로 길이 차이다. 0%이면 한 칸이 정확한 정사각형이다.
    Grid orthogonality error
        가로·세로 격자가 90도에서 벗어난 각도다. 0도에 가까울수록 좋다.

QUALITY PASS 기준:
    가로/세로 한 칸 크기 차이 <= 2%
    가로 간격 CV <= 5%
    세로 간격 CV <= 5%
    격자 직각 오차 <= 2도
    metric BEV를 계산한 경우 목표 25 mm 배율 오차 <= 3%

조정 요령:
    가로/세로 크기 차이가 크면 ROI의 상·하단 폭 또는 높이를 조절한다.
    간격 CV가 크면 체커보드가 휘었거나 바닥과 다른 평면에 있는지 확인한다.
    직각 오차가 크면 좌우 점의 x를 조절해 가로·세로 격자를 직각으로 맞춘다.
    체커보드는 주행 바닥과 같은 평면에 완전히 밀착해야 한다.

키/마우스:
    SPACE       화면 정지 또는 실시간 재개
    왼쪽 클릭  새 점 선택 또는 기존 점 드래그
    오른쪽 클릭 마지막 점 취소
    R           네 점 초기화
    S           강한 품질 재검사 후 저장
    Q / ESC     종료

기본 저장 결과:
    utils/bev_params_y_auto.npz
        주행용 bev_params_0731.npz와 동일하게 src_points, dst_points,
        warp_w, warp_h 네 항목만 저장한다. 상세 측정값은 JSON/CSV에 저장한다.
    utils/bev_preview_y_auto.jpg
    utils/bev_checkerboard_analysis.jpg
    utils/bev_checkerboard_report.json
    utils/bev_checkerboard_spacing.csv

주의:
    확장 캔버스는 영상 밖 좌표로 homography를 외삽하기 위한 기능이다.
    카메라가 실제로 촬영하지 않은 영역의 픽셀을 복원하지는 못하므로 선택
    영역이 영상 밖으로 크게 나가면 BEV 가장자리가 검게 보일 수 있다.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node as RosNode
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import CompressedImage, Image
except ImportError as ros_import_error:
    rclpy = None
    RosNode = object
    CompressedImage = object
    Image = object
    ROS_IMPORT_ERROR: ImportError | None = ros_import_error
else:
    ROS_IMPORT_ERROR = None


# ============================================================
# 사용자 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# usb_cam이 발행하는 영상 토픽. raw Image는 DDS 부하가 커질 수 있으므로
# 640x480에서도 압축 토픽을 기본값으로 사용한다.
DEFAULT_RAW_TOPIC = "/camera/lane/raw"
DEFAULT_COMPRESSED_TOPIC = "/camera/lane/raw/compressed"
DEFAULT_TRANSPORT = "compressed"
DEFAULT_TOPIC_TIMEOUT = 15.0
EXPECTED_IMAGE_WIDTH = 640
EXPECTED_IMAGE_HEIGHT = 480

# 사용자가 프로젝트 루트에 올린 640x480 카메라 캘리브레이션
DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "camera_calibration.npz"

# BEV는 정사각형으로 강제하지 않는다. 체커보드의 실제 치수로 동일한
# px/mm 배율을 적용하고, 메모리/처리량 보호를 위해서만 상한을 둔다.
DEFAULT_TARGET_SQUARE_PIXELS = 32.0
DEFAULT_MAX_OUTPUT_SIDE = 1280
DEFAULT_MAX_OUTPUT_PIXELS = 1_500_000
MIN_OUTPUT_SIDE = 96
MAX_BEV_PREVIEW_WIDTH = 900
MAX_BEV_PREVIEW_HEIGHT = 720
DIRECT_PREVIEW_WIDTH = 640
DIRECT_PREVIEW_HEIGHT = 640
SELECTION_CANVAS_WIDTH = 1000
SELECTION_CANVAS_HEIGHT = 800

# 현재 프로젝트가 사용하는 체커보드 내부 코너 수.
# 실제 값은 캘리브레이션 NPZ에서도 읽어 일치하는지 검증한다.
EXPECTED_CHECKERBOARD_SIZE = (10, 7)
EXPECTED_SQUARE_SIZE_MM = 25.0

# 드래그 중 체커보드 검사 갱신 간격
# 작을수록 자주 검사하지만 화면이 느려질 수 있음
DETECTION_INTERVAL_SEC = 0.12

# 원본 카메라 화면에 표시할 격자 간격
GRID_STEP_X = 40
GRID_STEP_Y = 40

# 캘리브레이션과 같은 프로젝트 루트에 새 BEV 설정을 저장한다.
# 검증이 끝난 뒤 lane_vision_pkg config에 두 640x480 파일을 함께 반영한다.
DEFAULT_OUTPUT_NPZ = PROJECT_ROOT / "bev_params_y_auto.npz"

# 사람이 확인하는 보조 결과는 프로젝트 루트에 저장한다.
OUTPUT_TXT = PROJECT_ROOT / "selected_bev_points_y_auto.txt"

# 다른 코드에서 그대로 사용할 순수 BEV 이미지
OUTPUT_PREVIEW = PROJECT_ROOT / "bev_preview_y_auto.jpg"

# 체커보드 검출 결과가 표시된 BEV 이미지
OUTPUT_ANALYSIS_IMAGE = PROJECT_ROOT / "bev_checkerboard_analysis.jpg"

# 측정 결과
OUTPUT_REPORT_JSON = PROJECT_ROOT / "bev_checkerboard_report.json"
OUTPUT_SPACING_CSV = PROJECT_ROOT / "bev_checkerboard_spacing.csv"

POINT_PICK_RADIUS_PX = 16
MIN_EDGE_LENGTH_PX = 10.0
MIN_ROI_WIDTH_PX = 20
MIN_ROI_AREA_PX2 = 1000.0

PLANE_RANSAC_THRESHOLD_PX = 2.5
MIN_PLANE_INLIER_RATIO = 0.90
MAX_PLANE_REPROJECTION_RMS_PX = 1.5
MIN_SOURCE_CHECKER_SPACING_PX = 8.0
MIN_SOURCE_CHECKER_AREA_PX2 = 4000.0
MAX_AXIS_PARALLEL_DOT = 0.92
MAX_RAW_ORTHOGONALITY_ERROR_DEG = 20.0
MAX_OPPOSITE_EDGE_ANGLE_DEG = 20.0
MAX_OPPOSITE_EDGE_LENGTH_ERROR = 0.35
MAX_SNAP_ERROR_SQUARES = 1.5
WARN_SNAP_ERROR_SQUARES = 0.5

MAX_FINAL_ASPECT_ERROR_PERCENT = 2.0
MAX_FINAL_SPACING_CV_PERCENT = 5.0
MAX_FINAL_ORTHOGONALITY_ERROR_DEG = 2.0
MAX_FINAL_SCALE_ERROR_PERCENT = 3.0


@dataclass
class PlaneModel:
    """왜곡 보정 영상과 체커보드의 실제 mm 평면 사이 변환."""

    corners: np.ndarray
    plane_to_image: np.ndarray
    image_to_plane: np.ndarray
    reprojection_rms_px: float
    inlier_ratio: float
    checker_mean_spacing_px: float = float("nan")
    checker_area_px2: float = float("nan")


@dataclass
class BevGeometry:
    """선택 네 점에서 계산한 실제 비율 BEV 기하."""

    selected_src_points: np.ndarray
    effective_src_points: np.ndarray
    selected_metric_points_mm: np.ndarray
    effective_metric_points_mm: np.ndarray
    dst_points: np.ndarray
    homography: np.ndarray
    warp_width: int
    warp_height: int
    physical_width_mm: float
    physical_height_mm: float
    pixels_per_mm: float
    snap_rms_mm: float
    snap_max_mm: float
    horizontal_parallel_error_deg: float
    vertical_parallel_error_deg: float
    orthogonality_error_deg: float
    opposite_width_error_ratio: float
    opposite_height_error_ratio: float


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 usb_cam 영상에서 왜곡 보정된 640x480 화면의 "
            "BEV 영역을 선택하고 lane_vision_pkg 설정으로 저장합니다."
        )
    )
    parser.add_argument(
        "--transport",
        choices=("compressed", "raw"),
        default=DEFAULT_TRANSPORT,
        help=(
            "입력 영상 전송 방식. DDS 부하가 작은 compressed 권장 "
            f"(기본값: {DEFAULT_TRANSPORT})"
        ),
    )
    parser.add_argument(
        "--topic",
        default=None,
        help=(
            "입력 토픽의 전체 이름. 생략하면 transport에 따라 "
            f"{DEFAULT_COMPRESSED_TOPIC} 또는 {DEFAULT_RAW_TOPIC} 사용"
        ),
    )
    parser.add_argument(
        "--topic-timeout",
        type=float,
        default=DEFAULT_TOPIC_TIMEOUT,
        help=(
            "첫 프레임 또는 실행 중 프레임을 기다리는 시간(초) "
            f"(기본값: {DEFAULT_TOPIC_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
        help=(
            "카메라 캘리브레이션 NPZ "
            "(기본값: 프로젝트 루트 camera_calibration.npz)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_NPZ,
        help=(
            "BEV 설정 NPZ "
            "(기본값: 프로젝트 루트 bev_params_y_auto.npz)"
        ),
    )
    parser.add_argument(
        "--target-square-pixels",
        type=float,
        default=DEFAULT_TARGET_SQUARE_PIXELS,
        help=(
            "BEV에서 체커보드 한 칸의 목표 픽셀 수. 실제 출력은 "
            "크기 상한에 따라 같은 비율로 줄어들 수 있음 "
            f"(기본값: {DEFAULT_TARGET_SQUARE_PIXELS:g})"
        ),
    )
    parser.add_argument(
        "--max-output-side",
        type=int,
        default=DEFAULT_MAX_OUTPUT_SIDE,
        help=(
            "BEV 가로/세로 중 긴 변의 안전 상한(px) "
            f"(기본값: {DEFAULT_MAX_OUTPUT_SIDE})"
        ),
    )

    arguments = parser.parse_args(argv)

    if arguments.topic is None:
        arguments.topic = (
            DEFAULT_COMPRESSED_TOPIC
            if arguments.transport == "compressed"
            else DEFAULT_RAW_TOPIC
        )
    else:
        arguments.topic = arguments.topic.strip()

    if not arguments.topic:
        parser.error("--topic은 빈 문자열일 수 없습니다.")

    if (
        not np.isfinite(arguments.topic_timeout)
        or arguments.topic_timeout <= 0.0
    ):
        parser.error("--topic-timeout은 0보다 커야 합니다.")

    if (
        not np.isfinite(arguments.target_square_pixels)
        or arguments.target_square_pixels < 8.0
        or arguments.target_square_pixels > 200.0
    ):
        parser.error("--target-square-pixels는 8 이상 200 이하여야 합니다.")
    if (
        arguments.max_output_side < MIN_OUTPUT_SIDE
        or arguments.max_output_side > 4096
    ):
        parser.error(
            f"--max-output-side는 {MIN_OUTPUT_SIDE} 이상 4096 이하여야 합니다."
        )

    for name in ("calibration", "output"):
        path: Path = getattr(arguments, name)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        setattr(arguments, name, path.resolve())

    if arguments.calibration.suffix.lower() != ".npz":
        parser.error("--calibration 파일 확장자는 .npz여야 합니다.")
    if arguments.output.suffix.lower() != ".npz":
        parser.error("--output 파일 확장자는 .npz여야 합니다.")
    if arguments.calibration == arguments.output:
        parser.error("--output은 카메라 캘리브레이션 파일과 달라야 합니다.")

    return arguments


def ros_image_to_bgr(message: Image) -> np.ndarray:
    """sensor_msgs/Image의 일반적인 8-bit 인코딩을 BGR 배열로 변환한다."""
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).strip().lower()

    if width <= 0 or height <= 0:
        raise ValueError(
            f"ROS 이미지 크기가 유효하지 않습니다: {width}x{height}"
        )

    if encoding in ("bgr8", "rgb8", "8uc3"):
        channels = 3
    elif encoding in ("bgra8", "rgba8", "8uc4"):
        channels = 4
    elif encoding in ("mono8", "8uc1"):
        channels = 1
    elif encoding in ("yuv422_yuy2", "yuv422", "yuyv"):
        channels = 2
    else:
        raise ValueError(
            f"지원하지 않는 ROS 이미지 인코딩입니다: {message.encoding}"
        )

    packed_row_bytes = width * channels
    if step < packed_row_bytes:
        raise ValueError(
            f"ROS 이미지 step이 너무 작습니다: {step} < {packed_row_bytes}"
        )

    buffer = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = step * height
    if buffer.size < required_bytes:
        raise ValueError(
            "ROS 이미지 데이터가 부족합니다: "
            f"{buffer.size} < {required_bytes}"
        )

    packed = (
        buffer[:required_bytes]
        .reshape(height, step)[:, :packed_row_bytes]
        .copy()
    )

    if channels == 1:
        mono = packed.reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    image = packed.reshape(height, width, channels)

    if encoding in ("bgr8", "8uc3"):
        return image
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUY2)


def ros_compressed_image_to_bgr(
    message: CompressedImage,
) -> np.ndarray:
    """sensor_msgs/CompressedImage의 JPEG/PNG 데이터를 BGR로 디코딩한다."""
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("ROS 압축 이미지 데이터가 비어 있습니다.")

    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError(
            "ROS 압축 이미지를 OpenCV로 디코딩하지 못했습니다: "
            f"{message.format}"
        )

    return frame


class UsbCamTopicSubscriber(RosNode):
    """usb_cam 토픽에서 가장 최신 프레임 하나만 유지한다."""

    def __init__(
        self,
        topic: str,
        transport: str,
        expected_width: int,
        expected_height: int,
    ) -> None:
        super().__init__(f"bev_checkerboard_setup_{os.getpid()}")

        self.topic = topic
        self.transport = transport
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.message_type_name = (
            "sensor_msgs/msg/CompressedImage"
            if transport == "compressed"
            else "sensor_msgs/msg/Image"
        )
        self.latest_frame: np.ndarray | None = None
        self.latest_encoding = ""
        self.latest_frame_id = ""
        self.frame_serial = 0
        self.fatal_error: str | None = None
        self.last_arrival_time = 0.0
        self.arrival_times: deque[float] = deque(maxlen=60)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        if transport == "compressed":
            self.subscription = self.create_subscription(
                CompressedImage,
                topic,
                self._on_compressed_image,
                qos,
            )
        elif transport == "raw":
            self.subscription = self.create_subscription(
                Image,
                topic,
                self._on_raw_image,
                qos,
            )
        else:
            raise ValueError(f"지원하지 않는 transport입니다: {transport}")

    def _on_raw_image(self, message: Image) -> None:
        try:
            frame = ros_image_to_bgr(message)
            encoding = str(message.encoding)
        except (ValueError, cv2.error) as error:
            self.fatal_error = str(error)
            return

        self._accept_frame(
            frame,
            encoding,
            str(message.header.frame_id),
        )

    def _on_compressed_image(
        self,
        message: CompressedImage,
    ) -> None:
        try:
            frame = ros_compressed_image_to_bgr(message)
            encoding = str(message.format)
        except (ValueError, cv2.error) as error:
            self.fatal_error = str(error)
            return

        self._accept_frame(
            frame,
            encoding,
            str(message.header.frame_id),
        )

    def _accept_frame(
        self,
        frame: np.ndarray,
        encoding: str,
        frame_id: str,
    ) -> None:
        if (
            frame.shape[1] != self.expected_width
            or frame.shape[0] != self.expected_height
        ):
            self.fatal_error = (
                "usb_cam 토픽 해상도와 캘리브레이션 해상도가 "
                "다릅니다. 영상을 리사이즈하면 보정값이 깨지므로 "
                "처리를 중단합니다: "
                f"topic={frame.shape[1]}x{frame.shape[0]}, "
                f"calibration={self.expected_width}x"
                f"{self.expected_height}"
            )
            return

        arrival_time = time.monotonic()
        self.latest_frame = frame
        self.latest_encoding = encoding
        self.latest_frame_id = frame_id
        self.frame_serial += 1
        self.last_arrival_time = arrival_time
        self.arrival_times.append(arrival_time)

    def measured_fps(self) -> float:
        if len(self.arrival_times) < 2:
            return 0.0

        elapsed = self.arrival_times[-1] - self.arrival_times[0]
        if elapsed <= 0.0:
            return 0.0

        return (len(self.arrival_times) - 1) / elapsed


def fsync_directory(directory: Path) -> None:
    """Ubuntu 파일시스템에서 rename/link 결과까지 디렉터리에 반영한다."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            encoding=encoding,
            newline="",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
        fsync_directory(path.parent)

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
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"이미지를 저장하지 못했습니다: {path}")

        temporary_path.chmod(0o644)
        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
        fsync_directory(path.parent)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_npz(path: Path, **arrays: object) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    previous_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.chmod(0o644)

        if path.exists():
            previous_path = path.with_name(
                f".{path.stem}.previous{path.suffix}"
            )
            shutil.copy2(path, previous_path)
            with previous_path.open("rb") as previous_file:
                os.fsync(previous_file.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
        fsync_directory(path.parent)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return previous_path


def npz_scalar(
    calibration: dict[str, np.ndarray],
    key: str,
) -> int | float | str:
    value = np.asarray(calibration[key])
    if value.size != 1:
        raise ValueError(
            f"캘리브레이션 {key}는 스칼라여야 합니다: {value.shape}"
        )

    scalar = value.reshape(-1)[0]
    return scalar.item() if isinstance(scalar, np.generic) else scalar


def npz_integer(
    calibration: dict[str, np.ndarray],
    key: str,
) -> int:
    """NPZ 정수 메타데이터를 소수점 잘림 없이 읽는다."""
    scalar = npz_scalar(calibration, key)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"캘리브레이션 {key}는 정수여야 합니다: {scalar!r}"
        ) from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"캘리브레이션 {key}는 정확한 정수여야 합니다: {scalar!r}"
        )
    return int(numeric)


class WebcamBEVCheckerboardSetup:
    def __init__(self, arguments: argparse.Namespace) -> None:
        # 사용자가 직접 고르는 좌표 순서: 좌하, 우하, 좌상, 우상.
        # 하단과 상단의 두 점은 각각 같은 원본 영상 y를 공유한다.
        self.src_points: list[tuple[int, int]] = []

        self.point_names = [
            "Left-Bottom",
            "Right-Bottom",
            "Left-Top",
            "Right-Top",
        ]

        # 화면 정지 및 점 편집 상태
        self.is_frozen = False
        self.frozen_frame: np.ndarray | None = None
        self.latest_frame: np.ndarray | None = None

        self.is_dragging = False
        self.active_point_index: int | None = None
        self.drag_original_point: tuple[int, int] | None = None
        self.drag_original_points: list[tuple[int, int]] | None = None
        self.drag_added_new_point = False

        # 정지 원본에서 구한 체커보드 평면과 현재 네 점의 BEV 해.
        self.plane_model: PlaneModel | None = None
        self.plane_error_message: str | None = None
        self.plane_detection_attempted = False
        self.bev_geometry: BevGeometry | None = None
        self.geometry_error_message: str | None = None

        # 체커보드 검사 캐시
        self.checker_dirty = True
        self.checker_found = False
        self.checker_corners: np.ndarray | None = None
        self.checker_metrics: dict[str, float | str] | None = None
        self.horizontal_spacing: np.ndarray | None = None
        self.vertical_spacing: np.ndarray | None = None
        self.last_detection_time = 0.0

        self.topic: str = arguments.topic
        self.transport: str = arguments.transport
        self.topic_timeout: float = arguments.topic_timeout
        self.calibration_path: Path = arguments.calibration
        self.output_npz: Path = arguments.output
        self.target_square_pixels = float(
            getattr(
                arguments,
                "target_square_pixels",
                DEFAULT_TARGET_SQUARE_PIXELS,
            )
        )
        self.max_output_side = int(
            getattr(
                arguments,
                "max_output_side",
                DEFAULT_MAX_OUTPUT_SIDE,
            )
        )

        calibration = self.load_calibration(self.calibration_path)

        self.camera_matrix = calibration["camera_matrix"]
        self.dist_coeffs = calibration["distortion_coefficients"]
        self.new_camera_matrix = calibration["new_camera_matrix"]

        self.image_width = npz_integer(calibration, "image_width")
        self.image_height = npz_integer(calibration, "image_height")
        self.selection_offset_x = (
            SELECTION_CANVAS_WIDTH - self.image_width
        ) // 2
        self.selection_offset_y = (
            SELECTION_CANVAS_HEIGHT - self.image_height
        ) // 2
        self.checkerboard_size = (
            npz_integer(calibration, "checkerboard_columns"),
            npz_integer(calibration, "checkerboard_rows"),
        )
        self.square_size_mm = float(
            npz_scalar(calibration, "square_size_mm")
        )
        if not math.isclose(
            self.square_size_mm,
            EXPECTED_SQUARE_SIZE_MM,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "이 도구는 한 칸 25mm 체커보드용입니다: "
                f"calibration={self.square_size_mm:g}mm"
            )

        # 실시간 왜곡 보정용 맵
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
            (self.image_width, self.image_height),
            cv2.CV_32FC1,
        )
        if (
            self.map_x.shape != (self.image_height, self.image_width)
            or self.map_y.shape != (self.image_height, self.image_width)
            or not np.all(np.isfinite(self.map_x))
            or not np.all(np.isfinite(self.map_y))
        ):
            raise ValueError(
                "캘리브레이션으로 유효한 왜곡 보정 맵을 만들지 못했습니다."
            )

        self.original_window = "Undistorted - Select Paired-Y BEV Points"
        self.live_window = "Live Undistorted Camera"
        self.bev_window = "BEV Snapshot + Checkerboard Inspection"
        self.live_bev_window = "Live BEV Preview"
        self.live_window_open = False
        self.bev_window_open = False
        self.live_bev_window_open = False
        self.bev_window_image_shape: tuple[int, int] | None = None
        self.node: UsbCamTopicSubscriber | None = None
        self.last_remapped_serial = 0

        try:
            self.node = UsbCamTopicSubscriber(
                self.topic,
                self.transport,
                self.image_width,
                self.image_height,
            )
            cv2.namedWindow(
                self.original_window,
                cv2.WINDOW_AUTOSIZE,
            )
            cv2.setMouseCallback(
                self.original_window,
                self.mouse_callback,
            )
        except Exception:
            if self.node is not None:
                self.node.destroy_node()
                self.node = None
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            raise

    @staticmethod
    def load_calibration(
        calibration_path: Path,
    ) -> dict[str, np.ndarray]:
        """카메라 캘리브레이션 파일을 불러온다."""
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"{calibration_path} 파일이 없습니다.\n"
                "먼저 calibrate_from_images.py를 실행하세요."
            )

        required = {
            "camera_matrix",
            "distortion_coefficients",
            "new_camera_matrix",
            "image_width",
            "image_height",
            "checkerboard_columns",
            "checkerboard_rows",
            "square_size_mm",
        }

        with np.load(
            calibration_path,
            allow_pickle=False,
        ) as data:
            missing = required - set(data.files)

            if missing:
                raise KeyError(
                    f"캘리브레이션 값 누락: {sorted(missing)}"
                )

            calibration = {
                key: np.asarray(data[key]).copy()
                for key in data.files
            }

        camera_matrix = np.asarray(
            calibration["camera_matrix"],
            dtype=np.float64,
        )
        new_camera_matrix = np.asarray(
            calibration["new_camera_matrix"],
            dtype=np.float64,
        )
        distortion_coefficients_raw = np.asarray(
            calibration["distortion_coefficients"],
            dtype=np.float64,
        )

        if camera_matrix.shape != (3, 3):
            raise ValueError(
                "camera_matrix shape가 잘못됐습니다: "
                f"{camera_matrix.shape}"
            )
        if new_camera_matrix.shape != (3, 3):
            raise ValueError(
                "new_camera_matrix shape가 잘못됐습니다: "
                f"{new_camera_matrix.shape}"
            )
        if (
            distortion_coefficients_raw.ndim not in (1, 2)
            or (
                distortion_coefficients_raw.ndim == 2
                and 1 not in distortion_coefficients_raw.shape
            )
            or distortion_coefficients_raw.size not in (4, 5, 8, 12, 14)
        ):
            raise ValueError(
                "distortion_coefficients는 길이 4/5/8/12/14의 "
                f"벡터여야 합니다: {distortion_coefficients_raw.shape}"
            )
        distortion_coefficients = distortion_coefficients_raw.reshape(-1)
        if (
            not np.all(np.isfinite(camera_matrix))
            or not np.all(np.isfinite(new_camera_matrix))
            or not np.all(np.isfinite(distortion_coefficients))
        ):
            raise ValueError("캘리브레이션 행렬에 NaN 또는 inf가 있습니다.")
        if (
            camera_matrix[0, 0] <= 0.0
            or camera_matrix[1, 1] <= 0.0
            or new_camera_matrix[0, 0] <= 0.0
            or new_camera_matrix[1, 1] <= 0.0
        ):
            raise ValueError("캘리브레이션 초점거리는 양수여야 합니다.")
        if (
            np.linalg.matrix_rank(camera_matrix) < 3
            or np.linalg.matrix_rank(new_camera_matrix) < 3
            or abs(float(camera_matrix[2, 2])) < 1.0e-12
            or abs(float(new_camera_matrix[2, 2])) < 1.0e-12
        ):
            raise ValueError("캘리브레이션 카메라 행렬이 특이행렬입니다.")

        image_width = npz_integer(calibration, "image_width")
        image_height = npz_integer(calibration, "image_height")
        if (
            image_width != EXPECTED_IMAGE_WIDTH
            or image_height != EXPECTED_IMAGE_HEIGHT
        ):
            raise ValueError(
                "현재 캘리브레이션이 640x480 결과가 아닙니다: "
                f"{image_width}x{image_height}\n"
                "프로젝트 루트의 camera_calibration.npz를 "
                "확인하세요."
            )

        checkerboard_size = (
            npz_integer(calibration, "checkerboard_columns"),
            npz_integer(calibration, "checkerboard_rows"),
        )
        if checkerboard_size != EXPECTED_CHECKERBOARD_SIZE:
            raise ValueError(
                "캘리브레이션 체커보드 내부 코너가 다릅니다: "
                f"{checkerboard_size} != {EXPECTED_CHECKERBOARD_SIZE}"
            )

        square_size_mm = float(
            npz_scalar(calibration, "square_size_mm")
        )
        if not np.isfinite(square_size_mm) or square_size_mm <= 0.0:
            raise ValueError(
                "캘리브레이션 square_size_mm은 양수여야 합니다."
            )

        if "calibration_model" in calibration:
            calibration_model = str(
                npz_scalar(calibration, "calibration_model")
            )
            if calibration_model != "opencv_pinhole_brown_conrady":
                raise ValueError(
                    "지원하지 않는 카메라 캘리브레이션 모델입니다: "
                    f"{calibration_model}"
                )

        calibration["camera_matrix"] = camera_matrix
        calibration["new_camera_matrix"] = new_camera_matrix
        calibration["distortion_coefficients"] = distortion_coefficients
        return calibration

    def source_geometry_error(
        self,
        points: list[tuple[int, int]] | np.ndarray,
    ) -> str | None:
        """독립적으로 고른 LB, RB, LT, RT 사각형을 검사한다."""
        source = np.asarray(points, dtype=np.float64)
        if source.shape != (4, 2):
            return f"BEV 원본 좌표 shape가 잘못됐습니다: {source.shape}"
        if not np.all(np.isfinite(source)):
            return "BEV 원본 좌표에 NaN 또는 inf가 있습니다."

        minimum_x = -float(self.selection_offset_x)
        maximum_x = float(
            SELECTION_CANVAS_WIDTH - self.selection_offset_x - 1
        )
        minimum_y = -float(self.selection_offset_y)
        maximum_y = float(
            SELECTION_CANVAS_HEIGHT - self.selection_offset_y - 1
        )
        if (
            np.any(source[:, 0] < minimum_x)
            or np.any(source[:, 0] > maximum_x)
            or np.any(source[:, 1] < minimum_y)
            or np.any(source[:, 1] > maximum_y)
        ):
            return "BEV 원본 좌표가 확장 선택 캔버스를 벗어났습니다."

        left_bottom, right_bottom, left_top, right_top = source
        left_center_x = float(
            (left_bottom[0] + left_top[0]) / 2.0
        )
        right_center_x = float(
            (right_bottom[0] + right_top[0]) / 2.0
        )
        bottom_center_y = float(
            (left_bottom[1] + right_bottom[1]) / 2.0
        )
        top_center_y = float(
            (left_top[1] + right_top[1]) / 2.0
        )
        if right_center_x - left_center_x < MIN_EDGE_LENGTH_PX:
            return (
                "LB/LT는 화면 왼쪽, RB/RT는 화면 오른쪽이어야 합니다."
            )
        if bottom_center_y - top_center_y < MIN_EDGE_LENGTH_PX:
            return (
                "LB/RB는 화면 아래, LT/RT는 화면 위쪽이어야 합니다."
            )

        polygon = np.asarray(
            [left_top, right_top, right_bottom, left_bottom],
            dtype=np.float32,
        )

        differences = (
            source[:, None, :] - source[None, :, :]
        )
        distances = np.linalg.norm(differences, axis=2)
        distances += np.eye(4, dtype=np.float64) * 1.0e9
        if float(np.min(distances)) < MIN_EDGE_LENGTH_PX:
            return "BEV 점이 서로 겹치거나 너무 가깝습니다."

        edge_vectors = np.roll(polygon, -1, axis=0) - polygon
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        if float(np.min(edge_lengths)) < MIN_EDGE_LENGTH_PX:
            return (
                "BEV 영역의 변이 너무 짧습니다: "
                f"{float(np.min(edge_lengths)):.1f}px"
            )
        if (
            np.linalg.norm(right_bottom - left_bottom) < MIN_ROI_WIDTH_PX
            or np.linalg.norm(right_top - left_top) < MIN_ROI_WIDTH_PX
        ):
            return "BEV 영역의 위·아래 폭이 너무 좁습니다."

        if not cv2.isContourConvex(polygon):
            return (
                "BEV 네 점이 오목하거나 선이 교차합니다. "
                "LB, RB, LT, RT 순서를 확인하세요."
            )

        cross_products = np.cross(
            edge_vectors,
            np.roll(edge_vectors, -1, axis=0),
        )
        if np.any(np.abs(cross_products) < 1.0):
            return "BEV 점 세 개가 거의 한 직선 위에 있습니다."
        if not (
            np.all(cross_products > 0.0)
            or np.all(cross_products < 0.0)
        ):
            return "BEV 네 점의 순서가 뒤섞였습니다."

        area = float(abs(cv2.contourArea(polygon)))
        if area < MIN_ROI_AREA_PX2:
            return f"BEV 영역 면적이 너무 작습니다: {area:.1f}px²"

        return None

    def mark_checker_dirty(self) -> None:
        """BEV가 바뀌면 이전 검출 결과까지 즉시 무효화한다."""
        self.checker_dirty = True
        self.checker_found = False
        self.checker_corners = None
        self.checker_metrics = None
        self.horizontal_spacing = None
        self.vertical_spacing = None

    def mark_geometry_dirty(self) -> None:
        """선택 점 변경으로 BEV 기하와 최종 검사를 무효화한다."""
        self.bev_geometry = None
        self.mark_checker_dirty()

    def nearest_point_index(
        self,
        point: tuple[int, int],
    ) -> int | None:
        """편집할 수 있을 만큼 가까운 기존 점의 인덱스를 반환한다."""
        if not self.src_points:
            return None

        points = np.asarray(self.src_points, dtype=np.float64)
        distances = np.linalg.norm(
            points - np.asarray(point, dtype=np.float64),
            axis=1,
        )
        index = int(np.argmin(distances))
        if float(distances[index]) > POINT_PICK_RADIUS_PX:
            return None
        return index

    def constrain_pair_y(
        self,
        index: int,
        point: tuple[int, int],
    ) -> tuple[int, int]:
        """RB는 LB, RT는 LT와 같은 원본 영상 y를 사용한다."""

        x, y = point
        if index == 1 and len(self.src_points) >= 1:
            y = self.src_points[0][1]
        elif index == 3 and len(self.src_points) >= 3:
            y = self.src_points[2][1]
        return int(x), int(y)

    def set_point_with_paired_y(
        self,
        index: int,
        point: tuple[int, int],
    ) -> None:
        """선택점을 갱신하고 이미 존재하는 짝점의 y도 정렬한다."""

        point = self.constrain_pair_y(index, point)
        self.src_points[index] = point
        _, y = point
        if index == 0 and len(self.src_points) >= 2:
            partner_x, _ = self.src_points[1]
            self.src_points[1] = (partner_x, y)
        elif index == 2 and len(self.src_points) >= 4:
            partner_x, _ = self.src_points[3]
            self.src_points[3] = (partner_x, y)

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        """LB/RB와 LT/RT의 y를 맞춰 네 BEV 점을 배치한다."""
        del param

        if not self.is_frozen:
            if event == cv2.EVENT_LBUTTONDOWN:
                print("[WARNING] 먼저 SPACE를 눌러 화면을 멈추세요.")
            return

        canvas_x = int(np.clip(x, 0, SELECTION_CANVAS_WIDTH - 1))
        canvas_y = int(np.clip(y, 0, SELECTION_CANVAS_HEIGHT - 1))
        point = (
            canvas_x - self.selection_offset_x,
            canvas_y - self.selection_offset_y,
        )

        if event == cv2.EVENT_RBUTTONDOWN:
            if self.is_dragging:
                return
            if self.src_points:
                removed_name = self.point_names[len(self.src_points) - 1]
                removed = self.src_points.pop()
                self.mark_geometry_dirty()
                print(f"[되돌리기] {removed_name} {removed} 제거")
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_original_points = list(self.src_points)
            if len(self.src_points) < 4:
                point = self.constrain_pair_y(len(self.src_points), point)
                self.src_points.append(point)
                self.active_point_index = len(self.src_points) - 1
                self.drag_original_point = None
                self.drag_added_new_point = True
                print(
                    f"[{self.point_names[self.active_point_index]} 선택] "
                    f"{point}"
                )
            else:
                self.active_point_index = self.nearest_point_index(point)
                if self.active_point_index is None:
                    print(
                        "[점 편집] 이동할 점의 원 안쪽을 누르세요."
                    )
                    return
                self.drag_original_point = self.src_points[
                    self.active_point_index
                ]
                self.drag_added_new_point = False

            self.is_dragging = True
            self.set_point_with_paired_y(self.active_point_index, point)
            self.mark_geometry_dirty()

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.is_dragging
            and (flags & cv2.EVENT_FLAG_LBUTTON)
        ):
            if self.active_point_index is not None:
                self.set_point_with_paired_y(
                    self.active_point_index,
                    point,
                )
                self.mark_geometry_dirty()

        elif (
            event == cv2.EVENT_LBUTTONUP
            and self.is_dragging
        ):
            self.is_dragging = False
            if self.active_point_index is not None:
                self.set_point_with_paired_y(
                    self.active_point_index,
                    point,
                )

            if len(self.src_points) == 4:
                geometry_error = self.source_geometry_error(self.src_points)
                if geometry_error is not None:
                    if self.drag_added_new_point:
                        self.src_points.pop()
                    elif self.drag_original_points is not None:
                        self.src_points = self.drag_original_points
                    print(f"[선택 취소] {geometry_error}")
                else:
                    print("[네 점 확정]")
                    for name, selected in zip(
                        self.point_names,
                        self.src_points,
                    ):
                        print(f"  {name}: {selected}")
                    print(
                        "각 점을 다시 드래그해 미세 조정할 수 있습니다. "
                        "BEV를 확인한 뒤 S를 누르세요."
                    )

            elif len(self.src_points) < 4:
                next_name = self.point_names[len(self.src_points)]
                print(f"[다음 점] {next_name}을 선택하세요.")

            self.active_point_index = None
            self.drag_original_point = None
            self.drag_original_points = None
            self.drag_added_new_point = False
            self.mark_geometry_dirty()

    @staticmethod
    def transform_points(
        homography: np.ndarray,
        points: np.ndarray | list[tuple[int, int]],
    ) -> np.ndarray:
        """2D 점들을 homography로 옮기고 무한대/NaN 결과를 거부한다."""
        matrix = np.asarray(homography, dtype=np.float64)
        source = np.asarray(points, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"homography shape가 잘못됐습니다: {matrix.shape}")
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError(f"점 배열 shape가 잘못됐습니다: {source.shape}")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(source)):
            raise ValueError("homography 또는 점 배열에 NaN/inf가 있습니다.")

        transformed = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2),
            matrix,
        ).reshape(-1, 2)
        if not np.all(np.isfinite(transformed)):
            raise ValueError("평면 투영 결과가 무한대입니다.")
        return transformed

    @staticmethod
    def vector_angle_degrees(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """두 2D 벡터 사이의 0~180도 각도를 반환한다."""
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        denominator = float(
            np.linalg.norm(first) * np.linalg.norm(second)
        )
        if denominator <= 1.0e-12:
            raise ValueError("길이가 0인 방향 벡터가 있습니다.")
        cosine = float(
            np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
        )
        return float(np.degrees(np.arccos(cosine)))

    def build_plane_model(
        self,
        frame: np.ndarray,
        *,
        exhaustive: bool,
    ) -> PlaneModel:
        """원본 체커보드로 image↔ground-plane(mm) 변환을 계산한다."""
        corners = self.detect_checkerboard_corners(
            frame,
            exhaustive=exhaustive,
            use_classic_fallback=True,
        )
        if corners is None:
            raise ValueError(
                "왜곡 보정 원본에서 체커보드 "
                f"{self.checkerboard_size[0]}x"
                f"{self.checkerboard_size[1]} 내부 코너를 "
                "검출하지 못했습니다."
            )

        columns, rows = self.checkerboard_size
        grid_x, grid_y = np.meshgrid(
            np.arange(columns, dtype=np.float64),
            np.arange(rows, dtype=np.float64),
        )
        object_points_mm = np.column_stack(
            (grid_x.reshape(-1), grid_y.reshape(-1))
        )
        object_points_mm *= self.square_size_mm
        image_points = np.asarray(
            corners,
            dtype=np.float64,
        ).reshape(-1, 2)
        image_grid = image_points.reshape(rows, columns, 2)
        horizontal_spacing = np.linalg.norm(
            image_grid[:, 1:, :] - image_grid[:, :-1, :],
            axis=2,
        )
        vertical_spacing = np.linalg.norm(
            image_grid[1:, :, :] - image_grid[:-1, :, :],
            axis=2,
        )
        checker_mean_spacing_px = float(np.mean(np.concatenate(
            (
                horizontal_spacing.reshape(-1),
                vertical_spacing.reshape(-1),
            )
        )))
        checker_area_px2 = float(cv2.contourArea(
            cv2.convexHull(
                np.asarray(image_points, dtype=np.float32)
            )
        ))
        if checker_mean_spacing_px < MIN_SOURCE_CHECKER_SPACING_PX:
            raise ValueError(
                "원본 체커보드가 너무 작거나 멉니다: "
                f"평균 한 칸 {checker_mean_spacing_px:.2f}px "
                f"(필요 {MIN_SOURCE_CHECKER_SPACING_PX:.0f}px 이상)"
            )
        if checker_area_px2 < MIN_SOURCE_CHECKER_AREA_PX2:
            raise ValueError(
                "원본 체커보드 영상 면적이 너무 작습니다: "
                f"{checker_area_px2:.0f}px² "
                f"(필요 {MIN_SOURCE_CHECKER_AREA_PX2:.0f}px² 이상)"
            )

        plane_to_image, inlier_mask = cv2.findHomography(
            object_points_mm,
            image_points,
            cv2.RANSAC,
            PLANE_RANSAC_THRESHOLD_PX,
        )
        if plane_to_image is None or inlier_mask is None:
            raise ValueError("체커보드 바닥 평면 homography 계산에 실패했습니다.")

        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_ratio = float(np.mean(inliers))
        if (
            int(np.count_nonzero(inliers)) < 4
            or inlier_ratio < MIN_PLANE_INLIER_RATIO
        ):
            raise ValueError(
                "체커보드 평면 inlier가 부족합니다: "
                f"{inlier_ratio * 100.0:.1f}% "
                f"(필요 {MIN_PLANE_INLIER_RATIO * 100.0:.0f}% 이상)"
            )

        refined, _ = cv2.findHomography(
            object_points_mm[inliers],
            image_points[inliers],
            0,
        )
        if refined is not None:
            plane_to_image = refined

        plane_to_image = np.asarray(plane_to_image, dtype=np.float64)
        if (
            plane_to_image.shape != (3, 3)
            or not np.all(np.isfinite(plane_to_image))
            or np.linalg.matrix_rank(plane_to_image) < 3
        ):
            raise ValueError("체커보드 바닥 평면 homography가 유효하지 않습니다.")

        projected = self.transform_points(
            plane_to_image,
            object_points_mm[inliers],
        )
        residual = projected - image_points[inliers]
        reprojection_rms_px = float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
        )
        if reprojection_rms_px > MAX_PLANE_REPROJECTION_RMS_PX:
            raise ValueError(
                "체커보드 평면 오차가 너무 큽니다: "
                f"RMS {reprojection_rms_px:.3f}px "
                f"(허용 {MAX_PLANE_REPROJECTION_RMS_PX:.1f}px)"
            )

        try:
            image_to_plane = np.linalg.inv(plane_to_image)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "체커보드 평면 homography의 역행렬이 없습니다."
            ) from error

        if not np.all(np.isfinite(image_to_plane)):
            raise ValueError(
                "체커보드 평면 homography 역행렬에 NaN/inf가 있습니다."
            )
        return PlaneModel(
            corners=np.asarray(corners, dtype=np.float32),
            plane_to_image=plane_to_image,
            image_to_plane=image_to_plane,
            reprojection_rms_px=reprojection_rms_px,
            inlier_ratio=inlier_ratio,
            checker_mean_spacing_px=checker_mean_spacing_px,
            checker_area_px2=checker_area_px2,
        )

    def ensure_plane_model(
        self,
        frame: np.ndarray,
        *,
        force: bool = False,
        strict: bool = False,
    ) -> PlaneModel | None:
        """정지 프레임의 체커보드 평면을 한 번 계산해 재사용한다."""
        if self.plane_model is not None and not force:
            return self.plane_model
        if (
            self.plane_detection_attempted
            and self.plane_model is None
            and not force
        ):
            return None

        try:
            model = self.build_plane_model(
                frame,
                exhaustive=force,
            )
        except (ValueError, cv2.error) as error:
            self.plane_detection_attempted = True
            self.plane_model = None
            message = str(error)
            if message != self.plane_error_message:
                print(f"[체커보드 평면 실패] {message}")
            self.plane_error_message = message
            if strict:
                raise ValueError(message) from error
            return None

        self.plane_model = model
        self.plane_detection_attempted = True
        self.plane_error_message = None
        self.bev_geometry = None
        self.mark_checker_dirty()
        print(
            "[체커보드 평면 준비] "
            f"RMS {model.reprojection_rms_px:.3f}px, "
            f"inlier {model.inlier_ratio * 100.0:.1f}%, "
            f"square {model.checker_mean_spacing_px:.1f}px"
        )
        return model

    def choose_output_layout(
        self,
        physical_width_mm: float,
        physical_height_mm: float,
    ) -> tuple[int, int, float, np.ndarray]:
        """동일 px/mm 배율을 유지하면서 실제 비율의 출력 크기를 정한다."""
        if (
            not np.isfinite(physical_width_mm)
            or not np.isfinite(physical_height_mm)
            or physical_width_mm <= 0.0
            or physical_height_mm <= 0.0
        ):
            raise ValueError("BEV 실제 가로·세로 길이가 유효하지 않습니다.")

        preferred_scale = (
            self.target_square_pixels / self.square_size_mm
        )
        side_limited_scale = (
            (self.max_output_side - 1)
            / max(physical_width_mm, physical_height_mm)
        )
        pixels_per_mm = min(preferred_scale, side_limited_scale)
        if not np.isfinite(pixels_per_mm) or pixels_per_mm <= 0.0:
            raise ValueError("BEV px/mm 배율을 계산하지 못했습니다.")

        def snap_near_integer(extent: float) -> float:
            tolerance = max(
                1.0e-6,
                abs(extent) * 1.0e-6,
            )
            nearest_integer = float(round(extent))
            if abs(extent - nearest_integer) <= tolerance:
                return nearest_integer
            return extent

        def canvas_size(scale: float) -> tuple[int, int]:
            width_extent = snap_near_integer(
                physical_width_mm * scale
            )
            height_extent = snap_near_integer(
                physical_height_mm * scale
            )
            width = int(math.ceil(width_extent)) + 1
            height = int(math.ceil(height_extent)) + 1
            return width, height

        warp_width, warp_height = canvas_size(pixels_per_mm)
        for _ in range(12):
            if (
                warp_width <= self.max_output_side
                and warp_height <= self.max_output_side
                and warp_width * warp_height <= DEFAULT_MAX_OUTPUT_PIXELS
            ):
                break
            pixel_factor = math.sqrt(
                DEFAULT_MAX_OUTPUT_PIXELS
                / max(1, warp_width * warp_height)
            )
            side_factor = min(
                self.max_output_side / max(1, warp_width),
                self.max_output_side / max(1, warp_height),
            )
            pixels_per_mm *= min(pixel_factor, side_factor) * 0.999
            warp_width, warp_height = canvas_size(pixels_per_mm)
        else:
            raise ValueError("BEV 출력 크기 안전 상한을 맞추지 못했습니다.")

        actual_square_pixels = pixels_per_mm * self.square_size_mm
        if actual_square_pixels < 5.0:
            raise ValueError(
                "출력 상한을 적용하면 체커보드 한 칸이 너무 작습니다: "
                f"{actual_square_pixels:.2f}px. 선택 영역을 줄이거나 "
                "--max-output-side를 늘리세요."
            )
        if (
            warp_width < MIN_OUTPUT_SIDE
            or warp_height < MIN_OUTPUT_SIDE
        ):
            raise ValueError(
                "실제 비율을 유지하면 BEV 한 변이 너무 작습니다: "
                f"{warp_width}x{warp_height}. 선택 영역을 조정하세요."
            )

        content_width = snap_near_integer(
            physical_width_mm * pixels_per_mm
        )
        content_height = snap_near_integer(
            physical_height_mm * pixels_per_mm
        )
        destination = np.float32([
            [0.0, content_height],
            [content_width, content_height],
            [0.0, 0.0],
            [content_width, 0.0],
        ])
        return (
            warp_width,
            warp_height,
            float(pixels_per_mm),
            destination,
        )

    def compute_bev_geometry(
        self,
        plane_model: PlaneModel,
    ) -> BevGeometry:
        """클릭 네 점을 바닥 mm 평면의 최적 직사각형으로 맞춘다."""
        source_error = self.source_geometry_error(self.src_points)
        if source_error is not None:
            raise ValueError(source_error)

        selected_source = np.asarray(
            self.src_points,
            dtype=np.float64,
        )
        selected_metric = self.transform_points(
            plane_model.image_to_plane,
            selected_source,
        )
        left_bottom, right_bottom, left_top, right_top = selected_metric

        horizontal_bottom = right_bottom - left_bottom
        horizontal_top = right_top - left_top
        vertical_left = left_top - left_bottom
        vertical_right = right_top - right_bottom

        vectors = (
            horizontal_bottom,
            horizontal_top,
            vertical_left,
            vertical_right,
        )
        lengths = [float(np.linalg.norm(vector)) for vector in vectors]
        if min(lengths) < 2.0 * self.square_size_mm:
            raise ValueError(
                "선택 영역의 실제 변 길이가 체커보드 두 칸보다 작습니다."
            )

        horizontal_parallel_error = self.vector_angle_degrees(
            horizontal_bottom,
            horizontal_top,
        )
        vertical_parallel_error = self.vector_angle_degrees(
            vertical_left,
            vertical_right,
        )
        if (
            horizontal_parallel_error > MAX_OPPOSITE_EDGE_ANGLE_DEG
            or vertical_parallel_error > MAX_OPPOSITE_EDGE_ANGLE_DEG
        ):
            raise ValueError(
                "반대편 변이 실제 바닥에서 충분히 평행하지 않습니다: "
                f"가로 {horizontal_parallel_error:.1f}°, "
                f"세로 {vertical_parallel_error:.1f}°"
            )

        width_error_ratio = abs(lengths[0] - lengths[1]) / (
            (lengths[0] + lengths[1]) / 2.0
        )
        height_error_ratio = abs(lengths[2] - lengths[3]) / (
            (lengths[2] + lengths[3]) / 2.0
        )
        if (
            width_error_ratio > MAX_OPPOSITE_EDGE_LENGTH_ERROR
            or height_error_ratio > MAX_OPPOSITE_EDGE_LENGTH_ERROR
        ):
            raise ValueError(
                "반대편 변 길이 차이가 너무 큽니다: "
                f"가로 {width_error_ratio * 100.0:.1f}%, "
                f"세로 {height_error_ratio * 100.0:.1f}%"
            )

        horizontal_axis_hint = (
            horizontal_bottom / lengths[0]
            + horizontal_top / lengths[1]
        )
        vertical_axis_hint = (
            vertical_left / lengths[2]
            + vertical_right / lengths[3]
        )
        horizontal_axis_hint /= np.linalg.norm(horizontal_axis_hint)
        vertical_axis_hint /= np.linalg.norm(vertical_axis_hint)

        hint_dot = float(
            abs(np.dot(horizontal_axis_hint, vertical_axis_hint))
        )
        if hint_dot > MAX_AXIS_PARALLEL_DOT:
            raise ValueError(
                "가로·세로 방향이 거의 평행합니다. 점 순서를 확인하세요."
            )
        raw_orthogonality_error = abs(
            90.0
            - self.vector_angle_degrees(
                horizontal_axis_hint,
                vertical_axis_hint,
            )
        )
        if (
            raw_orthogonality_error
            > MAX_RAW_ORTHOGONALITY_ERROR_DEG
        ):
            raise ValueError(
                "선택 영역의 실제 가로·세로가 직각에서 너무 벗어났습니다: "
                f"{raw_orthogonality_error:.1f}° "
                f"(허용 {MAX_RAW_ORTHOGONALITY_ERROR_DEG:.0f}°)"
            )

        axes = np.column_stack(
            (horizontal_axis_hint, vertical_axis_hint)
        )
        left_singular, _, right_singular = np.linalg.svd(axes)
        orthogonal_axes = left_singular @ right_singular
        horizontal_axis = orthogonal_axes[:, 0]
        vertical_axis = orthogonal_axes[:, 1]
        if np.dot(horizontal_axis, horizontal_axis_hint) < 0.0:
            horizontal_axis *= -1.0
        if np.dot(vertical_axis, vertical_axis_hint) < 0.0:
            vertical_axis *= -1.0

        x_left = float(np.mean(
            selected_metric[[0, 2]] @ horizontal_axis
        ))
        x_right = float(np.mean(
            selected_metric[[1, 3]] @ horizontal_axis
        ))
        y_bottom = float(np.mean(
            selected_metric[[0, 1]] @ vertical_axis
        ))
        y_top = float(np.mean(
            selected_metric[[2, 3]] @ vertical_axis
        ))

        physical_width_mm = x_right - x_left
        physical_height_mm = y_top - y_bottom
        if (
            physical_width_mm < 2.0 * self.square_size_mm
            or physical_height_mm < 2.0 * self.square_size_mm
        ):
            raise ValueError(
                "점 순서가 잘못됐거나 실제 BEV 영역이 너무 작습니다."
            )

        effective_metric = np.asarray([
            x_left * horizontal_axis + y_bottom * vertical_axis,
            x_right * horizontal_axis + y_bottom * vertical_axis,
            x_left * horizontal_axis + y_top * vertical_axis,
            x_right * horizontal_axis + y_top * vertical_axis,
        ], dtype=np.float64)

        snap_errors = np.linalg.norm(
            selected_metric - effective_metric,
            axis=1,
        )
        snap_rms_mm = float(
            np.sqrt(np.mean(snap_errors * snap_errors))
        )
        snap_max_mm = float(np.max(snap_errors))
        if snap_max_mm > MAX_SNAP_ERROR_SQUARES * self.square_size_mm:
            raise ValueError(
                "선택 네 점의 직사각형 보정량이 너무 큽니다: "
                f"최대 {snap_max_mm:.1f}mm "
                f"({snap_max_mm / self.square_size_mm:.2f}칸)"
            )

        effective_source = self.transform_points(
            plane_model.plane_to_image,
            effective_metric,
        )
        effective_error = self.source_geometry_error(effective_source)
        if effective_error is not None:
            raise ValueError(
                "물리 직사각형 보정 후 영상 영역이 유효하지 않습니다: "
                f"{effective_error}"
            )

        (
            warp_width,
            warp_height,
            pixels_per_mm,
            destination,
        ) = self.choose_output_layout(
            physical_width_mm,
            physical_height_mm,
        )

        homography = cv2.getPerspectiveTransform(
            np.asarray(effective_source, dtype=np.float32),
            destination,
        )
        if (
            homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or np.linalg.matrix_rank(homography) < 3
        ):
            raise ValueError("유효한 metric BEV homography를 계산하지 못했습니다.")
        condition_number = float(np.linalg.cond(homography))
        if (
            not np.isfinite(condition_number)
            or condition_number > 1.0e10
        ):
            raise ValueError(
                "metric BEV homography가 수치적으로 불안정합니다: "
                f"condition={condition_number:.3e}"
            )

        projected_destination = self.transform_points(
            homography,
            effective_source,
        )
        if float(np.max(np.abs(
            projected_destination - destination
        ))) > 0.05:
            raise ValueError("BEV 네 점의 변환 일치 검증에 실패했습니다.")

        return BevGeometry(
            selected_src_points=np.asarray(
                selected_source,
                dtype=np.float32,
            ),
            effective_src_points=np.asarray(
                effective_source,
                dtype=np.float32,
            ),
            selected_metric_points_mm=np.asarray(
                selected_metric,
                dtype=np.float64,
            ),
            effective_metric_points_mm=effective_metric,
            dst_points=destination,
            homography=np.asarray(homography, dtype=np.float64),
            warp_width=warp_width,
            warp_height=warp_height,
            physical_width_mm=physical_width_mm,
            physical_height_mm=physical_height_mm,
            pixels_per_mm=pixels_per_mm,
            snap_rms_mm=snap_rms_mm,
            snap_max_mm=snap_max_mm,
            horizontal_parallel_error_deg=horizontal_parallel_error,
            vertical_parallel_error_deg=vertical_parallel_error,
            orthogonality_error_deg=raw_orthogonality_error,
            opposite_width_error_ratio=width_error_ratio,
            opposite_height_error_ratio=height_error_ratio,
        )

    def read_undistorted_frame(
        self,
    ) -> np.ndarray | None:
        """선택 GUI용 프레임을 반환한다. 정지 중에는 고정 화면이다."""
        live_frame = self.read_live_undistorted_frame()
        if self.is_frozen:
            if self.frozen_frame is None:
                return None

            return self.frozen_frame

        return live_frame

    def read_live_undistorted_frame(
        self,
    ) -> np.ndarray | None:
        """정지 상태와 무관하게 가장 최근 ROS 프레임을 왜곡 보정한다."""

        if self.node is None or self.node.latest_frame is None:
            return self.latest_frame

        if self.node.frame_serial == self.last_remapped_serial:
            return self.latest_frame

        undistorted = cv2.remap(
            self.node.latest_frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        self.latest_frame = undistorted
        self.last_remapped_serial = self.node.frame_serial
        return undistorted

    def wait_for_first_frame(self) -> None:
        """첫 usb_cam 프레임을 기다리고 실패 원인을 자세히 알린다."""
        if self.node is None:
            raise RuntimeError("ROS 구독 노드가 생성되지 않았습니다.")

        deadline = time.monotonic() + self.topic_timeout
        while self.node.latest_frame is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

            if self.node.fatal_error is not None:
                raise RuntimeError(self.node.fatal_error)

            if time.monotonic() >= deadline:
                publisher_count = self.node.count_publishers(self.topic)
                raise RuntimeError(
                    "usb_cam 프레임을 받지 못했습니다.\n"
                    f"토픽: {self.topic}\n"
                    f"transport: {self.transport}\n"
                    f"발견한 publisher 수: {publisher_count}\n"
                    "usb_cam launch와 토픽 이름을 확인하세요:\n"
                    f"  ros2 topic info {self.topic} --verbose"
                )

        if self.read_undistorted_frame() is None:
            raise RuntimeError("첫 ROS 프레임의 왜곡 보정에 실패했습니다.")

    def spin_topic_once(self) -> None:
        """한 번 spin하고 디코딩 오류 또는 영상 중단을 확인한다."""
        if self.node is None:
            raise RuntimeError("ROS 구독 노드가 생성되지 않았습니다.")

        rclpy.spin_once(self.node, timeout_sec=0.01)

        if self.node.fatal_error is not None:
            raise RuntimeError(self.node.fatal_error)

        if (
            self.node.last_arrival_time > 0.0
            and time.monotonic() - self.node.last_arrival_time
            > self.topic_timeout
        ):
            raise RuntimeError(
                "usb_cam 토픽 프레임 수신이 중단됐습니다: "
                f"{self.topic}"
            )

    def draw_grid(
        self,
        image: np.ndarray,
    ) -> None:
        """원본 카메라 화면에 위치 확인용 픽셀 격자를 표시한다."""
        height, width = image.shape[:2]

        for x in range(0, width, GRID_STEP_X):
            cv2.line(
                image,
                (x, 0),
                (x, height - 1),
                (100, 100, 100),
                1,
            )

        for y in range(0, height, GRID_STEP_Y):
            cv2.line(
                image,
                (0, y),
                (width - 1, y),
                (100, 100, 100),
                1,
            )

    def show_bev_window(self, image: np.ndarray) -> None:
        """저장 해상도는 유지하고 GUI 창만 화면 안에 비율 보존 축소한다."""
        if self.bev_window_open:
            try:
                if cv2.getWindowProperty(
                    self.bev_window,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    self.bev_window_open = False
                    self.bev_window_image_shape = None
            except cv2.error:
                self.bev_window_open = False
                self.bev_window_image_shape = None
        if not self.bev_window_open:
            cv2.namedWindow(
                self.bev_window,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            self.bev_window_open = True
        height, width = image.shape[:2]
        if self.bev_window_image_shape != (height, width):
            preview_scale = min(
                1.0,
                MAX_BEV_PREVIEW_WIDTH / width,
                MAX_BEV_PREVIEW_HEIGHT / height,
            )
            cv2.resizeWindow(
                self.bev_window,
                max(1, int(round(width * preview_scale))),
                max(1, int(round(height * preview_scale))),
            )
            self.bev_window_image_shape = (height, width)
        cv2.imshow(self.bev_window, image)

    def show_live_window(self, image: np.ndarray) -> None:
        """네 점 확정 후 최신 왜곡 보정 원본을 별도 창에 표시한다."""

        if self.live_window_open:
            try:
                if cv2.getWindowProperty(
                    self.live_window,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    self.live_window_open = False
            except cv2.error:
                self.live_window_open = False
        if not self.live_window_open:
            cv2.namedWindow(self.live_window, cv2.WINDOW_AUTOSIZE)
            self.live_window_open = True
        cv2.imshow(self.live_window, image)

    def close_live_window(self) -> None:
        if not self.live_window_open:
            return
        try:
            cv2.destroyWindow(self.live_window)
        except cv2.error:
            pass
        self.live_window_open = False

    def show_live_bev_window(self, image: np.ndarray) -> None:
        """현재 선택 좌표를 적용한 최신 BEV를 별도 창에 표시한다."""

        if self.live_bev_window_open:
            try:
                if cv2.getWindowProperty(
                    self.live_bev_window,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    self.live_bev_window_open = False
            except cv2.error:
                self.live_bev_window_open = False
        if not self.live_bev_window_open:
            cv2.namedWindow(
                self.live_bev_window,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.resizeWindow(
                self.live_bev_window,
                DIRECT_PREVIEW_WIDTH,
                DIRECT_PREVIEW_HEIGHT,
            )
            self.live_bev_window_open = True
        cv2.imshow(self.live_bev_window, image)

    def close_live_bev_window(self) -> None:
        if not self.live_bev_window_open:
            return
        try:
            cv2.destroyWindow(self.live_bev_window)
        except cv2.error:
            pass
        self.live_bev_window_open = False

    def image_to_canvas_point(
        self,
        point: tuple[int, int] | np.ndarray,
    ) -> tuple[int, int]:
        """카메라 픽셀 좌표를 확장 선택 캔버스 좌표로 바꾼다."""

        return (
            int(round(float(point[0]))) + self.selection_offset_x,
            int(round(float(point[1]))) + self.selection_offset_y,
        )

    def draw_original_overlay(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """원본 보정 화면에 선택 ROI를 표시한다."""
        camera_display = frame.copy()
        self.draw_grid(camera_display)
        display = np.full(
            (SELECTION_CANVAS_HEIGHT, SELECTION_CANVAS_WIDTH, 3),
            28,
            dtype=np.uint8,
        )
        x0 = self.selection_offset_x
        y0 = self.selection_offset_y
        display[
            y0 : y0 + self.image_height,
            x0 : x0 + self.image_width,
        ] = camera_display
        cv2.rectangle(
            display,
            (x0, y0),
            (x0 + self.image_width - 1, y0 + self.image_height - 1),
            (180, 180, 180),
            2,
        )

        if (
            self.plane_model is not None
            and self.plane_model.corners is not None
        ):
            shifted_corners = self.plane_model.corners.copy()
            shifted_corners[:, 0, 0] += self.selection_offset_x
            shifted_corners[:, 0, 1] += self.selection_offset_y
            cv2.drawChessboardCorners(
                display,
                self.checkerboard_size,
                shifted_corners,
                True,
            )

        labels = ["LB", "RB", "LT", "RT"]

        for index, point in enumerate(self.src_points):
            canvas_point = self.image_to_canvas_point(point)
            point_color = (
                (0, 255, 255)
                if index == self.active_point_index
                else (0, 255, 0)
            )
            cv2.circle(
                display,
                canvas_point,
                7,
                point_color,
                -1,
            )

            cv2.putText(
                display,
                labels[index],
                (canvas_point[0] + 8, canvas_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                point_color,
                2,
                cv2.LINE_AA,
            )

        if len(self.src_points) == 4:
            polygon = np.array([
                self.image_to_canvas_point(self.src_points[2]),  # 좌상
                self.image_to_canvas_point(self.src_points[3]),  # 우상
                self.image_to_canvas_point(self.src_points[1]),  # 우하
                self.image_to_canvas_point(self.src_points[0]),  # 좌하
            ], dtype=np.int32)

            cv2.polylines(
                display,
                [polygon],
                True,
                (
                    (0, 0, 255)
                    if self.source_geometry_error(self.src_points) is not None
                    else (0, 255, 0)
                ),
                2,
            )

        if self.bev_geometry is not None:
            effective_polygon = np.asarray([
                self.image_to_canvas_point(
                    self.bev_geometry.effective_src_points[2]
                ),
                self.image_to_canvas_point(
                    self.bev_geometry.effective_src_points[3]
                ),
                self.image_to_canvas_point(
                    self.bev_geometry.effective_src_points[1]
                ),
                self.image_to_canvas_point(
                    self.bev_geometry.effective_src_points[0]
                ),
            ], dtype=np.int32)
            cv2.polylines(
                display,
                [effective_polygon],
                True,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        status = "FROZEN" if self.is_frozen else "LIVE"
        if len(self.src_points) < 4:
            action = f"Click {labels[len(self.src_points)]}"
        else:
            action = "Drag handles | metric rectangle: MAGENTA"

        cv2.putText(
            display,
            (
                f"{status} | {action} | "
                f"App {self.node.measured_fps() if self.node else 0.0:.1f} FPS"
            ),
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.bev_geometry is not None:
            geometry_text = (
                f"BEV {self.bev_geometry.warp_width}x"
                f"{self.bev_geometry.warp_height}px | "
                f"{self.bev_geometry.physical_width_mm:.0f}x"
                f"{self.bev_geometry.physical_height_mm:.0f}mm"
            )
            geometry_color = (255, 0, 255)
        elif self.is_frozen and self.plane_model is None:
            geometry_text = "Checkerboard ground plane: NOT READY"
            geometry_color = (0, 0, 255)
        else:
            geometry_text = ""
            geometry_color = (255, 255, 255)

        if geometry_text:
            cv2.putText(
                display,
                geometry_text,
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                geometry_color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            "SPACE: freeze  RMB: undo  R: reset  S: save  Q: quit",
            (15, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return display

    def make_bev(
        self,
        frame: np.ndarray,
        *,
        strict: bool = False,
        refresh_plane: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """체커보드 mm 평면과 현재 네 점으로 실제 비율 BEV를 만든다."""
        if len(self.src_points) != 4:
            return None

        geometry_error = self.source_geometry_error(self.src_points)
        if geometry_error is not None:
            if strict:
                raise ValueError(geometry_error)
            return None

        if (
            frame.shape[1] != self.image_width
            or frame.shape[0] != self.image_height
        ):
            raise ValueError(
                "BEV 입력 프레임 해상도가 캘리브레이션과 다릅니다: "
                f"{frame.shape[1]}x{frame.shape[0]} != "
                f"{self.image_width}x{self.image_height}"
            )

        plane_model = self.ensure_plane_model(
            frame,
            force=refresh_plane,
            strict=strict,
        )
        if plane_model is None:
            return None

        if self.bev_geometry is None:
            try:
                geometry = self.compute_bev_geometry(plane_model)
            except (ValueError, cv2.error) as error:
                message = str(error)
                if message != self.geometry_error_message:
                    print(f"[BEV 계산 실패] {message}")
                self.geometry_error_message = message
                if strict:
                    raise ValueError(message) from error
                return None

            self.bev_geometry = geometry
            self.geometry_error_message = None
            self.mark_checker_dirty()
            print(
                "[실제 비율 BEV] "
                f"{geometry.physical_width_mm:.1f} x "
                f"{geometry.physical_height_mm:.1f} mm -> "
                f"{geometry.warp_width} x {geometry.warp_height} px, "
                f"{geometry.pixels_per_mm:.4f} px/mm"
            )
            if (
                geometry.snap_max_mm
                > WARN_SNAP_ERROR_SQUARES * self.square_size_mm
            ):
                print(
                    "[주의] 네 점의 직사각형 자동 보정량: "
                    f"최대 {geometry.snap_max_mm:.1f}mm "
                    f"({geometry.snap_max_mm / self.square_size_mm:.2f}칸)"
                )

        geometry = self.bev_geometry

        bev = cv2.warpPerspective(
            frame,
            geometry.homography,
            (geometry.warp_width, geometry.warp_height),
            flags=cv2.INTER_LINEAR,
        )

        return bev, geometry.homography

    def make_direct_preview_bev(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """체커보드 평면과 무관하게 네 클릭점으로 즉시 BEV를 만든다."""

        if len(self.src_points) != 4:
            return None
        if self.source_geometry_error(self.src_points) is not None:
            return None
        source = np.asarray(self.src_points, dtype=np.float32)
        destination = np.asarray(
            [
                [0.0, DIRECT_PREVIEW_HEIGHT - 1.0],
                [DIRECT_PREVIEW_WIDTH - 1.0, DIRECT_PREVIEW_HEIGHT - 1.0],
                [0.0, 0.0],
                [DIRECT_PREVIEW_WIDTH - 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination)
        if not np.all(np.isfinite(homography)):
            return None
        bev = cv2.warpPerspective(
            frame,
            homography,
            (DIRECT_PREVIEW_WIDTH, DIRECT_PREVIEW_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return bev, homography

    @staticmethod
    def make_statistics(
        values: np.ndarray,
    ) -> dict[str, float]:
        """픽셀 간격의 평균, 표준편차, 최소·최대, 변동계수를 계산한다."""
        flat = values.reshape(-1).astype(np.float64)

        mean = float(np.mean(flat))
        std = float(np.std(flat))

        return {
            "mean_px": mean,
            "std_px": std,
            "min_px": float(np.min(flat)),
            "max_px": float(np.max(flat)),
            "cv_percent": (
                std / mean * 100.0
                if mean > 0.0
                else 0.0
            ),
        }

    @staticmethod
    def evaluate_uniformity(
        cv_percent: float,
    ) -> str:
        """
        BEV 비교용 경험적 평가.
        OpenCV의 공식 합격 기준은 아니다.
        """
        if cv_percent < 2.0:
            return "VERY GOOD"
        if cv_percent < 5.0:
            return "GOOD"
        if cv_percent < 10.0:
            return "CHECK"
        return "POOR"

    def detect_checkerboard_corners(
        self,
        image: np.ndarray,
        *,
        exhaustive: bool = False,
        use_classic_fallback: bool = True,
    ) -> np.ndarray | None:
        """SB 우선, classic 보조로 체커보드 내부 코너만 검출한다."""
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(
                f"체커보드 입력 영상 shape가 잘못됐습니다: {image.shape}"
            )
        equalized = cv2.equalizeHist(gray)

        found = False
        corners: np.ndarray | None = None
        if hasattr(cv2, "findChessboardCornersSB"):
            flags = cv2.CALIB_CB_NORMALIZE_IMAGE
            if exhaustive:
                flags |= (
                    cv2.CALIB_CB_EXHAUSTIVE
                    | cv2.CALIB_CB_ACCURACY
                )
            found, corners = cv2.findChessboardCornersSB(
                equalized,
                self.checkerboard_size,
                flags=flags,
            )
            if (
                (not found or corners is None)
                and not use_classic_fallback
            ):
                return None

        if not found or corners is None:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not exhaustive:
                flags |= cv2.CALIB_CB_FAST_CHECK
            found, corners = cv2.findChessboardCorners(
                equalized,
                self.checkerboard_size,
                flags=flags,
            )
            if found and corners is not None:
                criteria = (
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    40,
                    0.001,
                )
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria,
                )

        if not found or corners is None:
            return None

        columns, rows = self.checkerboard_size
        corners = np.asarray(
            corners,
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        if (
            len(corners) != columns * rows
            or not np.all(np.isfinite(corners))
        ):
            return None
        return corners

    def find_checkerboard(
        self,
        bev: np.ndarray,
        exhaustive: bool = False,
        use_classic_fallback: bool = True,
    ) -> tuple[
        bool,
        np.ndarray | None,
        dict[str, float | str] | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        """
        BEV에서 체커보드를 검출하고 한 칸 픽셀 수를 계산한다.

        드래그 중에는 빠른 SB 검출기만 사용한다. 저장 직전에는
        exhaustive=True와 기존 검출기 fallback으로 더 강하게 검사한다.
        """
        corners = self.detect_checkerboard_corners(
            bev,
            exhaustive=exhaustive,
            use_classic_fallback=use_classic_fallback,
        )
        if corners is None:
            return False, None, None, None, None

        columns, rows = self.checkerboard_size
        grid = corners.reshape(rows, columns, 2)

        horizontal_vectors = (
            grid[:, 1:, :] - grid[:, :-1, :]
        )
        vertical_vectors = (
            grid[1:, :, :] - grid[:-1, :, :]
        )
        horizontal = np.linalg.norm(horizontal_vectors, axis=2)
        vertical = np.linalg.norm(vertical_vectors, axis=2)

        horizontal_stats = self.make_statistics(
            horizontal
        )

        vertical_stats = self.make_statistics(
            vertical
        )

        mean_x = horizontal_stats["mean_px"]
        mean_y = vertical_stats["mean_px"]

        aspect_error_percent = (
            abs(mean_x - mean_y)
            / ((mean_x + mean_y) / 2.0)
            * 100.0
            if (mean_x + mean_y) > 0.0
            else 0.0
        )
        mean_horizontal_vector = np.mean(
            horizontal_vectors.reshape(-1, 2),
            axis=0,
        )
        mean_vertical_vector = np.mean(
            vertical_vectors.reshape(-1, 2),
            axis=0,
        )
        orthogonality_angle = self.vector_angle_degrees(
            mean_horizontal_vector,
            mean_vertical_vector,
        )
        orthogonality_error_degrees = abs(
            90.0 - orthogonality_angle
        )

        metrics: dict[str, float | str] = {
            "horizontal_mean_px": mean_x,
            "horizontal_std_px": horizontal_stats["std_px"],
            "horizontal_min_px": horizontal_stats["min_px"],
            "horizontal_max_px": horizontal_stats["max_px"],
            "horizontal_cv_percent": horizontal_stats["cv_percent"],
            "horizontal_evaluation": self.evaluate_uniformity(
                horizontal_stats["cv_percent"]
            ),
            "vertical_mean_px": mean_y,
            "vertical_std_px": vertical_stats["std_px"],
            "vertical_min_px": vertical_stats["min_px"],
            "vertical_max_px": vertical_stats["max_px"],
            "vertical_cv_percent": vertical_stats["cv_percent"],
            "vertical_evaluation": self.evaluate_uniformity(
                vertical_stats["cv_percent"]
            ),
            "aspect_error_percent": aspect_error_percent,
            "orthogonality_angle_degrees": orthogonality_angle,
            "orthogonality_error_degrees": orthogonality_error_degrees,
            "pixels_per_mm_x": mean_x / self.square_size_mm,
            "pixels_per_mm_y": mean_y / self.square_size_mm,
            "mm_per_pixel_x": (
                self.square_size_mm / mean_x
                if mean_x > 0.0
                else 0.0
            ),
            "mm_per_pixel_y": (
                self.square_size_mm / mean_y
                if mean_y > 0.0
                else 0.0
            ),
        }

        return (
            True,
            corners,
            metrics,
            horizontal,
            vertical,
        )

    def checkerboard_quality_errors(self) -> list[str]:
        """최종 metric BEV가 정사각 격자 품질 기준을 통과하는지 검사한다."""
        if not self.checker_found or self.checker_metrics is None:
            return ["체커보드가 검출되지 않았습니다."]

        metrics = self.checker_metrics
        errors: list[str] = []
        aspect_error = float(metrics["aspect_error_percent"])
        horizontal_cv = float(metrics["horizontal_cv_percent"])
        vertical_cv = float(metrics["vertical_cv_percent"])
        orthogonality_error = float(
            metrics["orthogonality_error_degrees"]
        )

        if aspect_error > MAX_FINAL_ASPECT_ERROR_PERCENT:
            errors.append(
                "한 칸의 가로/세로 크기 차이 "
                f"{aspect_error:.2f}% > "
                f"{MAX_FINAL_ASPECT_ERROR_PERCENT:.1f}%"
            )
        if horizontal_cv > MAX_FINAL_SPACING_CV_PERCENT:
            errors.append(
                "가로 간격 CV "
                f"{horizontal_cv:.2f}% > "
                f"{MAX_FINAL_SPACING_CV_PERCENT:.1f}%"
            )
        if vertical_cv > MAX_FINAL_SPACING_CV_PERCENT:
            errors.append(
                "세로 간격 CV "
                f"{vertical_cv:.2f}% > "
                f"{MAX_FINAL_SPACING_CV_PERCENT:.1f}%"
            )
        if (
            orthogonality_error
            > MAX_FINAL_ORTHOGONALITY_ERROR_DEG
        ):
            errors.append(
                "격자 직각 오차 "
                f"{orthogonality_error:.2f}° > "
                f"{MAX_FINAL_ORTHOGONALITY_ERROR_DEG:.1f}°"
            )
        if self.bev_geometry is not None:
            expected_square_pixels = (
                self.bev_geometry.pixels_per_mm
                * self.square_size_mm
            )
            measured_square_pixels = (
                float(metrics["horizontal_mean_px"])
                + float(metrics["vertical_mean_px"])
            ) / 2.0
            scale_error_percent = (
                abs(measured_square_pixels - expected_square_pixels)
                / expected_square_pixels
                * 100.0
            )
            if scale_error_percent > MAX_FINAL_SCALE_ERROR_PERCENT:
                errors.append(
                    "실제 mm 배율 오차 "
                    f"{scale_error_percent:.2f}% > "
                    f"{MAX_FINAL_SCALE_ERROR_PERCENT:.1f}%"
                )
        return errors

    def update_checkerboard_if_needed(
        self,
        bev: np.ndarray,
        force: bool = False,
    ) -> None:
        """
        ROI가 바뀌었을 때 체커보드 검사를 갱신한다.
        드래그 중에는 일정 시간 간격으로만 수행해 버벅임을 줄인다.
        """
        now = time.monotonic()

        if not force:
            if not self.checker_dirty:
                return

            if (
                now - self.last_detection_time
                < DETECTION_INTERVAL_SEC
            ):
                return

        (
            self.checker_found,
            self.checker_corners,
            self.checker_metrics,
            self.horizontal_spacing,
            self.vertical_spacing,
        ) = self.find_checkerboard(
            bev,
            exhaustive=force,
            use_classic_fallback=force,
        )

        self.last_detection_time = time.monotonic()
        self.checker_dirty = False

    def draw_checkerboard_result(
        self,
        bev: np.ndarray,
    ) -> np.ndarray:
        """체커보드 검출 및 측정값을 BEV 영상 위에 표시한다."""
        result = bev.copy()

        if (
            self.checker_found
            and self.checker_corners is not None
            and self.checker_metrics is not None
        ):
            cv2.drawChessboardCorners(
                result,
                self.checkerboard_size,
                self.checker_corners,
                True,
            )

            metrics = self.checker_metrics
            quality_errors = self.checkerboard_quality_errors()

            lines = [
                (
                    "Checkerboard: DETECTED | "
                    f"QUALITY {'PASS' if not quality_errors else 'FAIL'}"
                ),
                (
                    f"Physical square: {self.square_size_mm:.1f} mm | "
                    "target X:Y = 1.0000"
                ),
                (
                    f"X: {float(metrics['horizontal_mean_px']):.2f} px/square "
                    f"CV {float(metrics['horizontal_cv_percent']):.2f}% "
                    f"[{metrics['horizontal_evaluation']}]"
                ),
                (
                    f"Y: {float(metrics['vertical_mean_px']):.2f} px/square "
                    f"CV {float(metrics['vertical_cv_percent']):.2f}% "
                    f"[{metrics['vertical_evaluation']}]"
                ),
                (
                    f"X/Y square difference: "
                    f"{float(metrics['aspect_error_percent']):.2f}%"
                ),
                (
                    "Grid orthogonality error: "
                    f"{float(metrics['orthogonality_error_degrees']):.2f} deg"
                ),
            ]

            text_color = (
                (0, 255, 0)
                if not quality_errors
                else (0, 165, 255)
            )

        else:
            lines = [
                "Checkerboard: NOT DETECTED",
                (
                    f"Expected internal corners: "
                    f"{self.checkerboard_size[0]} x "
                    f"{self.checkerboard_size[1]}"
                ),
                "Move or enlarge the checkerboard inside the BEV ROI.",
            ]

            text_color = (0, 0, 255)

        panel_height = 30 * len(lines) + 10
        overlay = result.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (result.shape[1] - 1, panel_height),
            (0, 0, 0),
            -1,
        )

        result = cv2.addWeighted(
            overlay,
            0.60,
            result,
            0.40,
            0,
        )

        for index, text in enumerate(lines):
            cv2.putText(
                result,
                text,
                (12, 28 + index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.57,
                text_color,
                2,
                cv2.LINE_AA,
            )

        return result

    def save_spacing_csv(self) -> None:
        """체커보드 모든 인접 코너 간격을 CSV로 저장한다."""
        if (
            self.horizontal_spacing is None
            or self.vertical_spacing is None
        ):
            raise RuntimeError("저장할 체커보드 간격 데이터가 없습니다.")

        output = io.StringIO(newline="")
        writer = csv.writer(output)

        writer.writerow([
            "direction",
            "row",
            "column",
            "spacing_px",
        ])

        for row in range(
            self.horizontal_spacing.shape[0]
        ):
            for column in range(
                self.horizontal_spacing.shape[1]
            ):
                writer.writerow([
                    "horizontal",
                    row,
                    column,
                    float(
                        self.horizontal_spacing[
                            row,
                            column,
                        ]
                    ),
                ])

        for row in range(
            self.vertical_spacing.shape[0]
        ):
            for column in range(
                self.vertical_spacing.shape[1]
            ):
                writer.writerow([
                    "vertical",
                    row,
                    column,
                    float(
                        self.vertical_spacing[
                            row,
                            column,
                        ]
                    ),
                ])

        atomic_write_text(
            OUTPUT_SPACING_CSV,
            output.getvalue(),
            encoding="utf-8-sig",
        )

    def save_results(
        self,
        bev: np.ndarray,
        analyzed_bev: np.ndarray,
        homography: np.ndarray,
    ) -> None:
        """BEV 좌표, 미리보기, 체커보드 측정 결과를 저장한다."""
        if (
            not self.checker_found
            or self.checker_metrics is None
            or self.horizontal_spacing is None
            or self.vertical_spacing is None
        ):
            raise RuntimeError(
                "체커보드 10x7 내부 코너가 검출된 결과만 저장할 수 있습니다."
            )

        quality_errors = self.checkerboard_quality_errors()
        if quality_errors:
            raise RuntimeError(
                "최종 BEV 품질 기준을 통과하지 못했습니다:\n- "
                + "\n- ".join(quality_errors)
            )
        if self.bev_geometry is None or self.plane_model is None:
            raise RuntimeError("저장할 metric BEV 기하가 없습니다.")

        geometry = self.bev_geometry
        plane_model = self.plane_model
        source_points = np.asarray(
            geometry.effective_src_points,
            dtype=np.float32,
        )
        selected_source_points = np.asarray(
            geometry.selected_src_points,
            dtype=np.float32,
        )
        destination_points = np.asarray(
            geometry.dst_points,
            dtype=np.float32,
        )
        if (
            bev.shape[1] != geometry.warp_width
            or bev.shape[0] != geometry.warp_height
        ):
            raise ValueError(
                "저장할 BEV 영상 크기와 metric 기하가 다릅니다: "
                f"{bev.shape[1]}x{bev.shape[0]} != "
                f"{geometry.warp_width}x{geometry.warp_height}"
            )

        homography = np.asarray(homography, dtype=np.float64)
        expected_homography = cv2.getPerspectiveTransform(
            source_points,
            destination_points,
        )
        if (
            homography.shape != (3, 3)
            or not np.all(np.isfinite(homography))
            or abs(float(homography[2, 2])) < 1.0e-12
            or abs(float(expected_homography[2, 2])) < 1.0e-12
        ):
            raise ValueError("저장 homography가 유효하지 않습니다.")
        projected = self.transform_points(
            homography,
            source_points,
        )
        if float(np.max(np.abs(
            projected - destination_points
        ))) > 0.05:
            raise ValueError(
                "저장 좌표와 homography가 일치하지 않습니다."
            )

        text_lines = [
            "# BEV point order: Left-Bottom, Right-Bottom, "
            "Left-Top, Right-Top",
            "# Coordinate space: full undistorted camera image",
            "# Selection mode: independent four points + metric plane fit",
            "# selected: raw user clicks; effective: applied rectangle",
            f"# ROS topic: {self.topic}",
            f"# ROS transport: {self.transport}",
            f"# Calibration: {self.calibration_path.as_posix()}",
            f"# Calibration size: {self.image_width} x {self.image_height}",
            (
                f"# Physical size: {geometry.physical_width_mm:.3f} x "
                f"{geometry.physical_height_mm:.3f} mm"
            ),
            (
                f"# Warp size: {geometry.warp_width} x "
                f"{geometry.warp_height}"
            ),
            f"# Scale: {geometry.pixels_per_mm:.8f} px/mm",
            "",
            "# Selected source points",
        ]
        for name, point in zip(
            self.point_names,
            selected_source_points,
        ):
            text_lines.append(
                f"{float(point[0]):.3f}, {float(point[1]):.3f}  # {name}"
            )
        text_lines.extend(["", "# Effective source points"])
        for name, point in zip(self.point_names, source_points):
            text_lines.append(
                f"{float(point[0]):.6f}, "
                f"{float(point[1]):.6f}  # {name}"
            )

        report = {
            "checkerboard_detected": True,
            "checkerboard_quality_passed": True,
            "checkerboard_internal_corners": {
                "columns": self.checkerboard_size[0],
                "rows": self.checkerboard_size[1],
            },
            "square_size_mm": self.square_size_mm,
            "input_topic": self.topic,
            "input_transport": self.transport,
            "camera_frame_id": (
                self.node.latest_frame_id if self.node else ""
            ),
            "camera_encoding": (
                self.node.latest_encoding if self.node else ""
            ),
            "calibration_file": self.calibration_path.as_posix(),
            "calibration_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "coordinate_space": "full_undistorted_camera_image",
            "bev_output_size": {
                "width": geometry.warp_width,
                "height": geometry.warp_height,
            },
            "bev_physical_size_mm": {
                "width": geometry.physical_width_mm,
                "height": geometry.physical_height_mm,
            },
            "bev_physical_aspect_ratio": (
                geometry.physical_width_mm
                / geometry.physical_height_mm
            ),
            "pixels_per_mm": geometry.pixels_per_mm,
            "src_points_order": [
                "Left-Bottom",
                "Right-Bottom",
                "Left-Top",
                "Right-Top",
            ],
            "src_points": source_points.tolist(),
            "selected_src_points": selected_source_points.tolist(),
            "selected_metric_points_mm": (
                geometry.selected_metric_points_mm.tolist()
            ),
            "effective_metric_points_mm": (
                geometry.effective_metric_points_mm.tolist()
            ),
            "plane_fit": {
                "reprojection_rms_px": (
                    plane_model.reprojection_rms_px
                ),
                "inlier_ratio": plane_model.inlier_ratio,
                "checker_mean_spacing_px": (
                    plane_model.checker_mean_spacing_px
                ),
                "checker_area_px2": (
                    plane_model.checker_area_px2
                ),
            },
            "rectangle_fit": {
                "snap_rms_mm": geometry.snap_rms_mm,
                "snap_max_mm": geometry.snap_max_mm,
                "horizontal_parallel_error_deg": (
                    geometry.horizontal_parallel_error_deg
                ),
                "vertical_parallel_error_deg": (
                    geometry.vertical_parallel_error_deg
                ),
                "raw_orthogonality_error_deg": (
                    geometry.orthogonality_error_deg
                ),
                "opposite_width_error_ratio": (
                    geometry.opposite_width_error_ratio
                ),
                "opposite_height_error_ratio": (
                    geometry.opposite_height_error_ratio
                ),
            },
            "metrics": self.checker_metrics,
        }

        # 보조 결과가 모두 성공한 뒤 실제 ROS 설정 NPZ를 마지막에 교체한다.
        atomic_write_text(
            OUTPUT_TXT,
            "\n".join(text_lines) + "\n",
        )
        atomic_write_image(OUTPUT_PREVIEW, bev)
        atomic_write_image(OUTPUT_ANALYSIS_IMAGE, analyzed_bev)
        atomic_write_text(
            OUTPUT_REPORT_JSON,
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
        self.save_spacing_csv()
        # drive_pkg의 기존 bev_params_0731.npz와 키, shape, dtype을
        # 동일하게 유지한다. 검사/캘리브레이션 메타데이터는 위에서 저장한
        # JSON, CSV, TXT에 있으므로 주행용 NPZ에는 넣지 않는다.
        previous_bev_file = atomic_write_npz(
            self.output_npz,
            src_points=source_points,
            dst_points=destination_points,
            warp_w=np.int64(geometry.warp_width),
            warp_h=np.int64(geometry.warp_height),
        )

        print()
        print("=" * 70)
        print("저장 완료")
        print("=" * 70)
        print(f"BEV 설정 NPZ     : {self.output_npz}")
        if previous_bev_file is not None:
            print(f"이전 BEV 백업    : {previous_bev_file}")
        print(f"선택 좌표 TXT    : {OUTPUT_TXT.resolve()}")
        print(f"순수 BEV         : {OUTPUT_PREVIEW.resolve()}")
        print(f"분석 표시 BEV    : {OUTPUT_ANALYSIS_IMAGE.resolve()}")
        print(f"검사 보고서 JSON : {OUTPUT_REPORT_JSON.resolve()}")
        print(f"개별 간격 CSV    : {OUTPUT_SPACING_CSV.resolve()}")

        metrics = self.checker_metrics
        print()
        print(
            f"가로 한 칸: "
            f"{float(metrics['horizontal_mean_px']):.3f} px"
        )
        print(
            f"세로 한 칸: "
            f"{float(metrics['vertical_mean_px']):.3f} px"
        )
        print(
            f"가로 편차: "
            f"{float(metrics['horizontal_cv_percent']):.2f}%"
        )
        print(
            f"세로 편차: "
            f"{float(metrics['vertical_cv_percent']):.2f}%"
        )
        print(
            f"가로/세로 크기 차이: "
            f"{float(metrics['aspect_error_percent']):.2f}%"
        )
        print(
            f"격자 직각 오차: "
            f"{float(metrics['orthogonality_error_degrees']):.2f}°"
        )
        print(
            "실제 BEV 크기: "
            f"{geometry.physical_width_mm:.1f} x "
            f"{geometry.physical_height_mm:.1f} mm"
        )
        print(
            "출력 BEV 크기: "
            f"{geometry.warp_width} x {geometry.warp_height} px"
        )

        print("=" * 70)
        print(
            "[중요] 결과는 프로젝트 루트에 저장했습니다. "
            "lane_vision_pkg에는 아직 반영하지 않았습니다."
        )
        print(
            "640x480 camera_calibration.npz와 bev_params_y_auto.npz를 "
            "항상 한 세트로 반영하세요."
        )

    def reset_selection(
        self,
        *,
        clear_plane: bool = False,
    ) -> None:
        """선택 점과 BEV 검사 결과를 초기화한다."""
        self.src_points = []

        self.is_dragging = False
        self.active_point_index = None
        self.drag_original_point = None
        self.drag_original_points = None
        self.drag_added_new_point = False
        self.bev_geometry = None
        self.geometry_error_message = None
        self.mark_checker_dirty()
        if clear_plane:
            self.plane_model = None
            self.plane_error_message = None
            self.plane_detection_attempted = False

        print("[초기화] 네 점과 BEV 영역을 지웠습니다.")

        self.close_live_window()
        self.close_live_bev_window()

        try:
            cv2.destroyWindow(self.bev_window)
        except cv2.error:
            pass
        self.bev_window_open = False
        self.bev_window_image_shape = None

    def toggle_freeze(self) -> None:
        """실시간 영상과 정지 화면을 전환한다."""
        if not self.is_frozen:
            if self.latest_frame is None:
                print("[WARNING] 아직 카메라 프레임이 없습니다.")
                return

            self.frozen_frame = self.latest_frame.copy()
            self.is_frozen = True
            self.reset_selection(clear_plane=True)

            print("[화면 정지]")
            print(
                "LB, RB, LT, RT 순서로 네 점을 클릭하세요. "
                "RB의 y는 LB에, RT의 y는 LT에 자동 정렬됩니다. "
                "완료 후 각 점을 드래그해 수정할 수 있습니다."
            )
            self.ensure_plane_model(
                self.frozen_frame,
                force=False,
                strict=False,
            )

        else:
            self.is_frozen = False
            self.frozen_frame = None
            self.reset_selection(clear_plane=True)

            print("[실시간 영상 재개]")

    def save_current_configuration(self) -> None:
        """강한 재검출·품질 검사를 통과한 현재 설정만 저장한다."""
        if (
            not self.is_frozen
            or self.frozen_frame is None
            or self.is_dragging
            or len(self.src_points) != 4
        ):
            print(
                "[저장 안 함] 화면을 정지하고 "
                "LB, RB, LT, RT 네 점을 확정하세요."
            )
            return

        try:
            # 저장 순간의 정지 원본에서 체커보드 평면부터 강한 옵션으로
            # 다시 구한다. 이 과정에서 동적 출력 크기와 H도 함께 재계산된다.
            save_bev_data = self.make_bev(
                self.frozen_frame,
                strict=True,
                refresh_plane=True,
            )
            if save_bev_data is None:
                raise RuntimeError("유효한 metric BEV 영역이 없습니다.")
            save_bev, save_homography = save_bev_data

            self.update_checkerboard_if_needed(
                save_bev,
                force=True,
            )
            save_analyzed_bev = self.draw_checkerboard_result(
                save_bev
            )
            self.show_bev_window(save_analyzed_bev)

            quality_errors = self.checkerboard_quality_errors()
            if quality_errors:
                raise RuntimeError(
                    "최종 BEV 품질 검사 실패:\n  - "
                    + "\n  - ".join(quality_errors)
                )

            self.save_results(
                save_bev,
                save_analyzed_bev,
                save_homography,
            )
        except (ValueError, RuntimeError, OSError, cv2.error) as error:
            print(f"[저장 안 함] {error}")

    def run(self) -> None:
        """메인 실행 루프."""
        try:
            self.wait_for_first_frame()
            if self.node is None:
                raise RuntimeError("ROS 구독 노드가 없습니다.")

            print("=" * 75)
            print("ROS usb_cam 실제 비율 BEV 설정 + 체커보드 검사")
            print("=" * 75)
            print(f"입력 토픽             : {self.topic}")
            print(f"전송 방식             : {self.transport}")
            print(f"메시지 타입           : {self.node.message_type_name}")
            print("구독 QoS              : BEST_EFFORT, KEEP_LAST(1)")
            print(f"ROS 인코딩            : {self.node.latest_encoding}")
            print(
                f"ROS frame_id          : "
                f"{self.node.latest_frame_id or '(empty)'}"
            )
            print(f"캘리브레이션 파일     : {self.calibration_path}")
            print(
                f"캘리브레이션 해상도   : "
                f"{self.image_width} x {self.image_height}"
            )
            print(f"BEV 설정 저장 위치    : {self.output_npz}")
            print(
                "BEV 출력 크기         : 실제 비율로 자동 계산 "
                f"(긴 변 최대 {self.max_output_side}px)"
            )
            print(
                "목표 체커보드 배율     : "
                f"{self.target_square_pixels:g}px / square"
            )
            print(
                f"체커보드 내부 코너    : "
                f"{self.checkerboard_size[0]} x "
                f"{self.checkerboard_size[1]}"
            )
            print(
                f"체커보드 한 칸        : "
                f"{self.square_size_mm:g} mm"
            )
            print()
            print("사용 방법")
            print(
                "1. 10x7 체커보드를 BEV 영역과 같은 평평한 바닥에 놓기"
            )
            print("2. 체커보드 전체가 보일 때 SPACE로 화면 정지")
            print("3. LB, RB, LT, RT 순서로 클릭 (상·하단 y 자동 정렬)")
            print("4. 필요하면 점을 드래그해 수정 (짝점 y도 함께 정렬)")
            print("5. 정지 BEV에서 10x7 격자 QUALITY PASS 확인")
            print("   동시에 별도 창에서 실시간 원본/BEV 확인")
            print("6. S를 눌러 강한 재검사 후 저장")
            print()
            print("R     : 다시 그리기")
            print("우클릭: 마지막 점 되돌리기")
            print("SPACE : 정지/재개")
            print("S     : 강한 체커보드 재검사 후 저장")
            print("Q/ESC : 종료")
            print("=" * 75)

            while True:
                self.spin_topic_once()
                frame = self.read_undistorted_frame()

                if frame is None:
                    raise RuntimeError(
                        "usb_cam 프레임을 왜곡 보정하지 못했습니다."
                    )

                original_display = self.draw_original_overlay(
                    frame
                )

                live_frame = (
                    self.latest_frame
                    if self.is_frozen and self.latest_frame is not None
                    else frame
                )
                if len(self.src_points) == 4:
                    self.show_live_window(live_frame)
                else:
                    self.close_live_window()

                # 체커보드 검사 BEV는 점을 선택한 정지 사진만 사용한다.
                bev_data = self.make_bev(frame)
                if bev_data is None and len(self.src_points) == 4:
                    bev_data = self.make_direct_preview_bev(frame)

                if bev_data is not None:
                    bev, _ = bev_data

                    # 드래그 도중에는 고비용 코너 검출을 생략하고,
                    # 버튼을 놓은 직후 최종 위치에서 한 번만 갱신한다.
                    if not self.is_dragging:
                        self.update_checkerboard_if_needed(
                            bev,
                            force=False,
                        )

                    analyzed_bev = self.draw_checkerboard_result(
                        bev
                    )

                    self.show_bev_window(analyzed_bev)

                    live_bev_data = self.make_bev(live_frame)
                    if live_bev_data is None:
                        live_bev_data = self.make_direct_preview_bev(
                            live_frame
                        )
                    if live_bev_data is not None:
                        live_bev = live_bev_data[0].copy()
                        cv2.putText(
                            live_bev,
                            "LIVE BEV",
                            (15, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        self.show_live_bev_window(live_bev)
                    else:
                        self.close_live_bev_window()

                else:
                    bev = None
                    self.close_live_bev_window()
                    if self.bev_window_open:
                        try:
                            cv2.destroyWindow(self.bev_window)
                        except cv2.error:
                            pass
                        self.bev_window_open = False
                        self.bev_window_image_shape = None

                cv2.imshow(
                    self.original_window,
                    original_display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    ord("Q"),
                    27,
                ):
                    break

                if key == 32:
                    self.toggle_freeze()

                elif key in (
                    ord("r"),
                    ord("R"),
                ):
                    self.reset_selection()

                elif key in (
                    ord("s"),
                    ord("S"),
                ):
                    self.save_current_configuration()

                try:
                    if cv2.getWindowProperty(
                        self.original_window,
                        cv2.WND_PROP_VISIBLE,
                    ) < 1.0:
                        break
                except cv2.error:
                    pass

        finally:
            self.close()

    def close(self) -> None:
        """GUI와 ROS 구독 노드를 여러 번 호출해도 안전하게 정리한다."""
        had_node = self.node is not None

        try:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        finally:
            if self.node is not None:
                self.node.destroy_node()
                self.node = None

        if had_node:
            print("[종료] usb_cam 토픽 구독을 해제했습니다.")


def main(argv: list[str] | None = None) -> int:
    setup: WebcamBEVCheckerboardSetup | None = None
    initialized_here = False

    try:
        arguments = parse_arguments(argv)

        if rclpy is None:
            raise RuntimeError(
                "ROS 2 Python 모듈을 불러오지 못했습니다.\n"
                "다음 명령을 실행한 터미널에서 다시 시작하세요:\n"
                "  source /opt/ros/humble/setup.bash\n"
                f"원인: {ROS_IMPORT_ERROR}"
            )

        if not rclpy.ok():
            rclpy.init(args=[])
            initialized_here = True

        setup = WebcamBEVCheckerboardSetup(arguments)
        setup.run()

    except KeyboardInterrupt:
        print()
        print("[중단] 사용자가 BEV 설정을 중단했습니다.")
        return 130

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        KeyError,
        cv2.error,
        ValueError,
    ) as error:
        print()
        print("[오류]")
        print(error)
        return 1

    finally:
        if setup is not None:
            setup.close()

        if (
            initialized_here
            and rclpy is not None
            and rclpy.ok()
        ):
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
