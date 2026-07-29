#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import tempfile
import time

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
# 기본 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE_DIR = PROJECT_ROOT / "calibration_images"

DEFAULT_RAW_TOPIC = "/camera/lane/raw"
DEFAULT_COMPRESSED_TOPIC = "/camera/lane/raw/compressed"
DEFAULT_TRANSPORT = "compressed"
DEFAULT_TOPIC_TIMEOUT = 15.0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 내부 코너 수. 일반적으로 가로 11칸 x 세로 8칸 체커보드에 해당한다.
CHECKERBOARD_SIZE = (10, 7)
EXPECTED_CORNER_COUNT = CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1]

# SPACE 저장 품질 기준. S 강제 저장은 이 기준을 우회한다.
DEFAULT_MIN_SHARPNESS = 80.0
DEFAULT_MIN_POSE_CHANGE_PX = 20.0
DEFAULT_LIVE_DETECTION_INTERVAL_SEC = 0.2

WINDOW_NAME = "Calibration Image Capture"

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 usb_cam 토픽에서 640x480 체커보드 "
            "캘리브레이션 이미지를 저장합니다."
        )
    )

    parser.add_argument(
        "--transport",
        choices=("compressed", "raw"),
        default=DEFAULT_TRANSPORT,
        help=(
            "입력 영상 전송 방식. 640x480에서도 compressed 권장 "
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
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=(
            "저장 폴더. 상대경로는 utils 폴더 기준 "
            "(기본값: calibration_images)"
        ),
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=DEFAULT_MIN_SHARPNESS,
        help=(
            "SPACE 저장에 필요한 최소 Laplacian 선명도 "
            f"(기본값: {DEFAULT_MIN_SHARPNESS:g})"
        ),
    )
    parser.add_argument(
        "--min-pose-change",
        type=float,
        default=DEFAULT_MIN_POSE_CHANGE_PX,
        help=(
            "중복 구도 방지를 위한 코너 평균 이동량(px) "
            f"(기본값: {DEFAULT_MIN_POSE_CHANGE_PX:g})"
        ),
    )
    parser.add_argument(
        "--detection-interval",
        type=float,
        default=DEFAULT_LIVE_DETECTION_INTERVAL_SEC,
        help=(
            "실시간 체커보드 검출 간격(초). SPACE 저장 시에는 항상 "
            "현재 프레임을 정밀 재검사함 "
            f"(기본값: {DEFAULT_LIVE_DETECTION_INTERVAL_SEC:g})"
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
        not np.isfinite(arguments.min_sharpness)
        or arguments.min_sharpness < 0.0
    ):
        parser.error("--min-sharpness는 0 이상이어야 합니다.")

    if (
        not np.isfinite(arguments.min_pose_change)
        or arguments.min_pose_change < 0.0
    ):
        parser.error("--min-pose-change는 0 이상이어야 합니다.")

    if (
        not np.isfinite(arguments.detection_interval)
        or arguments.detection_interval < 0.0
    ):
        parser.error("--detection-interval은 0 이상이어야 합니다.")

    if not arguments.save_dir.is_absolute():
        arguments.save_dir = PROJECT_ROOT / arguments.save_dir

    arguments.save_dir = arguments.save_dir.resolve()

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
    elif encoding in ("yuv422_yuy2", "yuyv"):
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

    def __init__(self, topic: str, transport: str) -> None:
        super().__init__(
            f"calibration_image_capture_{os.getpid()}"
        )

        self.topic = topic
        self.transport = transport
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
            frame.shape[1] != FRAME_WIDTH
            or frame.shape[0] != FRAME_HEIGHT
        ):
            self.fatal_error = (
                "usb_cam 토픽 해상도가 캘리브레이션 설정과 다릅니다: "
                f"{frame.shape[1]}x{frame.shape[0]} != "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
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


def find_checkerboard_corners(
    frame: np.ndarray,
    *,
    exhaustive: bool = True,
    use_classic_fallback: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """
    SB 검출기를 먼저 사용하고 필요할 때만 기존 검출기로 재시도한다.

    실시간 미리보기에서는 exhaustive와 느린 기존 fallback을 끈다.
    저장 직전과 기존 사진 검사에서는 둘 다 켜 검출 신뢰도를 유지한다.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if exhaustive:
            sb_flags |= cv2.CALIB_CB_EXHAUSTIVE

        found, corners = cv2.findChessboardCornersSB(
            equalized,
            CHECKERBOARD_SIZE,
            flags=sb_flags,
        )

        if found and corners is not None:
            corners = np.asarray(
                corners,
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            if (
                len(corners) == EXPECTED_CORNER_COUNT
                and np.all(np.isfinite(corners))
            ):
                return True, corners

        if not use_classic_fallback:
            return False, None

    normal_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(
        equalized,
        CHECKERBOARD_SIZE,
        flags=normal_flags,
    )

    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    refined = np.asarray(
        refined,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    if (
        len(refined) != EXPECTED_CORNER_COUNT
        or not np.all(np.isfinite(refined))
    ):
        return False, None

    return True, refined


def calculate_board_sharpness(
    frame: np.ndarray,
    corners: np.ndarray,
) -> float:
    """체커보드 주변 영역의 Laplacian 분산으로 선명도를 계산한다."""
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    height, width = frame.shape[:2]

    x_min = max(0, int(np.floor(np.min(points[:, 0]))) - 20)
    x_max = min(width, int(np.ceil(np.max(points[:, 0]))) + 21)
    y_min = max(0, int(np.floor(np.min(points[:, 1]))) - 20)
    y_max = min(height, int(np.ceil(np.max(points[:, 1]))) + 21)

    region = frame[y_min:y_max, x_min:x_max]
    if region.size == 0:
        return 0.0

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pose_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    코너 순서가 180도 뒤집히는 경우까지 고려한 평균 코너 이동량을 구한다.
    """
    first_points = np.asarray(
        first,
        dtype=np.float32,
    ).reshape(-1, 2)
    second_points = np.asarray(
        second,
        dtype=np.float32,
    ).reshape(-1, 2)

    if first_points.shape != second_points.shape:
        return float("inf")

    direct = float(
        np.mean(
            np.linalg.norm(
                first_points - second_points,
                axis=1,
            )
        )
    )
    reversed_order = float(
        np.mean(
            np.linalg.norm(
                first_points - second_points[::-1],
                axis=1,
            )
        )
    )

    return min(direct, reversed_order)


def minimum_saved_pose_distance(
    corners: np.ndarray,
    saved_poses: list[np.ndarray],
) -> float:
    if not saved_poses:
        return float("inf")

    return min(
        pose_distance(corners, saved)
        for saved in saved_poses
    )


def save_unique_png(
    save_dir: Path,
    prefix: str,
    frame: np.ndarray,
) -> Path:
    """
    완성된 무손실 PNG를 기존 파일을 덮어쓰지 않고 원자적으로 공개한다.

    임시 파일을 먼저 완성하고 fsync한 뒤 hard link를 생성한다. 따라서
    동시 실행 중 이름이 충돌해도 기존 파일을 보존하며, 저장 중 전원이
    끊겨도 최종 이름에 0바이트 또는 부분 PNG가 나타나지 않는다.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    output_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{prefix}_",
            suffix=".png",
            dir=save_dir,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        saved = cv2.imwrite(
            str(temporary_path),
            frame,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )
        if not saved:
            raise OSError("PNG 임시 파일을 저장하지 못했습니다.")

        temporary_path.chmod(0o644)

        with temporary_path.open("rb") as temporary_file:
            os.fsync(temporary_file.fileno())

        for sequence in range(1000):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            suffix = "" if sequence == 0 else f"_{sequence:03d}"
            candidate = (
                save_dir
                / f"{prefix}_{timestamp}{suffix}.png"
            )

            try:
                os.link(temporary_path, candidate)
            except FileExistsError:
                continue

            output_path = candidate
            break

        if output_path is None:
            raise RuntimeError("고유한 이미지 파일명을 만들지 못했습니다.")

        temporary_path.unlink()
        temporary_path = None

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    if output_path is None:
        raise RuntimeError("이미지 저장 결과 경로가 없습니다.")

    return output_path


def count_existing_images(
    save_dir: Path,
    prefix: str,
) -> int:
    if not save_dir.is_dir():
        return 0

    return sum(
        1
        for path in save_dir.iterdir()
        if (
            path.is_file()
            and path.name.startswith(f"{prefix}_")
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    )


def load_existing_checkerboard_poses(
    save_dir: Path,
) -> list[np.ndarray]:
    """
    이전 실행에서 저장한 640x480 검출 사진의 구도를 다시 읽는다.

    같은 폴더에서 프로그램을 재실행해도 거의 같은 구도를 중복 저장하지
    않도록 하기 위한 데이터다. 읽기 실패나 다른 해상도 사진은 건너뛴다.
    """
    if not save_dir.is_dir():
        return []

    saved_poses: list[np.ndarray] = []
    image_paths = sorted(
        (
            path
            for path in save_dir.iterdir()
            if (
                path.is_file()
                and path.name.startswith("checkerboard_")
                and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
        ),
        key=lambda path: path.name.casefold(),
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        if (
            image.shape[1] != FRAME_WIDTH
            or image.shape[0] != FRAME_HEIGHT
        ):
            continue

        found, corners = find_checkerboard_corners(image)
        if found and corners is not None:
            saved_poses.append(corners)

    return saved_poses


def draw_status_overlay(
    frame: np.ndarray,
    found: bool,
    corners: np.ndarray | None,
    sharpness: float,
    min_sharpness: float,
    nearest_pose_distance: float,
    min_pose_change: float,
    detected_saved_count: int,
    manual_saved_count: int,
    input_fps: float,
) -> np.ndarray:
    display = frame.copy()

    if found and corners is not None:
        cv2.drawChessboardCorners(
            display,
            CHECKERBOARD_SIZE,
            corners,
            True,
        )

    sharp_enough = found and sharpness >= min_sharpness
    pose_changed = (
        not np.isfinite(nearest_pose_distance)
        or nearest_pose_distance >= min_pose_change
    )
    ready = found and sharp_enough and pose_changed

    if not found:
        status = "Checkerboard not detected"
        color = (0, 0, 255)
    elif not sharp_enough:
        status = "Detected, but image is blurry"
        color = (0, 165, 255)
    elif not pose_changed:
        status = "Detected, but pose is too similar"
        color = (0, 165, 255)
    else:
        status = "Ready - press SPACE to save"
        color = (0, 255, 0)

    lines = [
        (status, color),
        (
            f"Resolution {FRAME_WIDTH}x{FRAME_HEIGHT} | "
            f"App {input_fps:.1f} FPS | "
            f"Corners {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]}",
            (255, 255, 255),
        ),
        (
            f"Sharpness {sharpness:.1f}/{min_sharpness:.1f} | "
            f"Saved detected {detected_saved_count} | "
            f"manual {manual_saved_count}",
            (255, 255, 255),
        ),
    ]

    if (
        found
        and np.isfinite(nearest_pose_distance)
        and min_pose_change > 0.0
    ):
        lines.append(
            (
                f"Pose change {nearest_pose_distance:.1f}/"
                f"{min_pose_change:.1f} px",
                (255, 255, 255),
            )
        )

    for index, (text, text_color) in enumerate(lines):
        cv2.putText(
            display,
            text,
            (20, 35 + index * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        "SPACE: quality save | S: force save | Q/ESC: quit",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if ready:
        cv2.rectangle(
            display,
            (2, 2),
            (display.shape[1] - 3, display.shape[0] - 3),
            (0, 255, 0),
            3,
        )

    return display


def run_capture(arguments: argparse.Namespace) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python 모듈을 불러오지 못했습니다.\n"
            "다음 명령을 실행한 터미널에서 다시 시작하세요:\n"
            "  source /opt/ros/humble/setup.bash\n"
            f"원인: {ROS_IMPORT_ERROR}"
        )

    save_dir: Path = arguments.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_checkerboard_files = count_existing_images(
        save_dir,
        "checkerboard",
    )
    saved_poses = load_existing_checkerboard_poses(save_dir)
    node: UsbCamTopicSubscriber | None = None
    initialized_here = False

    try:
        if not rclpy.ok():
            rclpy.init(args=[])
            initialized_here = True

        node = UsbCamTopicSubscriber(
            arguments.topic,
            arguments.transport,
        )

        deadline = time.monotonic() + arguments.topic_timeout
        while node.latest_frame is None:
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.fatal_error is not None:
                raise RuntimeError(node.fatal_error)

            if time.monotonic() >= deadline:
                publisher_count = node.count_publishers(
                    arguments.topic
                )
                raise RuntimeError(
                    "usb_cam 프레임을 받지 못했습니다.\n"
                    f"토픽: {arguments.topic}\n"
                    f"transport: {arguments.transport}\n"
                    f"발견한 publisher 수: {publisher_count}\n"
                    "usb_cam launch와 토픽 이름을 확인하세요:\n"
                    f"  ros2 topic info {arguments.topic} --verbose"
                )

        first_frame = node.latest_frame
        if first_frame is None:
            raise RuntimeError("첫 ROS 이미지 프레임이 없습니다.")

        detected_saved_count = len(saved_poses)
        manual_saved_count = count_existing_images(
            save_dir,
            "manual",
        )
        session_detected_count = 0
        session_manual_count = 0

        print("=" * 72)
        print("ROS 2 usb_cam 체커보드 캘리브레이션 사진 촬영")
        print("=" * 72)
        print(f"입력 토픽           : {arguments.topic}")
        print(f"전송 방식           : {arguments.transport}")
        print(f"메시지 타입         : {node.message_type_name}")
        print("구독 QoS            : BEST_EFFORT, KEEP_LAST(1)")
        print(
            f"첫 프레임           : "
            f"{first_frame.shape[1]} x {first_frame.shape[0]}"
        )
        print(f"ROS 인코딩          : {node.latest_encoding}")
        print(f"ROS frame_id        : {node.latest_frame_id or '(empty)'}")
        print(
            f"체커보드 내부 코너  : "
            f"{CHECKERBOARD_SIZE[0]} x {CHECKERBOARD_SIZE[1]}"
        )
        print(f"저장 폴더           : {save_dir}")
        print(f"기존 검출 파일 수   : {existing_checkerboard_files}")
        print(f"기존 유효 구도 수   : {len(saved_poses)}")
        print("SPACE: 검출·선명도·구도 검사 후 PNG 저장")
        print("S    : 품질 검사와 관계없이 원본 PNG 강제 저장")
        print("       강제 저장 사진은 후속 캘리브레이션에서 제외될 수 있음")
        print("Q/ESC: 종료")
        print("=" * 72)

        skipped_existing = (
            existing_checkerboard_files - len(saved_poses)
        )
        if skipped_existing > 0:
            print(
                f"[경고] 기존 검출 파일 중 {skipped_existing}개는 "
                "640x480·10x7 조건을 충족하지 않아 "
                "중복 구도 검사에서 제외했습니다."
            )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

        last_processed_serial = 0
        frame: np.ndarray | None = None
        found = False
        corners: np.ndarray | None = None
        sharpness = 0.0
        nearest_pose_distance = float("inf")
        last_live_detection_time = float("-inf")

        while True:
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.fatal_error is not None:
                raise RuntimeError(node.fatal_error)

            if (
                node.last_arrival_time > 0.0
                and time.monotonic() - node.last_arrival_time
                > arguments.topic_timeout
            ):
                raise RuntimeError(
                    "usb_cam 토픽 프레임 수신이 중단됐습니다: "
                    f"{arguments.topic}"
                )

            if node.frame_serial != last_processed_serial:
                frame = node.latest_frame
                if frame is None:
                    continue

                now = time.monotonic()
                detection_updated = (
                    now - last_live_detection_time
                    >= arguments.detection_interval
                )
                if detection_updated:
                    found, corners = find_checkerboard_corners(
                        frame,
                        exhaustive=False,
                        use_classic_fallback=False,
                    )

                    if found and corners is not None:
                        sharpness = calculate_board_sharpness(
                            frame,
                            corners,
                        )
                        nearest_pose_distance = (
                            minimum_saved_pose_distance(
                                corners,
                                saved_poses,
                            )
                        )
                    else:
                        sharpness = 0.0
                        nearest_pose_distance = float("inf")

                    last_live_detection_time = time.monotonic()

                display = draw_status_overlay(
                    frame=frame,
                    found=found,
                    corners=(
                        corners
                        if detection_updated
                        else None
                    ),
                    sharpness=sharpness,
                    min_sharpness=arguments.min_sharpness,
                    nearest_pose_distance=nearest_pose_distance,
                    min_pose_change=arguments.min_pose_change,
                    detected_saved_count=detected_saved_count,
                    manual_saved_count=manual_saved_count,
                    input_fps=node.measured_fps(),
                )
                cv2.imshow(WINDOW_NAME, display)
                last_processed_serial = node.frame_serial

            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                if frame is None:
                    print("[저장 안 함] 아직 ROS 프레임이 없습니다.")
                    continue

                # 미리보기는 속도를 위해 빠른 검출만 사용한다. 저장할
                # 순간에는 같은 원본 프레임을 강한 옵션으로 다시 검사한다.
                found, corners = find_checkerboard_corners(
                    frame,
                    exhaustive=True,
                    use_classic_fallback=True,
                )

                if found and corners is not None:
                    sharpness = calculate_board_sharpness(
                        frame,
                        corners,
                    )
                    nearest_pose_distance = (
                        minimum_saved_pose_distance(
                            corners,
                            saved_poses,
                        )
                    )
                else:
                    sharpness = 0.0
                    nearest_pose_distance = float("inf")

                if not found or corners is None:
                    print(
                        "[저장 안 함] 체커보드 10x7 내부 코너를 "
                        "검출하지 못했습니다."
                    )
                elif sharpness < arguments.min_sharpness:
                    print(
                        "[저장 안 함] 이미지가 흐립니다: "
                        f"{sharpness:.1f} < "
                        f"{arguments.min_sharpness:.1f}"
                    )
                elif (
                    np.isfinite(nearest_pose_distance)
                    and nearest_pose_distance
                    < arguments.min_pose_change
                ):
                    print(
                        "[저장 안 함] 이전 저장 구도와 너무 비슷합니다: "
                        f"{nearest_pose_distance:.1f} px"
                    )
                else:
                    output_path = save_unique_png(
                        save_dir,
                        "checkerboard",
                        frame,
                    )
                    saved_poses.append(corners.copy())
                    detected_saved_count += 1
                    session_detected_count += 1
                    print(
                        f"[저장 완료] {output_path.name} "
                        f"(선명도 {sharpness:.1f})"
                    )

            elif key in (ord("s"), ord("S")):
                if frame is None:
                    print("[저장 안 함] 아직 ROS 프레임이 없습니다.")
                else:
                    output_path = save_unique_png(
                        save_dir,
                        "manual",
                        frame,
                    )
                    manual_saved_count += 1
                    session_manual_count += 1
                    print(
                        f"[강제 저장 완료] {output_path.name} "
                        "(품질 기준 미적용)"
                    )

            elif key in (ord("q"), ord("Q"), 27):
                break

            try:
                if cv2.getWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                ) < 1.0:
                    break
            except cv2.error:
                pass

        print()
        print("=" * 72)
        print("촬영 종료")
        print(f"이번 실행 검출 저장 : {session_detected_count}")
        print(f"이번 실행 강제 저장 : {session_manual_count}")
        print(f"유효 검출 구도 누적 : {detected_saved_count}")
        print(f"강제 저장 파일 누적 : {manual_saved_count}")
        print(f"저장 위치         : {save_dir}")
        print("=" * 72)

    finally:
        try:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

            if node is not None:
                node.destroy_node()

        finally:
            if initialized_here and rclpy.ok():
                rclpy.shutdown()


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        run_capture(arguments)

    except KeyboardInterrupt:
        print()
        print("[중단] 사용자가 촬영을 중단했습니다.")
        return 130

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as error:
        print()
        print("[오류]")
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
