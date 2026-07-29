#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS 2 실시간 카메라용 평행 가이드 BEV 설정 및 이미지 추출 도구."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


# ---------------------------------------------------------------------------
# 자주 조정하는 기본 파라미터
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_TOPIC = "/image_raw/compressed"
DEFAULT_CALIBRATION = PROJECT_ROOT / "camera_calibration.npz"
DEFAULT_BEV_PARAMS = (
    PROJECT_ROOT / "src" / "drive_pkg" / "resource" / "bev_params_0729.npz"
)
DEFAULT_POINTS_TXT = SCRIPT_DIR / "selected_bev_src_points.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "roboflow_bev_images"

DEFAULT_WARP_WIDTH = 640
DEFAULT_WARP_HEIGHT = 640
DEFAULT_CANVAS_WIDTH = 1000
DEFAULT_CANVAS_HEIGHT = 800
DEFAULT_EXPORT_COUNT = 1000
DEFAULT_SAVE_EVERY = 1
DEFAULT_JPEG_QUALITY = 95
DEFAULT_CHECKERBOARD_COLUMNS = 10
DEFAULT_CHECKERBOARD_ROWS = 7
DEFAULT_CHECKERBOARD_DETECT_EVERY = 3
GUI_PERIOD_SECONDS = 0.03
LOW_LATENCY_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 압축 카메라 토픽을 실시간으로 받아 밑변 평행 가이드를 "
            "보면서 네 점으로 BEV를 설정하고 Roboflow용 JPG를 저장합니다."
        )
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="sensor_msgs/msg/Image 토픽을 구독합니다. 기본값은 CompressedImage입니다.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="카메라 왜곡 보정을 적용하지 않습니다.",
    )
    parser.add_argument("--bev-params", type=Path, default=DEFAULT_BEV_PARAMS)
    parser.add_argument("--points-txt", type=Path, default=DEFAULT_POINTS_TXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warp-width", type=int, default=DEFAULT_WARP_WIDTH)
    parser.add_argument("--warp-height", type=int, default=DEFAULT_WARP_HEIGHT)
    parser.add_argument("--canvas-width", type=int, default=DEFAULT_CANVAS_WIDTH)
    parser.add_argument("--canvas-height", type=int, default=DEFAULT_CANVAS_HEIGHT)
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_EXPORT_COUNT,
        help="'s' 입력 후 저장할 BEV 이미지 수",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="몇 번째 수신 프레임마다 한 장을 저장할지 지정",
    )
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument(
        "--checkerboard-columns",
        type=int,
        default=DEFAULT_CHECKERBOARD_COLUMNS,
        help="체커보드 가로 내부 코너 개수",
    )
    parser.add_argument(
        "--checkerboard-rows",
        type=int,
        default=DEFAULT_CHECKERBOARD_ROWS,
        help="체커보드 세로 내부 코너 개수",
    )
    parser.add_argument(
        "--checkerboard-every",
        type=int,
        default=DEFAULT_CHECKERBOARD_DETECT_EVERY,
        help="지연 감소를 위해 몇 프레임마다 체커보드를 검출할지 지정",
    )
    return parser.parse_args()


def scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0])


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

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(
            data["distortion_coefficients"], dtype=np.float64
        ).reshape(-1)
        new_camera_matrix = np.asarray(
            data["new_camera_matrix"], dtype=np.float64
        )
        width = scalar_int(data, "image_width")
        height = scalar_int(data, "image_height")

    if (
        camera_matrix.shape != (3, 3)
        or new_camera_matrix.shape != (3, 3)
        or distortion.size not in (4, 5, 8, 12, 14)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("카메라 캘리브레이션 값이 유효하지 않습니다.")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return Calibration(width=width, height=height, map_x=map_x, map_y=map_y)


def destination_points(width: int, height: int) -> np.ndarray:
    # src와 동일한 순서: 좌하, 우하, 좌상, 우상
    return np.float32(
        [
            [0, height],
            [width, height],
            [0, 0],
            [width, 0],
        ]
    )


def detect_checkerboard(
    frame: np.ndarray,
    columns: int,
    rows: int,
) -> tuple[bool, np.ndarray | None]:
    """영상에서 지정한 개수의 체커보드 내부 코너를 검출한다."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern_size = (columns, rows)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return False, None

    refined = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        ),
    )
    return True, np.asarray(refined, dtype=np.float32).reshape(-1, 1, 2)


def decode_compressed_message(message: CompressedImage) -> np.ndarray:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("CompressedImage 디코딩에 실패했습니다.")
    return frame


def decode_raw_message(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    height = int(message.height)
    width = int(message.width)
    row_step = int(message.step)
    data = np.frombuffer(message.data, dtype=np.uint8)

    if encoding in {"bgr8", "rgb8"}:
        required_step = width * 3
        if row_step < required_step or data.size < row_step * height:
            raise ValueError(f"잘못된 {message.encoding} Image 데이터입니다.")
        frame = data[: row_step * height].reshape(height, row_step)
        frame = frame[:, :required_step].reshape(height, width, 3)
        if encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(frame)

    if encoding in {"mono8", "8uc1"}:
        if row_step < width or data.size < row_step * height:
            raise ValueError(f"잘못된 {message.encoding} Image 데이터입니다.")
        gray = data[: row_step * height].reshape(height, row_step)[:, :width]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if encoding in {"yuyv", "yuy2", "yuv422_yuy2"}:
        required_step = width * 2
        if row_step < required_step or data.size < row_step * height:
            raise ValueError(f"잘못된 {message.encoding} Image 데이터입니다.")
        yuyv = data[: row_step * height].reshape(height, row_step)
        yuyv = yuyv[:, :required_step].reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    raise ValueError(f"지원하지 않는 Image encoding입니다: {message.encoding}")


class LiveBevExporter(Node):
    WINDOW_ORIGINAL = "ROS camera - checkerboard and 4-point BEV"
    WINDOW_BEV = "Live BEV"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("live_bev_roboflow_exporter")
        self.args = args
        self.calibration = (
            None
            if args.no_undistort
            else load_calibration(args.calibration)
        )

        self.latest_frame: np.ndarray | None = None
        self.latest_timestamp_ns = 0
        self.frame_sequence = 0
        self.last_rendered_sequence = -1
        self.last_saved_sequence = -1

        self.manual_points: list[tuple[int, int]] = []
        self.homography: np.ndarray | None = None
        self.offset_x = 0
        self.offset_y = 0
        self.canvas_width = args.canvas_width
        self.canvas_height = args.canvas_height
        self.checkerboard_found = False
        self.checkerboard_corners: np.ndarray | None = None
        self.last_checkerboard_sequence = -1

        self.capture_active = False
        self.capture_seen = 0
        self.written = 0
        self.manifest_file = None
        self.manifest_writer = None
        self.session_output_dir: Path | None = None

        message_type = Image if args.raw else CompressedImage
        self.subscription = self.create_subscription(
            message_type,
            args.topic,
            self.image_callback,
            LOW_LATENCY_SENSOR_QOS,
        )
        self.gui_timer = self.create_timer(GUI_PERIOD_SECONDS, self.gui_callback)

        cv2.namedWindow(self.WINDOW_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.WINDOW_BEV, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_ORIGINAL, self.mouse_callback)

        type_name = "sensor_msgs/msg/Image" if args.raw else (
            "sensor_msgs/msg/CompressedImage"
        )
        self.get_logger().info(f"topic: {args.topic}")
        self.get_logger().info(f"type: {type_name}")
        self.get_logger().info(
            "checkerboard inner corners: "
            f"{args.checkerboard_columns}x{args.checkerboard_rows}"
        )
        print("\n[사용 방법]")
        print("0. 체커보드가 검출되면 영상에 녹색 코너가 표시됩니다.")
        print("1. 좌하단 -> 우하단 -> 좌상단 순서로 클릭합니다.")
        print("2. 좌상단을 지나는 밑변 평행 가이드선이 표시됩니다.")
        print("3. 가이드선을 참고해 우상단을 직접 클릭합니다.")
        print("4. 네 점의 좌표는 강제로 보정하지 않고 클릭값 그대로 사용합니다.")
        print("5. 실시간 BEV 확인 후 's': 파라미터 저장 및 이미지 추출 시작")
        print("6. 'r': 마지막 점 취소 / 'q': 종료\n")

    @property
    def src_points(self) -> list[tuple[int, int]]:
        return list(self.manual_points)

    def correct_distortion(self, frame: np.ndarray) -> np.ndarray:
        if self.calibration is None:
            return frame
        height, width = frame.shape[:2]
        if width != self.calibration.width or height != self.calibration.height:
            raise ValueError(
                "카메라 영상과 캘리브레이션 해상도가 다릅니다: "
                f"{width}x{height} != "
                f"{self.calibration.width}x{self.calibration.height}"
            )
        return cv2.remap(
            frame,
            self.calibration.map_x,
            self.calibration.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def image_callback(self, message: Image | CompressedImage) -> None:
        try:
            if self.args.raw:
                frame = decode_raw_message(message)
            else:
                frame = decode_compressed_message(message)
            self.latest_frame = self.correct_distortion(frame)
            self.latest_timestamp_ns = (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec)
            )
            self.frame_sequence += 1
        except Exception as error:
            self.get_logger().error(f"이미지 처리 실패: {error}")

    def mouse_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_frame is None:
            return
        if self.capture_active:
            print("[WARNING] 이미지 추출 중에는 점을 변경할 수 없습니다.")
            return
        if len(self.manual_points) >= 4:
            print("[WARNING] 네 점이 선택되었습니다. 'r'로 다시 선택하세요.")
            return

        point = (x - self.offset_x, y - self.offset_y)
        labels = ["Left-Bottom", "Right-Bottom", "Left-Top", "Right-Top"]
        self.manual_points.append(point)
        self.update_homography()
        print(
            f"[INFO] Added {labels[len(self.manual_points) - 1]}: "
            f"{point} ({len(self.manual_points)}/4)"
        )
        if len(self.manual_points) == 3:
            print("[INFO] 밑변 평행 가이드선을 따라 우상단을 클릭하세요.")
        elif len(self.manual_points) == 4:
            print("[INFO] 실시간 BEV를 확인하고 's'를 누르세요.")

    def update_homography(self) -> None:
        if len(self.manual_points) != 4:
            self.homography = None
            return
        self.homography = cv2.getPerspectiveTransform(
            np.float32(self.src_points),
            destination_points(self.args.warp_width, self.args.warp_height),
        )

    def draw_point(
        self,
        display: np.ndarray,
        point: tuple[int, int],
        label: str,
    ) -> None:
        frame_height, frame_width = self.latest_frame.shape[:2]
        canvas_point = (
            point[0] + self.offset_x,
            point[1] + self.offset_y,
        )
        inside = (
            0 <= point[0] < frame_width
            and 0 <= point[1] < frame_height
        )
        color = (0, 255, 0) if inside else (0, 165, 255)
        cv2.circle(display, canvas_point, 6, color, -1)
        cv2.putText(
            display,
            label,
            (canvas_point[0] + 7, canvas_point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    def make_bev(self, frame: np.ndarray) -> np.ndarray | None:
        if self.homography is None:
            return None
        return cv2.warpPerspective(
            frame,
            self.homography,
            (self.args.warp_width, self.args.warp_height),
            flags=cv2.INTER_LINEAR,
        )

    def update_checkerboard_detection(self, frame: np.ndarray) -> None:
        if (
            self.last_checkerboard_sequence >= 0
            and self.frame_sequence % self.args.checkerboard_every != 0
        ):
            return
        self.checkerboard_found, self.checkerboard_corners = detect_checkerboard(
            frame,
            self.args.checkerboard_columns,
            self.args.checkerboard_rows,
        )
        self.last_checkerboard_sequence = self.frame_sequence

    def draw_checkerboard_status(
        self,
        image: np.ndarray,
        found: bool,
        origin: tuple[int, int],
    ) -> None:
        if found:
            text = (
                "CHECKERBOARD DETECTED "
                f"{self.args.checkerboard_columns}x"
                f"{self.args.checkerboard_rows}"
            )
            color = (0, 255, 0)
        else:
            text = (
                "CHECKERBOARD NOT FOUND "
                f"{self.args.checkerboard_columns}x"
                f"{self.args.checkerboard_rows}"
            )
            color = (0, 0, 255)
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def render(self, frame: np.ndarray) -> None:
        self.update_checkerboard_detection(frame)
        annotated_frame = frame.copy()
        if self.checkerboard_found and self.checkerboard_corners is not None:
            cv2.drawChessboardCorners(
                annotated_frame,
                (
                    self.args.checkerboard_columns,
                    self.args.checkerboard_rows,
                ),
                self.checkerboard_corners,
                True,
            )

        frame_height, frame_width = frame.shape[:2]
        canvas_width = max(self.args.canvas_width, frame_width)
        canvas_height = max(self.args.canvas_height, frame_height)
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.offset_x = (canvas_width - frame_width) // 2
        self.offset_y = (canvas_height - frame_height) // 2

        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[
            self.offset_y : self.offset_y + frame_height,
            self.offset_x : self.offset_x + frame_width,
        ] = annotated_frame
        display = canvas.copy()
        cv2.rectangle(
            display,
            (self.offset_x, self.offset_y),
            (
                self.offset_x + frame_width - 1,
                self.offset_y + frame_height - 1,
            ),
            (255, 255, 0),
            2,
        )

        labels = ["1 L-Bottom", "2 R-Bottom", "3 L-Top", "4 R-Top"]
        points = self.src_points
        for index, point in enumerate(points):
            self.draw_point(
                display,
                point,
                labels[index],
            )

        if len(points) >= 2:
            left_bottom = (
                points[0][0] + self.offset_x,
                points[0][1] + self.offset_y,
            )
            right_bottom = (
                points[1][0] + self.offset_x,
                points[1][1] + self.offset_y,
            )
            cv2.line(display, left_bottom, right_bottom, (0, 255, 255), 2)

        if len(points) >= 3:
            # 좌상단을 지나고 밑변과 평행한 무한 직선을 화면 안에 표시한다.
            dx = points[1][0] - points[0][0]
            dy = points[1][1] - points[0][1]
            vector_length = float(np.hypot(dx, dy))
            if vector_length > 0.0:
                center = (
                    points[2][0] + self.offset_x,
                    points[2][1] + self.offset_y,
                )
                scale = int(
                    2 * max(canvas_width, canvas_height) / vector_length
                ) + 1
                line_start = (
                    center[0] - dx * scale,
                    center[1] - dy * scale,
                )
                line_end = (
                    center[0] + dx * scale,
                    center[1] + dy * scale,
                )
                visible, clipped_start, clipped_end = cv2.clipLine(
                    (0, 0, canvas_width, canvas_height),
                    line_start,
                    line_end,
                )
                if visible:
                    cv2.line(
                        display,
                        clipped_start,
                        clipped_end,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        "Parallel guide: click R-Top on this line",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                    )

        if len(points) == 4:
            polygon_order = (0, 1, 3, 2)
            polygon = np.array(
                [
                    (
                        points[index][0] + self.offset_x,
                        points[index][1] + self.offset_y,
                    )
                    for index in polygon_order
                ],
                dtype=np.int32,
            )
            cv2.polylines(display, [polygon], True, (0, 0, 255), 2)

        status = "SELECT 4 POINTS"
        if self.capture_active:
            status = f"EXPORTING {self.written}/{self.args.count}"
        elif self.written >= self.args.count:
            status = f"EXPORT COMPLETE: {self.written}"
        elif self.homography is not None:
            status = "LIVE BEV READY - PRESS S"

        cv2.putText(
            display,
            "Click: L-Bottom -> R-Bottom -> L-Top -> R-Top",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display,
            status,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 200, 255),
            2,
        )
        self.draw_checkerboard_status(
            display,
            self.checkerboard_found,
            (20, 140),
        )
        cv2.imshow(self.WINDOW_ORIGINAL, display)

        bev = self.make_bev(frame)
        if bev is not None:
            bev_display = bev.copy()
            if (
                self.checkerboard_found
                and self.checkerboard_corners is not None
                and self.homography is not None
            ):
                bev_corners = cv2.perspectiveTransform(
                    self.checkerboard_corners,
                    self.homography,
                )
                cv2.drawChessboardCorners(
                    bev_display,
                    (
                        self.args.checkerboard_columns,
                        self.args.checkerboard_rows,
                    ),
                    bev_corners,
                    True,
                )
            self.draw_checkerboard_status(
                bev_display,
                self.checkerboard_found,
                (15, 30),
            )
            cv2.imshow(self.WINDOW_BEV, bev_display)

    def save_parameters(self) -> None:
        if self.latest_frame is None or self.homography is None:
            raise RuntimeError("BEV 설정이 완료되지 않았습니다.")

        src_points = np.float32(self.src_points)
        dst_points = destination_points(
            self.args.warp_width,
            self.args.warp_height,
        )
        self.args.bev_params.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.args.bev_params,
            homography=self.homography,
            warp_width=self.args.warp_width,
            warp_height=self.args.warp_height,
            src_points=src_points,
            dst_points=dst_points,
            warp_w=self.args.warp_width,
            warp_h=self.args.warp_height,
        )

        frame_height, frame_width = self.latest_frame.shape[:2]
        labels = [
            "Left-Bottom",
            "Right-Bottom",
            "Left-Top",
            "Right-Top",
        ]
        self.args.points_txt.parent.mkdir(parents=True, exist_ok=True)
        with self.args.points_txt.open("w", encoding="utf-8") as file:
            file.write("# ROS live 4-point BEV settings with parallel guide\n")
            file.write(f"# Image size: {frame_width} x {frame_height}\n")
            file.write("# All four coordinates are unmodified click positions\n")
            for label, point in zip(labels, src_points):
                file.write(f"{point[0]:.3f}, {point[1]:.3f} # {label}\n")

        print(f"[INFO] BEV 파라미터 저장: {self.args.bev_params}")
        print(f"[INFO] 선택 좌표 저장: {self.args.points_txt}")

    def start_capture(self) -> None:
        if self.homography is None:
            print("[WARNING] 네 점을 모두 선택해야 합니다.")
            return
        if self.capture_active:
            print("[WARNING] 이미 이미지 추출 중입니다.")
            return
        if self.written >= self.args.count:
            print("[WARNING] 설정한 이미지 수만큼 이미 추출했습니다.")
            return

        self.save_parameters()
        session_name = datetime.now().strftime("live_%Y%m%d_%H%M%S")
        self.session_output_dir = self.args.output_dir / session_name
        self.session_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.session_output_dir / "manifest.csv"
        self.manifest_file = manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.manifest_writer = csv.writer(self.manifest_file)
        self.manifest_writer.writerow(
            ["filename", "source_sequence", "timestamp_ns"]
        )
        self.manifest_file.flush()
        self.capture_active = True
        self.capture_seen = 0
        self.last_saved_sequence = self.frame_sequence
        print(f"[INFO] 실시간 이미지 추출 시작: {self.session_output_dir}")

    def save_live_frame(self, frame: np.ndarray) -> None:
        if (
            not self.capture_active
            or self.homography is None
            or self.frame_sequence == self.last_saved_sequence
        ):
            return

        self.last_saved_sequence = self.frame_sequence
        self.capture_seen += 1
        if (self.capture_seen - 1) % self.args.save_every != 0:
            return

        bev = self.make_bev(frame)
        if bev is None:
            return
        timestamp_ns = self.latest_timestamp_ns
        if timestamp_ns <= 0:
            timestamp_ns = self.get_clock().now().nanoseconds
        filename = f"bev_{self.written:04d}_ts{timestamp_ns}.jpg"
        output_path = self.session_output_dir / filename
        ok = cv2.imwrite(
            str(output_path),
            bev,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        )
        if not ok:
            raise OSError(f"이미지 저장 실패: {output_path}")

        self.manifest_writer.writerow(
            [filename, self.frame_sequence, timestamp_ns]
        )
        self.manifest_file.flush()
        self.written += 1
        if self.written % 100 == 0 or self.written == self.args.count:
            print(f"[INFO] 이미지 추출: {self.written}/{self.args.count}")

        if self.written >= self.args.count:
            self.capture_active = False
            self.close_manifest()
            print(f"[INFO] 이미지 추출 완료: {self.session_output_dir}")

    def gui_callback(self) -> None:
        try:
            frame = self.latest_frame
            if (
                frame is not None
                and self.frame_sequence != self.last_rendered_sequence
            ):
                self.render(frame)
                self.save_live_frame(frame)
                self.last_rendered_sequence = self.frame_sequence

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.get_logger().info("사용자 요청으로 종료합니다.")
                rclpy.shutdown()
            elif key == ord("r"):
                if self.capture_active:
                    print("[WARNING] 이미지 추출 중에는 점을 변경할 수 없습니다.")
                elif self.manual_points:
                    removed = self.manual_points.pop()
                    self.update_homography()
                    print(f"[INFO] 취소한 점: {removed}")
                else:
                    print("[INFO] 취소할 점이 없습니다.")
            elif key == ord("s"):
                self.start_capture()
        except Exception as error:
            self.get_logger().error(f"GUI/저장 처리 실패: {error}")
            self.capture_active = False
            self.close_manifest()

    def close_manifest(self) -> None:
        if self.manifest_file is not None:
            self.manifest_file.close()
            self.manifest_file = None
            self.manifest_writer = None

    def close(self) -> None:
        self.capture_active = False
        self.close_manifest()
        cv2.destroyAllWindows()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.warp_width <= 0 or args.warp_height <= 0:
        raise ValueError("BEV 출력 크기는 1 이상이어야 합니다.")
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("캔버스 크기는 1 이상이어야 합니다.")
    if args.count <= 0:
        raise ValueError("--count는 1 이상이어야 합니다.")
    if args.save_every <= 0:
        raise ValueError("--save-every는 1 이상이어야 합니다.")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality는 0~100이어야 합니다.")
    if args.checkerboard_columns <= 1 or args.checkerboard_rows <= 1:
        raise ValueError("체커보드 내부 코너 개수는 가로·세로 모두 2 이상이어야 합니다.")
    if args.checkerboard_every <= 0:
        raise ValueError("--checkerboard-every는 1 이상이어야 합니다.")


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)
    rclpy.init()
    node = None
    try:
        node = LiveBevExporter(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
