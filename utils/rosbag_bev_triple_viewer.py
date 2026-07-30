#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS 2 bag을 원본 / ROI 오버레이 / BEV 세 화면으로 재생한다."""

from __future__ import annotations

import argparse
import bisect
import math
from pathlib import Path
import time

import cv2
import numpy as np

from captured_frame_bev_parallel_roi import load_calibration
from rosbag_frame_capture_viewer import (
    FrameRecord,
    LEFT_KEY_CODES,
    RIGHT_KEY_CODES,
    RosbagImageSource,
    compressed_suffix,
    deserialize_message,
    header_timestamp_ns,
    ros_raw_image_to_bgr,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAG = PROJECT_ROOT / "rosbag2_2026_07_23-14_38_52"
DEFAULT_CALIBRATION = PROJECT_ROOT / "camera_calibration.npz"
DEFAULT_BEV = PROJECT_ROOT / "bev_params_parallel_marker.npz"

WINDOW_NAME = "ROS bag | ORIGINAL | ROI | BEV"
DISPLAY_HEIGHT = 480
PANEL_GAP = 4
POINT_NAMES = ("LB", "RB", "LT", "RT")
LB, RB, LT, RT = 0, 1, 2, 3


def absolute_project_path(path: Path) -> Path:
    result = path.expanduser()
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    return result.resolve()


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS 2 bag을 원본, undistorted+ROI, BEV 세 패널로 재생합니다."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--bev", type=Path, default=DEFAULT_BEV)
    parser.add_argument(
        "--topic",
        default=None,
        help="생략하면 bag 영상 토픽을 자동 선택",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=39.0,
        help="bag 첫 영상 기준 시작 시각 (기본값: 39초)",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="재생 배속 (기본값: 1.0)",
    )
    arguments = parser.parse_args(argv)
    for name in ("bag", "calibration", "bev"):
        setattr(
            arguments,
            name,
            absolute_project_path(Path(getattr(arguments, name))),
        )
    if (
        not math.isfinite(arguments.start_seconds)
        or arguments.start_seconds < 0.0
    ):
        parser.error("--start-seconds는 0 이상의 유한한 값이어야 합니다.")
    if (
        not math.isfinite(arguments.playback_rate)
        or arguments.playback_rate <= 0.0
        or arguments.playback_rate > 16.0
    ):
        parser.error("--playback-rate는 0보다 크고 16 이하여야 합니다.")
    return arguments


class BevParameters:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"BEV NPZ가 없습니다: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {
                "src_points",
                "homography",
                "warp_width",
                "warp_height",
                "calibration_width",
                "calibration_height",
                "coordinate_space",
            }
            missing = required - set(data.files)
            if missing:
                raise KeyError(f"BEV NPZ 키 누락: {sorted(missing)}")
            self.src_points = np.asarray(
                data["src_points"],
                dtype=np.float64,
            )
            self.homography = np.asarray(
                data["homography"],
                dtype=np.float64,
            )
            self.width = int(np.asarray(data["warp_width"]).item())
            self.height = int(np.asarray(data["warp_height"]).item())
            self.calibration_width = int(
                np.asarray(data["calibration_width"]).item()
            )
            self.calibration_height = int(
                np.asarray(data["calibration_height"]).item()
            )
            self.coordinate_space = str(
                np.asarray(data["coordinate_space"]).item()
            )
            self.extension_px = (
                int(np.asarray(data["source_extension_px"]).item())
                if "source_extension_px" in data.files
                else 0
            )
        if (
            self.src_points.shape != (4, 2)
            or self.homography.shape != (3, 3)
            or not np.all(np.isfinite(self.src_points))
            or not np.all(np.isfinite(self.homography))
            or np.linalg.matrix_rank(self.homography) < 3
        ):
            raise ValueError("BEV 좌표 또는 homography가 유효하지 않습니다.")
        if (
            self.width < 2
            or self.height < 2
            or self.width > 4096
            or self.height > 4096
            or self.extension_px < 0
            or self.extension_px > 200
        ):
            raise ValueError("BEV 출력 크기 또는 확장 픽셀이 유효하지 않습니다.")
        if self.coordinate_space != "full_undistorted_camera_image":
            raise ValueError(
                "지원하지 않는 BEV 좌표계입니다: "
                f"{self.coordinate_space}"
            )


class PlaybackImageSource(RosbagImageSource):
    """연속 재생에서는 DB seek 없이 다음 메시지를 바로 읽는다."""

    def __init__(
        self,
        bag_path: Path,
        requested_topic: str | None,
    ) -> None:
        super().__init__(bag_path, requested_topic)
        self.next_reader_index = 0
        self.last_index = -1
        self.last_record: FrameRecord | None = None

    def decode_record(
        self,
        serialized: bytes,
        timestamp: int,
    ) -> FrameRecord:
        if deserialize_message is None:
            raise RuntimeError("ROS 메시지 역직렬화 모듈이 없습니다.")
        message = deserialize_message(serialized, self.message_type)
        frame_id = str(message.header.frame_id)
        if self.type_name == "sensor_msgs/msg/CompressedImage":
            payload = bytes(message.data)
            image = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None or image.size == 0:
                raise ValueError("CompressedImage 디코딩에 실패했습니다.")
            message_format = str(message.format)
            return FrameRecord(
                image=image,
                bag_timestamp_ns=timestamp,
                header_timestamp_ns=header_timestamp_ns(message),
                frame_id=frame_id,
                encoding=message_format,
                compressed_payload=payload,
                compressed_suffix=compressed_suffix(
                    payload,
                    message_format,
                ),
            )
        return FrameRecord(
            image=ros_raw_image_to_bgr(message),
            bag_timestamp_ns=timestamp,
            header_timestamp_ns=header_timestamp_ns(message),
            frame_id=frame_id,
            encoding=str(message.encoding),
            compressed_payload=None,
            compressed_suffix=None,
        )

    def read(self, index: int) -> FrameRecord:
        if index < 0 or index >= len(self.timestamps):
            raise IndexError(f"bag frame index 범위 초과: {index}")
        if index == self.last_index and self.last_record is not None:
            return self.last_record

        # 짧은 프레임 드롭은 순서대로 건너뛰고, 큰 점프/역방향만 seek한다.
        if (
            index < self.next_reader_index
            or index - self.next_reader_index > 8
        ):
            target_timestamp = self.timestamps[index]
            self.reader.seek(target_timestamp)
            self.next_reader_index = bisect.bisect_left(
                self.timestamps,
                target_timestamp,
            )

        serialized = None
        timestamp = 0
        while self.next_reader_index <= index:
            if not self.reader.has_next():
                raise RuntimeError("bag 순차 재생 중 프레임이 끝났습니다.")
            topic, current_serialized, current_timestamp = (
                self.reader.read_next()
            )
            expected_timestamp = self.timestamps[
                self.next_reader_index
            ]
            if (
                topic != self.topic
                or int(current_timestamp) != expected_timestamp
            ):
                raise RuntimeError(
                    "bag 순차 프레임이 timestamp index와 일치하지 않습니다."
                )
            serialized = current_serialized
            timestamp = int(current_timestamp)
            self.next_reader_index += 1
        if serialized is None:
            raise RuntimeError("bag 프레임을 읽지 못했습니다.")
        record = self.decode_record(serialized, timestamp)
        self.last_index = index
        self.last_record = record
        return record


def resize_to_height(
    image: np.ndarray,
    target_height: int,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("패널 영상이 유효하지 않습니다.")
    if image.shape[0] == target_height:
        return image.copy()
    scale = target_height / image.shape[0]
    target_width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )


def add_panel_label(
    panel: np.ndarray,
    text: str,
    color: tuple[int, int, int],
) -> None:
    overlay = panel.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (panel.shape[1] - 1, 34),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, panel, 0.28, 0.0, dst=panel)
    cv2.putText(
        panel,
        text[:92],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


class TripleBevViewer:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.source = PlaybackImageSource(arguments.bag, arguments.topic)
        self.calibration = load_calibration(arguments.calibration)
        self.bev = BevParameters(arguments.bev)
        if (
            self.bev.calibration_width != self.calibration.width
            or self.bev.calibration_height != self.calibration.height
        ):
            raise ValueError(
                "BEV와 카메라 캘리브레이션 해상도가 다릅니다: "
                f"{self.bev.calibration_width}x"
                f"{self.bev.calibration_height} != "
                f"{self.calibration.width}x{self.calibration.height}"
            )

        target_timestamp = (
            self.source.timestamps[0]
            + int(round(arguments.start_seconds * 1_000_000_000.0))
        )
        self.current_index = self.source.index_near_time(target_timestamp)
        self.playing = True
        self.play_anchor_wall = time.monotonic()
        self.play_anchor_timestamp = self.source.timestamps[
            self.current_index
        ]
        self.current_display: np.ndarray | None = None
        self.dirty = True

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )

    def relative_seconds(self, timestamp_ns: int) -> float:
        return (
            timestamp_ns - self.source.timestamps[0]
        ) / 1_000_000_000.0

    def overlay_roi(self, undistorted: np.ndarray) -> np.ndarray:
        extension = self.bev.extension_px
        result = cv2.copyMakeBorder(
            undistorted,
            extension,
            extension,
            extension,
            extension,
            cv2.BORDER_CONSTANT,
            value=(18, 18, 18),
        )
        if extension > 0:
            cv2.rectangle(
                result,
                (extension, extension),
                (
                    extension + undistorted.shape[1] - 1,
                    extension + undistorted.shape[0] - 1,
                ),
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
        shift = np.asarray([extension, extension], dtype=np.float64)
        points = np.rint(self.bev.src_points + shift).astype(np.int32)
        polygon = np.asarray([
            points[LT],
            points[RT],
            points[RB],
            points[LB],
        ])
        cv2.polylines(
            result,
            [polygon],
            True,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        for name, point in zip(POINT_NAMES, points):
            center = tuple(map(int, point))
            cv2.circle(result, center, 6, (0, 255, 255), -1)
            cv2.putText(
                result,
                name,
                (center[0] + 7, center[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return result

    def render_frame(self) -> np.ndarray:
        record = self.source.read(self.current_index)
        original = record.image
        if (
            original.shape[1] != self.calibration.width
            or original.shape[0] != self.calibration.height
        ):
            raise ValueError(
                "bag 영상과 캘리브레이션 해상도가 다릅니다: "
                f"{original.shape[1]}x{original.shape[0]}"
            )
        undistorted = cv2.remap(
            original,
            self.calibration.map_x,
            self.calibration.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        roi_overlay = self.overlay_roi(undistorted)
        bev_frame = cv2.warpPerspective(
            undistorted,
            self.bev.homography,
            (self.bev.width, self.bev.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        original_panel = resize_to_height(original, DISPLAY_HEIGHT)
        roi_panel = resize_to_height(roi_overlay, DISPLAY_HEIGHT)
        bev_panel = resize_to_height(bev_frame, DISPLAY_HEIGHT)
        relative = self.relative_seconds(record.bag_timestamp_ns)
        state = "PLAY" if self.playing else "PAUSE"
        add_panel_label(
            original_panel,
            (
                f"ORIGINAL | t={relative:.3f}s | "
                f"{self.current_index + 1}/{len(self.source)} | {state}"
            ),
            (0, 255, 0),
        )
        add_panel_label(
            roi_panel,
            (
                "UNDISTORTED + ROI"
                + (
                    f" | {self.bev.extension_px}px EXT"
                    if self.bev.extension_px
                    else ""
                )
            ),
            (0, 255, 255),
        )
        add_panel_label(
            bev_panel,
            f"BEV | {self.bev.width}x{self.bev.height}",
            (255, 255, 0),
        )
        gap = np.full(
            (DISPLAY_HEIGHT, PANEL_GAP, 3),
            48,
            dtype=np.uint8,
        )
        return np.hstack((
            original_panel,
            gap,
            roi_panel,
            gap,
            bev_panel,
        ))

    def reset_play_anchor(self) -> None:
        self.play_anchor_wall = time.monotonic()
        self.play_anchor_timestamp = self.source.timestamps[
            self.current_index
        ]

    def set_frame(self, index: int, *, pause: bool = True) -> None:
        self.current_index = int(np.clip(
            index,
            0,
            len(self.source) - 1,
        ))
        if pause:
            self.playing = False
        self.dirty = True

    def update_playback(self) -> None:
        if not self.playing:
            return
        elapsed_ns = int(
            (time.monotonic() - self.play_anchor_wall)
            * self.arguments.playback_rate
            * 1_000_000_000.0
        )
        target = self.play_anchor_timestamp + elapsed_ns
        index = bisect.bisect_right(
            self.source.timestamps,
            target,
        ) - 1
        index = int(np.clip(index, 0, len(self.source) - 1))
        if index != self.current_index:
            self.current_index = index
            self.dirty = True
        if (
            self.current_index == len(self.source) - 1
            and target >= self.source.timestamps[-1]
        ):
            self.playing = False
            self.dirty = True

    def handle_key(self, key: int) -> bool:
        if key < 0:
            return True
        if key in LEFT_KEY_CODES:
            self.set_frame(self.current_index - 1)
            return True
        if key in RIGHT_KEY_CODES:
            self.set_frame(self.current_index + 1)
            return True
        ascii_key = key & 0xFF
        if ascii_key in (27, ord("q"), ord("Q")):
            return False
        if ascii_key == 32:
            self.playing = not self.playing
            if self.playing:
                self.reset_play_anchor()
            self.dirty = True
        elif ascii_key in (ord("j"), ord("J")):
            self.set_frame(self.current_index - 1)
        elif ascii_key in (ord("l"), ord("L")):
            self.set_frame(self.current_index + 1)
        elif ascii_key in (ord("u"), ord("U")):
            target = self.source.timestamps[
                self.current_index
            ] - 1_000_000_000
            self.set_frame(self.source.index_near_time(target))
        elif ascii_key in (ord("o"), ord("O")):
            target = self.source.timestamps[
                self.current_index
            ] + 1_000_000_000
            self.set_frame(self.source.index_near_time(target))
        return True

    def print_summary(self) -> None:
        print("=" * 78)
        print("ROS bag 원본 / ROI / BEV 3분할 재생")
        print("=" * 78)
        print(f"bag           : {self.arguments.bag}")
        print(f"topic         : {self.source.topic}")
        print(f"calibration   : {self.arguments.calibration}")
        print(f"BEV NPZ       : {self.arguments.bev}")
        print(
            f"BEV output    : {self.bev.width}x{self.bev.height}, "
            f"selection extension {self.bev.extension_px}px"
        )
        print(
            f"start         : frame {self.current_index + 1}/"
            f"{len(self.source)}, "
            f"t={self.relative_seconds(self.source.timestamps[self.current_index]):.3f}s"
        )
        print(
            "SPACE 재생/정지 | J/L 또는 ←/→ 한 프레임 | "
            "U/O ±1초 | Q/ESC 종료"
        )
        print("=" * 78)

    def run(self) -> None:
        self.print_summary()
        # 창 생성 시간과 무관하게 지정한 첫 프레임부터 재생한다.
        self.reset_play_anchor()
        try:
            while True:
                self.update_playback()
                if self.dirty:
                    self.current_display = self.render_frame()
                    scale = min(
                        1.0,
                        1800.0 / self.current_display.shape[1],
                        900.0 / self.current_display.shape[0],
                    )
                    cv2.resizeWindow(
                        WINDOW_NAME,
                        max(1, int(round(
                            self.current_display.shape[1] * scale
                        ))),
                        max(1, int(round(
                            self.current_display.shape[0] * scale
                        ))),
                    )
                    cv2.imshow(WINDOW_NAME, self.current_display)
                    self.dirty = False
                if not self.handle_key(cv2.waitKeyEx(5)):
                    break
                try:
                    if cv2.getWindowProperty(
                        WINDOW_NAME,
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
        viewer = TripleBevViewer(arguments)
        viewer.run()
        return 0
    except KeyboardInterrupt:
        print("\n[중단] 3분할 재생기를 종료했습니다.")
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
        print("ROS bag 3분할 재생기를 시작하지 못했습니다.")
        print("=" * 72)
        print(error)
        print("=" * 72)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
