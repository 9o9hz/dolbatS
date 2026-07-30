#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS 2 bag 영상을 직접 탐색하고 S 키로 원본 프레임을 저장한다."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as ros_import_error:
    rosbag2_py = None
    deserialize_message = None
    get_message = None
    ROS_IMPORT_ERROR: ImportError | None = ros_import_error
else:
    ROS_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAG = PROJECT_ROOT / "rosbag2_2026_07_23-14_38_52"
SUPPORTED_TYPES = {
    "sensor_msgs/msg/CompressedImage",
    "sensor_msgs/msg/Image",
}
PREFERRED_TOPICS = (
    "/image_raw/compressed",
    "/camera/lane/raw/compressed",
    "/image_raw",
    "/camera/lane/raw",
)

WINDOW_NAME = "ROS bag frame capture - S saves exact frame"
TRACKBAR_NAME = "Frame"

# cv2.waitKeyEx()의 Ubuntu Qt/X11 및 Windows 확장 키 코드.
LEFT_KEY_CODES = {2424832, 65361}
UP_KEY_CODES = {2490368, 65362}
RIGHT_KEY_CODES = {2555904, 65363}
DOWN_KEY_CODES = {2621440, 65364}


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    bag_timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    encoding: str
    compressed_payload: bytes | None
    compressed_suffix: str | None


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
                "--bag은 ROS bag 디렉터리, metadata.yaml 또는 db3여야 합니다."
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
            "ROS 2 bag 영상을 직접 재생하고 S 키로 현재 원본 프레임을 저장합니다."
        )
    )
    parser.add_argument(
        "--bag",
        type=Path,
        default=DEFAULT_BAG,
        help=f"ROS bag 경로 (기본값: {DEFAULT_BAG.name})",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="영상 토픽. 생략하면 Image/CompressedImage 토픽 자동 선택",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="캡처 저장 폴더. 생략하면 <bag 이름>_captured_frames",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="처음 표시할 0-based 프레임 번호",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="재생 배속 (기본값: 1.0)",
    )
    arguments = parser.parse_args(argv)

    arguments.bag = normalize_bag_path(arguments.bag)
    if arguments.topic is not None:
        arguments.topic = arguments.topic.strip()
        if not arguments.topic:
            parser.error("--topic은 빈 문자열일 수 없습니다.")
    if arguments.start_frame < 0:
        parser.error("--start-frame은 0 이상이어야 합니다.")
    if (
        not np.isfinite(arguments.playback_rate)
        or arguments.playback_rate <= 0.0
        or arguments.playback_rate > 16.0
    ):
        parser.error("--playback-rate는 0보다 크고 16 이하여야 합니다.")

    if arguments.output_dir is None:
        arguments.output_dir = (
            PROJECT_ROOT / f"{arguments.bag.name}_captured_frames"
        )
    else:
        output_dir = arguments.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        arguments.output_dir = output_dir.resolve()
    return arguments


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
    elif encoding in ("yuv422_yuy2", "yuyv", "yuv422"):
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


def compressed_suffix(payload: bytes, message_format: str) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if (
        len(payload) >= 12
        and payload[:4] == b"RIFF"
        and payload[8:12] == b"WEBP"
    ):
        return ".webp"
    normalized = message_format.lower()
    if "jpeg" in normalized or "jpg" in normalized:
        return ".jpg"
    if "png" in normalized:
        return ".png"
    return ".bin"


def header_timestamp_ns(message: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class RosbagImageSource:
    def __init__(
        self,
        bag_path: Path,
        requested_topic: str | None,
    ) -> None:
        if rosbag2_py is None:
            raise RuntimeError(
                "ROS 2 rosbag Python 모듈을 불러오지 못했습니다.\n"
                "먼저 다음 명령을 실행하세요:\n"
                "  source /opt/ros/humble/setup.bash\n"
                f"원인: {ROS_IMPORT_ERROR}"
            )

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
            if topic_types[requested_topic] not in SUPPORTED_TYPES:
                raise ValueError(
                    "지원하지 않는 영상 메시지 타입입니다: "
                    f"{topic_types[requested_topic]}"
                )
            self.topic = requested_topic
        else:
            supported = [
                name
                for name, type_name in topic_types.items()
                if type_name in SUPPORTED_TYPES
            ]
            if not supported:
                raise ValueError(
                    "bag에 sensor_msgs/Image 또는 CompressedImage가 없습니다."
                )
            preferred = [
                topic for topic in PREFERRED_TOPICS if topic in supported
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
            raise ValueError(f"선택 토픽에 메시지가 없습니다: {self.topic}")
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

    def read(self, index: int) -> FrameRecord:
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
        frame_id = str(message.header.frame_id)

        if self.type_name == "sensor_msgs/msg/CompressedImage":
            payload = bytes(message.data)
            encoded = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError("CompressedImage 디코딩에 실패했습니다.")
            message_format = str(message.format)
            return FrameRecord(
                image=image,
                bag_timestamp_ns=target_timestamp,
                header_timestamp_ns=header_timestamp_ns(message),
                frame_id=frame_id,
                encoding=message_format,
                compressed_payload=payload,
                compressed_suffix=compressed_suffix(
                    payload,
                    message_format,
                ),
            )

        image = ros_raw_image_to_bgr(message)
        return FrameRecord(
            image=image,
            bag_timestamp_ns=target_timestamp,
            header_timestamp_ns=header_timestamp_ns(message),
            frame_id=frame_id,
            encoding=str(message.encoding),
            compressed_payload=None,
            compressed_suffix=None,
        )

    def index_near_time(self, timestamp_ns: int) -> int:
        index = bisect.bisect_left(self.timestamps, timestamp_ns)
        if index <= 0:
            return 0
        if index >= len(self.timestamps):
            return len(self.timestamps) - 1
        before = self.timestamps[index - 1]
        after = self.timestamps[index]
        if timestamp_ns - before <= after - timestamp_ns:
            return index - 1
        return index


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.stem}_",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FrameCaptureViewer:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.source = RosbagImageSource(
            arguments.bag,
            arguments.topic,
        )
        self.output_dir: Path = arguments.output_dir
        self.playback_rate = float(arguments.playback_rate)
        self.current_index = -1
        self.requested_index = min(
            int(arguments.start_frame),
            len(self.source) - 1,
        )
        self.current_record: FrameRecord | None = None
        self.playing = False
        self.next_play_deadline = 0.0
        self.status_message = ""
        self.dirty = True
        self.updating_trackbar = False

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        self.has_trackbar = len(self.source) > 1
        if self.has_trackbar:
            cv2.createTrackbar(
                TRACKBAR_NAME,
                WINDOW_NAME,
                self.requested_index,
                len(self.source) - 1,
                self.trackbar_callback,
            )

    def trackbar_callback(self, value: int) -> None:
        if self.updating_trackbar:
            return
        self.playing = False
        self.requested_index = int(np.clip(
            value,
            0,
            len(self.source) - 1,
        ))

    def set_requested_frame(
        self,
        index: int,
        *,
        pause: bool = True,
    ) -> None:
        self.requested_index = int(np.clip(
            index,
            0,
            len(self.source) - 1,
        ))
        if pause:
            self.playing = False

    def load_requested_frame(self) -> None:
        if self.current_index == self.requested_index:
            return
        self.current_record = self.source.read(self.requested_index)
        self.current_index = self.requested_index
        if self.has_trackbar:
            self.updating_trackbar = True
            cv2.setTrackbarPos(
                TRACKBAR_NAME,
                WINDOW_NAME,
                self.current_index,
            )
            self.updating_trackbar = False
        self.dirty = True

    def relative_seconds(self, timestamp_ns: int) -> float:
        return (
            timestamp_ns - self.source.timestamps[0]
        ) / 1_000_000_000.0

    def draw_frame(self) -> np.ndarray:
        if self.current_record is None:
            raise RuntimeError("표시할 bag 프레임이 없습니다.")
        display = self.current_record.image.copy()
        relative = self.relative_seconds(
            self.current_record.bag_timestamp_ns
        )
        lines = [
            (
                f"Frame {self.current_index + 1}/{len(self.source)} "
                f"| index {self.current_index} "
                f"| t={relative:.3f}s "
                f"| {'PLAY' if self.playing else 'PAUSE'}"
            ),
            (
                "SPACE play | J/L or LEFT/RIGHT 1 frame | "
                "U/O 1 sec | S SAVE EXACT | Q quit"
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
        display = cv2.addWeighted(overlay, 0.68, display, 0.32, 0.0)
        for line_index, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (8, 22 + line_index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48 if line_index == 0 else 0.39,
                (0, 255, 0) if line_index == 0 else (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if self.status_message:
            overlay = display.copy()
            y0 = display.shape[0] - 34
            cv2.rectangle(
                overlay,
                (0, y0),
                (display.shape[1] - 1, display.shape[0] - 1),
                (0, 0, 0),
                -1,
            )
            display = cv2.addWeighted(
                overlay,
                0.70,
                display,
                0.30,
                0.0,
            )
            cv2.putText(
                display,
                self.status_message[:92],
                (8, display.shape[0] - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        return display

    def render(self) -> None:
        self.load_requested_frame()
        cv2.imshow(WINDOW_NAME, self.draw_frame())
        self.dirty = False

    def capture_payload(
        self,
    ) -> tuple[bytes, str, str]:
        if self.current_record is None:
            raise RuntimeError("저장할 현재 프레임이 없습니다.")
        record = self.current_record
        if record.compressed_payload is not None:
            return (
                record.compressed_payload,
                record.compressed_suffix or ".bin",
                "original_ros_compressed_payload",
            )
        ok, encoded = cv2.imencode(".png", record.image)
        if not ok:
            raise OSError("raw ROS Image를 PNG로 인코딩하지 못했습니다.")
        return (
            encoded.tobytes(),
            ".png",
            "lossless_png_from_raw_ros_image",
        )

    def unique_capture_paths(
        self,
        base_stem: str,
        suffix: str,
        payload: bytes,
    ) -> tuple[Path, Path, bool]:
        sequence = 1
        while True:
            suffix_stem = "" if sequence == 1 else f"_{sequence:02d}"
            stem = f"{base_stem}{suffix_stem}"
            image_path = self.output_dir / f"{stem}{suffix}"
            metadata_path = self.output_dir / f"{stem}.json"
            if not image_path.exists() and not metadata_path.exists():
                return image_path, metadata_path, False
            if image_path.is_file() and image_path.read_bytes() == payload:
                return image_path, metadata_path, True
            sequence += 1

    def save_current_frame(self) -> Path:
        if self.current_record is None or self.current_index < 0:
            raise RuntimeError("저장할 현재 프레임이 없습니다.")
        self.playing = False
        record = self.current_record
        payload, suffix, storage_mode = self.capture_payload()
        relative_ms = int(round(
            self.relative_seconds(record.bag_timestamp_ns) * 1000.0
        ))
        base_stem = (
            f"frame_{self.current_index + 1:06d}_"
            f"t{relative_ms:09d}ms_"
            f"ts{record.bag_timestamp_ns}"
        )
        image_path, metadata_path, already_exists = (
            self.unique_capture_paths(
                base_stem,
                suffix,
                payload,
            )
        )
        payload_sha256 = sha256_bytes(payload)
        metadata = {
            "bag_uri": str(self.source.bag_path),
            "topic": self.source.topic,
            "message_type": self.source.type_name,
            "frame_number_1_based": self.current_index + 1,
            "frame_index_0_based": self.current_index,
            "frame_count": len(self.source),
            "bag_timestamp_ns": record.bag_timestamp_ns,
            "header_timestamp_ns": record.header_timestamp_ns,
            "relative_time_seconds": self.relative_seconds(
                record.bag_timestamp_ns
            ),
            "frame_id": record.frame_id,
            "encoding_or_format": record.encoding,
            "width": int(record.image.shape[1]),
            "height": int(record.image.shape[0]),
            "storage_mode": storage_mode,
            "saved_file": image_path.name,
            "payload_sha256": payload_sha256,
        }
        if not already_exists:
            atomic_write_bytes(image_path, payload)
        if not metadata_path.exists():
            atomic_write_text(
                metadata_path,
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            action = "저장"
        else:
            action = "이미 저장됨"

        # 원본 압축 payload도 다시 디코딩하여 현재 표시 프레임과 확인한다.
        if suffix in (".jpg", ".png", ".webp"):
            decoded = cv2.imdecode(
                np.frombuffer(payload, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if (
                decoded is None
                or decoded.shape != record.image.shape
                or not np.array_equal(decoded, record.image)
            ):
                raise RuntimeError("저장 payload와 현재 bag 프레임 검증 실패")

        print()
        print(
            f"[{action}] frame {self.current_index + 1}/"
            f"{len(self.source)} (index {self.current_index})"
        )
        print(f"이미지 : {image_path}")
        print(f"메타데이터: {metadata_path}")
        print(f"SHA-256: {payload_sha256}")
        self.status_message = (
            f"SAVED frame {self.current_index + 1}: {image_path.name}"
            if not already_exists
            else f"ALREADY SAVED: {image_path.name}"
        )
        self.dirty = True
        return image_path

    def reset_play_deadline(self) -> None:
        if self.current_index >= len(self.source) - 1:
            self.playing = False
            return
        delta = (
            self.source.timestamps[self.current_index + 1]
            - self.source.timestamps[self.current_index]
        ) / 1_000_000_000.0
        self.next_play_deadline = (
            time.monotonic()
            + max(0.001, delta / self.playback_rate)
        )

    def update_playback(self) -> None:
        if not self.playing or self.current_index < 0:
            return
        if time.monotonic() < self.next_play_deadline:
            return
        if self.current_index >= len(self.source) - 1:
            self.playing = False
            self.dirty = True
            return
        self.set_requested_frame(
            self.current_index + 1,
            pause=False,
        )
        self.load_requested_frame()
        self.reset_play_deadline()

    def handle_key(self, key: int) -> bool:
        if key < 0:
            return True

        # Linux 방향키의 하위 바이트가 Q/R/S/T와 겹치므로 먼저 처리한다.
        if key in LEFT_KEY_CODES:
            self.set_requested_frame(self.current_index - 1)
            return True
        if key in RIGHT_KEY_CODES:
            self.set_requested_frame(self.current_index + 1)
            return True
        if key in UP_KEY_CODES:
            self.set_requested_frame(self.current_index + 10)
            return True
        if key in DOWN_KEY_CODES:
            self.set_requested_frame(self.current_index - 10)
            return True

        ascii_key = key & 0xFF
        if ascii_key in (27, ord("q"), ord("Q")):
            return False
        if ascii_key == 32:
            self.playing = not self.playing
            if self.playing:
                self.reset_play_deadline()
            self.status_message = "PLAY" if self.playing else "PAUSE"
            self.dirty = True
            return True
        if ascii_key in (ord("j"), ord("J")):
            self.set_requested_frame(self.current_index - 1)
        elif ascii_key in (ord("l"), ord("L")):
            self.set_requested_frame(self.current_index + 1)
        elif ascii_key in (ord("u"), ord("U")):
            if self.current_record is not None:
                self.set_requested_frame(
                    self.source.index_near_time(
                        self.current_record.bag_timestamp_ns
                        - 1_000_000_000
                    )
                )
        elif ascii_key in (ord("o"), ord("O")):
            if self.current_record is not None:
                self.set_requested_frame(
                    self.source.index_near_time(
                        self.current_record.bag_timestamp_ns
                        + 1_000_000_000
                    )
                )
        elif ascii_key in (ord("s"), ord("S")):
            try:
                self.save_current_frame()
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                self.status_message = f"SAVE FAILED: {error}"
                self.dirty = True
                print(f"[저장 실패] {error}")
        return True

    def print_summary(self) -> None:
        duration = (
            self.source.timestamps[-1] - self.source.timestamps[0]
        ) / 1_000_000_000.0
        print("=" * 78)
        print("ROS bag 원본 프레임 캡처 뷰어")
        print("=" * 78)
        print(f"bag          : {self.source.bag_path}")
        print(f"토픽         : {self.source.topic}")
        print(f"메시지 타입  : {self.source.type_name}")
        print(f"프레임 수    : {len(self.source)}")
        print(f"구간         : {duration:.3f}초")
        print(f"저장 폴더    : {self.output_dir}")
        print()
        print("SPACE 재생/정지 | J/L 또는 좌우키 한 프레임")
        print("U/O 1초 이동 | 위/아래키 10프레임 | 트랙바 탐색")
        print("S 현재 bag 프레임 원본 저장 | Q/ESC 종료")
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
        viewer = FrameCaptureViewer(arguments)
        viewer.run()
        return 0
    except KeyboardInterrupt:
        print("\n[중단] 프레임 캡처 뷰어를 종료했습니다.")
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
        print("ROS bag 프레임 캡처 뷰어를 시작하지 못했습니다.")
        print("=" * 72)
        print(error)
        print("=" * 72)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
