#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import sqlite3
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAG = Path(
    "/home/tak/bag/rosbag2_02/rosbag2_2026_07_23-14_55_06_0.db3"
)
DEFAULT_BEV_PARAMS = PROJECT_ROOT / "bev_params_parallel_marker.npz.npz"
DEFAULT_CALIBRATION = PROJECT_ROOT / "camera_calibration.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "roboflow_bev_images"
JPEG_SIGNATURE = b"\xff\xd8\xff"
PNG_SIGNATURE = b"\x89PNG"


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "rosbag2 sqlite DB3의 CompressedImage 프레임에 BEV homography를 "
            "적용해 Roboflow 학습용 JPG 이미지를 추출합니다."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--bev-params", type=Path, default=DEFAULT_BEV_PARAMS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--topic", default="/image_raw/compressed")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0])


def load_bev_params(path: Path) -> tuple[np.ndarray, int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"BEV 파라미터 파일이 없습니다: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"homography", "warp_width", "warp_height"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"BEV 파라미터 키 누락: {sorted(missing)}")
        homography = np.asarray(data["homography"], dtype=np.float64)
        width = scalar_int(data, "warp_width")
        height = scalar_int(data, "warp_height")
    if homography.shape != (3, 3) or width <= 0 or height <= 0:
        raise ValueError("BEV homography 또는 출력 크기가 유효하지 않습니다.")
    return homography, width, height


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
    return Calibration(width=width, height=height, map_x=map_x, map_y=map_y)


def image_from_compressed_blob(blob: bytes) -> np.ndarray:
    start = blob.find(JPEG_SIGNATURE)
    if start < 0:
        start = blob.find(PNG_SIGNATURE)
    if start < 0:
        raise ValueError("CompressedImage blob에서 JPEG/PNG 데이터를 찾지 못했습니다.")
    encoded = np.frombuffer(blob[start:], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("CompressedImage 디코딩에 실패했습니다.")
    return image


def selected_indices(total: int, count: int) -> set[int]:
    if count <= 0:
        raise ValueError("--count는 1 이상이어야 합니다.")
    if total <= count:
        return set(range(total))
    indices = np.linspace(0, total - 1, count, dtype=np.int64)
    return set(int(index) for index in indices)


def main() -> int:
    args = parse_arguments()
    if not args.bag.is_file():
        raise FileNotFoundError(f"bag DB3 파일이 없습니다: {args.bag}")

    homography, warp_width, warp_height = load_bev_params(args.bev_params)
    calibration = load_calibration(args.calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(args.bag))
    cursor = connection.cursor()
    topic_row = cursor.execute(
        "select id, type from topics where name = ?",
        (args.topic,),
    ).fetchone()
    if topic_row is None:
        raise KeyError(f"토픽을 찾지 못했습니다: {args.topic}")
    topic_id, topic_type = topic_row
    if topic_type != "sensor_msgs/msg/CompressedImage":
        raise TypeError(f"지원하지 않는 토픽 타입입니다: {topic_type}")

    total = int(cursor.execute(
        "select count(*) from messages where topic_id = ?",
        (topic_id,),
    ).fetchone()[0])
    wanted = selected_indices(total, args.count)
    manifest_path = args.output_dir / "manifest.csv"

    written = 0
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["filename", "source_index", "timestamp_ns"])
        rows = cursor.execute(
            """
            select timestamp, data
            from messages
            where topic_id = ?
            order by timestamp
            """,
            (topic_id,),
        )
        for source_index, (timestamp_ns, blob) in enumerate(rows):
            if source_index not in wanted:
                continue
            image = image_from_compressed_blob(blob)
            if (
                image.shape[1] != calibration.width
                or image.shape[0] != calibration.height
            ):
                raise ValueError(
                    "bag 이미지와 캘리브레이션 해상도가 다릅니다: "
                    f"{image.shape[1]}x{image.shape[0]} != "
                    f"{calibration.width}x{calibration.height}"
                )
            undistorted = cv2.remap(
                image,
                calibration.map_x,
                calibration.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            bev = cv2.warpPerspective(
                undistorted,
                homography,
                (warp_width, warp_height),
                flags=cv2.INTER_LINEAR,
            )
            filename = f"bev_{written:04d}_ts{timestamp_ns}.jpg"
            output_path = args.output_dir / filename
            ok = cv2.imwrite(
                str(output_path),
                bev,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                raise OSError(f"이미지 저장 실패: {output_path}")
            writer.writerow([filename, source_index, timestamp_ns])
            written += 1

    connection.close()
    print(f"topic: {args.topic}")
    print(f"source frames: {total}")
    print(f"written images: {written}")
    print(f"output dir: {args.output_dir}")
    print(f"manifest: {manifest_path}")
    print(f"calibration: {args.calibration}")
    print(f"bev size: {warp_width}x{warp_height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
