#!/usr/bin/env python3
"""ROS 2 bag 영상을 일정 시간 간격으로 샘플링해 보정된 BEV 데이터셋을 만든다."""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import cv2
import numpy as np

from captured_frame_bev_parallel_roi import load_calibration
from rosbag_bev_triple_viewer import BevParameters, PlaybackImageSource


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BAGS = (
    PROJECT_ROOT / "rosbag2_2026_07_23-14_38_52",
    PROJECT_ROOT / "rosbag2_2026_07_23-14_55_06",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROS bag에서 Roboflow 라벨링용 BEV JPEG를 추출합니다."
    )
    parser.add_argument("--bags", nargs="+", type=Path, default=DEFAULT_BAGS)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "camera_calibration.npz",
    )
    parser.add_argument(
        "--bev",
        type=Path,
        default=PROJECT_ROOT / "bev(0729).npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "roboflow_bev_dataset",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
        help="시간 기준 샘플링 간격(기본 1초)",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    for name in ("calibration", "bev", "output_dir"):
        path = Path(getattr(args, name)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        setattr(args, name, path.resolve())
    args.bags = [
        (p if p.is_absolute() else PROJECT_ROOT / p).expanduser().resolve()
        for p in args.bags
    ]
    if not np.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        parser.error("--interval-seconds는 0보다 커야 합니다.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality는 1~100이어야 합니다.")
    return args


def sampled_indices(timestamps: list[int], interval_seconds: float) -> list[int]:
    """첫 프레임부터 시간 격자에 가장 가까운 프레임을 중복 없이 고른다."""
    first, last = timestamps[0], timestamps[-1]
    step_ns = int(round(interval_seconds * 1_000_000_000))
    targets = range(first, last + 1, step_ns)
    selected: list[int] = []
    for target in targets:
        right = bisect.bisect_left(timestamps, target)
        candidates = [i for i in (right - 1, right) if 0 <= i < len(timestamps)]
        index = min(candidates, key=lambda i: abs(timestamps[i] - target))
        if not selected or selected[-1] != index:
            selected.append(index)
    return selected


def main() -> None:
    args = parse_arguments()
    calibration = load_calibration(args.calibration)
    bev = BevParameters(args.bev)
    if (calibration.width, calibration.height) != (
        bev.calibration_width,
        bev.calibration_height,
    ):
        raise ValueError("카메라 캘리브레이션과 BEV 기준 해상도가 다릅니다.")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for bag in args.bags:
        source = PlaybackImageSource(bag, None)
        indices = sampled_indices(source.timestamps, args.interval_seconds)
        first_timestamp = source.timestamps[0]
        print(
            f"{bag.name}: {len(source)} frames, "
            f"{(source.timestamps[-1] - first_timestamp) / 1e9:.3f}s, "
            f"export {len(indices)} images"
        )
        for sequence, index in enumerate(indices):
            record = source.read(index)
            if record.image.shape[:2] != (calibration.height, calibration.width):
                raise ValueError(
                    f"{bag.name} frame {index}: 예상 해상도 "
                    f"{calibration.width}x{calibration.height}, 실제 "
                    f"{record.image.shape[1]}x{record.image.shape[0]}"
                )
            undistorted = cv2.remap(
                record.image,
                calibration.map_x,
                calibration.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            bev_image = cv2.warpPerspective(
                undistorted,
                bev.homography,
                (bev.width, bev.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            relative_ms = int(
                round((record.bag_timestamp_ns - first_timestamp) / 1_000_000)
            )
            filename = (
                f"{bag.name}__s{sequence:04d}__f{index:06d}"
                f"__t{relative_ms:06d}ms.jpg"
            )
            output_path = images_dir / filename
            if not cv2.imwrite(
                str(output_path),
                bev_image,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            ):
                raise OSError(f"이미지 저장 실패: {output_path}")
            manifest_rows.append(
                {
                    "filename": filename,
                    "bag": bag.name,
                    "source_frame_index": index,
                    "relative_time_seconds": f"{relative_ms / 1000:.3f}",
                    "bag_timestamp_ns": record.bag_timestamp_ns,
                    "width": bev.width,
                    "height": bev.height,
                }
            )

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"완료: {len(manifest_rows)} images -> {images_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
