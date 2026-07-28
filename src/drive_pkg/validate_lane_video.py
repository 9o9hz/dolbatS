#!/usr/bin/env python3
"""Validate lane post-processing on a raw camera video.

The input video is converted in the same coordinate order used to create the
BEV training images: undistortion first, then the saved BEV homography.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from yolo11_lane_post_processor import DEFAULT_MODEL_PATH, Yolo11LanePostProcessor


DEFAULT_VIDEO = Path("/home/tak/dolbatS/src/drive_pkg/Toy_project.mp4")
DEFAULT_CALIBRATION = Path("/home/tak/dolbatS/camera_calibration.npz")
DEFAULT_BEV_PARAMS = Path("/home/tak/dolbatS/bev_params_parallel_marker.npz.npz")
DEFAULT_OUTPUT = Path("/home/tak/dolbatS/lane_verification_bev.mp4")


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    map_x: np.ndarray
    map_y: np.ndarray


@dataclass(frozen=True)
class BevParameters:
    homography: np.ndarray
    width: int
    height: int


class BevTransformer:
    """Apply the project's saved camera calibration and BEV homography."""

    def __init__(
        self,
        calibration: CameraCalibration,
        bev_parameters: BevParameters,
    ) -> None:
        self.calibration = calibration
        self.bev_parameters = bev_parameters

    def transform(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if (width, height) != (
            self.calibration.width,
            self.calibration.height,
        ):
            raise ValueError(
                "영상 해상도와 calibration 해상도가 다릅니다: "
                f"{width}x{height} != "
                f"{self.calibration.width}x{self.calibration.height}"
            )

        undistorted = cv2.remap(
            frame,
            self.calibration.map_x,
            self.calibration.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return cv2.warpPerspective(
            undistorted,
            self.bev_parameters.homography,
            (self.bev_parameters.width, self.bev_parameters.height),
            flags=cv2.INTER_LINEAR,
        )


def _scalar_int(data: Any, key: str) -> int:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar")
    return int(value.reshape(-1)[0])


def load_calibration(path: Path) -> CameraCalibration:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")

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
            raise KeyError(f"Calibration keys missing: {sorted(missing)}")

        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(
            data["distortion_coefficients"],
            dtype=np.float64,
        ).reshape(-1)
        new_camera_matrix = np.asarray(
            data["new_camera_matrix"],
            dtype=np.float64,
        )
        width = _scalar_int(data, "image_width")
        height = _scalar_int(data, "image_height")

    if (
        camera_matrix.shape != (3, 3)
        or new_camera_matrix.shape != (3, 3)
        or distortion.size not in (4, 5, 8, 12, 14)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"Invalid calibration data: {path}")

    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return CameraCalibration(width, height, map_x, map_y)


def load_bev_parameters(path: Path) -> BevParameters:
    if not path.is_file():
        raise FileNotFoundError(f"BEV parameter file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"homography", "warp_width", "warp_height"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"BEV parameter keys missing: {sorted(missing)}")

        homography = np.asarray(data["homography"], dtype=np.float64)
        width = _scalar_int(data, "warp_width")
        height = _scalar_int(data, "warp_height")
        coordinate_space = str(np.asarray(
            data["coordinate_space"]
        ).reshape(-1)[0]) if "coordinate_space" in data.files else None

    if homography.shape != (3, 3) or width <= 0 or height <= 0:
        raise ValueError(f"Invalid BEV parameters: {path}")
    if coordinate_space not in (None, "full_undistorted_camera_image"):
        raise ValueError(
            "This homography is not defined for a full undistorted camera image: "
            f"{coordinate_space}"
        )
    return BevParameters(homography, width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw camera video to BEV and run lane post-processing."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--bev-params", type=Path, default=DEFAULT_BEV_PARAMS)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--left-class-id", type=int, default=0)
    parser.add_argument("--right-class-id", type=int, default=1)
    parser.add_argument(
        "--side-assignment",
        choices=("position", "class"),
        default="position",
        help="Assign lanes by BEV x-position or by YOLO class id.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the completed output video in a separate OpenCV window.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    calibration = load_calibration(args.calibration)
    bev_parameters = load_bev_parameters(args.bev_params)
    transformer = BevTransformer(calibration, bev_parameters)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps > 0.0 else 30.0
    output_fps = source_fps / args.frame_stride
    video_width = bev_parameters.width + bev_parameters.width % 2
    video_height = bev_parameters.height + bev_parameters.height % 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (video_width, video_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {args.output}")

    processor = Yolo11LanePostProcessor(
        model_path=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        left_class_id=args.left_class_id,
        right_class_id=args.right_class_id,
        side_assignment=args.side_assignment,
    )

    source_frame = 0
    processed_frames = 0
    left_detected = 0
    right_detected = 0
    try:
        while True:
            ok, raw_frame = capture.read()
            if not ok:
                break
            if source_frame % args.frame_stride != 0:
                source_frame += 1
                continue
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            bev_frame = transformer.transform(raw_frame)
            result = processor.process(
                bev_frame,
                dt=args.frame_stride / source_fps,
            )
            overlay = result["visualized_image"]
            if (video_width, video_height) != (bev_parameters.width, bev_parameters.height):
                padded = np.zeros((video_height, video_width, 3), dtype=overlay.dtype)
                padded[: overlay.shape[0], : overlay.shape[1]] = overlay
                overlay = padded
            writer.write(overlay)
            processed_frames += 1
            left_detected += int(result["left_detected"])
            right_detected += int(result["right_detected"])
            if processed_frames == 1 or processed_frames % 30 == 0:
                print(
                    f"processed {processed_frames} frames",
                    flush=True,
                )

            if args.display:
                cv2.imshow("lane verification BEV", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            source_frame += 1
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"source_video: {args.video}")
    print(f"processed_frames: {processed_frames}")
    print(f"left_detected: {left_detected}/{processed_frames}")
    print(f"right_detected: {right_detected}/{processed_frames}")
    print(f"bev_size: {bev_parameters.width}x{bev_parameters.height}")
    print(f"output_size: {video_width}x{video_height}")
    print(f"output_video: {args.output.resolve()}")

    if args.play:
        play_video(args.output)


def play_video(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open output video for playback: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, int(round(1000.0 / (fps if fps > 0.0 else 24.0))))
    window_name = "Lane verification BEV"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    run(parse_args())
