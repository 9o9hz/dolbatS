#!/usr/bin/env python3
"""Measure BEV distortion from a checkerboard visible on a ROS camera topic.

The tool applies optional camera undistortion and a supplied BEV NPZ, detects
checkerboard inner corners, and reports:

* horizontal/vertical square-size ratio error;
* horizontal and vertical spacing variation (CV);
* residual perspective scale change across rows/columns;
* checkerboard orthogonality error.

Controls:
    q / ESC: quit
    s: save the current annotated image and JSON report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Optional, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEV = (
    PROJECT_ROOT
    / "src"
    / "lane_vision_pkg"
    / "config"
    / "bev_params_y_auto.npz"
)
DEFAULT_CALIBRATION = (
    PROJECT_ROOT
    / "src"
    / "lane_vision_pkg"
    / "config"
    / "camera_calibration.npz"
)


def _scalar(data, names: Sequence[str], default: int) -> int:
    for name in names:
        if name in data.files:
            return int(data[name])
    return int(default)


def load_bev(path: Path):
    with np.load(path, allow_pickle=False) as data:
        width = _scalar(data, ("warp_width", "warp_w"), 640)
        height = _scalar(data, ("warp_height", "warp_h"), 640)
        calibration_width = _scalar(
            data, ("calibration_width",), 640
        )
        calibration_height = _scalar(
            data, ("calibration_height",), 480
        )
        if "homography" in data.files:
            homography = np.asarray(data["homography"], np.float64)
            source_points = None
            destination_points = None
        elif "src_points" in data.files and "dst_points" in data.files:
            source_points = np.asarray(data["src_points"], np.float32)
            destination_points = np.asarray(
                data["dst_points"], np.float32
            )
            homography = None
        else:
            raise KeyError(
                "BEV NPZ requires homography or src_points/dst_points"
            )

        if "checkerboard_size" in data.files:
            pattern = tuple(
                int(value) for value in data["checkerboard_size"].flat
            )
        else:
            pattern = (10, 7)
        square_size_mm = float(
            data["square_size_mm"]
            if "square_size_mm" in data.files
            else 25.0
        )

    return {
        "width": width,
        "height": height,
        "calibration_width": calibration_width,
        "calibration_height": calibration_height,
        "homography": homography,
        "source_points": source_points,
        "destination_points": destination_points,
        "pattern": pattern,
        "square_size_mm": square_size_mm,
    }


def load_calibration(path: Optional[Path]):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["camera_matrix"], np.float64),
            np.asarray(data["distortion_coefficients"], np.float64),
            np.asarray(
                data.get("new_camera_matrix", data["camera_matrix"]),
                np.float64,
            ),
            _scalar(data, ("image_width",), 640),
            _scalar(data, ("image_height",), 480),
        )


def decode_raw(message: Image) -> np.ndarray:
    height, width, step = (
        int(message.height),
        int(message.width),
        int(message.step),
    )
    rows = np.frombuffer(message.data, np.uint8)[: height * step].reshape(
        height, step
    )
    encoding = str(message.encoding).lower()
    if encoding in ("bgr8", "rgb8", "8uc3"):
        frame = rows[:, : width * 3].reshape(height, width, 3)
        return (
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if encoding == "rgb8"
            else frame.copy()
        )
    if encoding in ("mono8", "8uc1"):
        mono = rows[:, :width].reshape(height, width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if encoding in ("yuyv", "yuv422", "yuv422_yuy2"):
        yuyv = rows[:, : width * 2].reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f"Unsupported raw image encoding: {message.encoding}")


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return 100.0 * float(np.std(values)) / mean if mean > 1e-9 else float("nan")


def analyze_checkerboard(
    corners: np.ndarray,
    pattern: tuple[int, int],
) -> dict:
    columns, rows = pattern
    grid = corners.reshape(rows, columns, 2)
    horizontal_vectors = grid[:, 1:] - grid[:, :-1]
    vertical_vectors = grid[1:] - grid[:-1]
    horizontal_spacing = np.linalg.norm(horizontal_vectors, axis=2)
    vertical_spacing = np.linalg.norm(vertical_vectors, axis=2)
    horizontal_mean = float(np.mean(horizontal_spacing))
    vertical_mean = float(np.mean(vertical_spacing))

    horizontal_unit = horizontal_vectors / np.maximum(
        horizontal_spacing[..., None], 1e-9
    )
    vertical_unit = vertical_vectors / np.maximum(
        vertical_spacing[..., None], 1e-9
    )
    # Each cell has a horizontal top edge and vertical left edge.
    dots = np.sum(
        horizontal_unit[:-1, :, :] * vertical_unit[:, :-1, :],
        axis=2,
    )
    angle_errors = np.abs(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))) - 90.0)

    row_mean_spacing = np.mean(horizontal_spacing, axis=1)
    column_mean_spacing = np.mean(vertical_spacing, axis=0)
    ratio = horizontal_mean / max(vertical_mean, 1e-9)
    return {
        "detected": True,
        "horizontal_spacing_px": horizontal_mean,
        "vertical_spacing_px": vertical_mean,
        "square_ratio_x_over_y": ratio,
        "square_ratio_error_percent": abs(ratio - 1.0) * 100.0,
        "horizontal_spacing_cv_percent": coefficient_of_variation(
            horizontal_spacing
        ),
        "vertical_spacing_cv_percent": coefficient_of_variation(
            vertical_spacing
        ),
        "row_scale_change_percent": (
            (float(np.max(row_mean_spacing)) - float(np.min(row_mean_spacing)))
            / max(float(np.mean(row_mean_spacing)), 1e-9)
            * 100.0
        ),
        "column_scale_change_percent": (
            (
                float(np.max(column_mean_spacing))
                - float(np.min(column_mean_spacing))
            )
            / max(float(np.mean(column_mean_spacing)), 1e-9)
            * 100.0
        ),
        "mean_orthogonality_error_deg": float(np.mean(angle_errors)),
        "max_orthogonality_error_deg": float(np.max(angle_errors)),
    }


def draw_report(
    image: np.ndarray,
    corners: np.ndarray,
    pattern: tuple[int, int],
    report: dict,
) -> np.ndarray:
    output = image.copy()
    cv2.drawChessboardCorners(
        output,
        pattern,
        corners.reshape(-1, 1, 2),
        True,
    )
    lines = [
        f"X/Y ratio: {report['square_ratio_x_over_y']:.4f} "
        f"(error {report['square_ratio_error_percent']:.2f}%)",
        f"spacing CV: X {report['horizontal_spacing_cv_percent']:.2f}% "
        f"Y {report['vertical_spacing_cv_percent']:.2f}%",
        f"scale change: row {report['row_scale_change_percent']:.2f}% "
        f"col {report['column_scale_change_percent']:.2f}%",
        f"orthogonality: mean {report['mean_orthogonality_error_deg']:.2f}deg "
        f"max {report['max_orthogonality_error_deg']:.2f}deg",
        "S: save report | Q/ESC: quit",
    ]
    for index, text in enumerate(lines):
        y = 28 + index * 28
        cv2.putText(
            output,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return output


class BevCheckerboardDistortionNode(Node):
    def __init__(self, arguments: argparse.Namespace) -> None:
        super().__init__("bev_checkerboard_distortion")
        self.arguments = arguments
        self.bev = load_bev(arguments.bev)
        if arguments.pattern is not None:
            self.bev["pattern"] = tuple(arguments.pattern)
        self.calibration = load_calibration(arguments.calibration)
        self.last_report: Optional[dict] = None
        self.last_annotated: Optional[np.ndarray] = None
        self.last_log_time = 0.0

        message_type = (
            CompressedImage if arguments.transport == "compressed" else Image
        )
        self.subscription = self.create_subscription(
            message_type,
            arguments.topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"topic={arguments.topic}, BEV={arguments.bev}, "
            f"checkerboard inner corners={self.bev['pattern']}"
        )

    def _decode(self, message) -> np.ndarray:
        if isinstance(message, CompressedImage):
            frame = cv2.imdecode(
                np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise ValueError("Could not decode compressed image")
            return frame
        return decode_raw(message)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        if self.calibration is None:
            return frame
        camera, distortion, new_camera, width, height = self.calibration
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_AREA
            )
        return cv2.undistort(frame, camera, distortion, None, new_camera)

    def _warp(self, frame: np.ndarray) -> np.ndarray:
        input_width = self.bev["calibration_width"]
        input_height = self.bev["calibration_height"]
        if frame.shape[1] != input_width or frame.shape[0] != input_height:
            frame = cv2.resize(
                frame,
                (input_width, input_height),
                interpolation=cv2.INTER_AREA,
            )
        if self.bev["homography"] is not None:
            matrix = self.bev["homography"]
        else:
            matrix = cv2.getPerspectiveTransform(
                self.bev["source_points"],
                self.bev["destination_points"],
            )
        return cv2.warpPerspective(
            frame,
            matrix,
            (self.bev["width"], self.bev["height"]),
        )

    def on_image(self, message) -> None:
        try:
            original = self._decode(message)
            corrected = self._undistort(original)
            bev_image = self._warp(corrected)
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray,
                self.bev["pattern"],
                flags=(
                    cv2.CALIB_CB_NORMALIZE_IMAGE
                    | cv2.CALIB_CB_EXHAUSTIVE
                    | cv2.CALIB_CB_ACCURACY
                ),
            )
        except Exception as exc:
            self.get_logger().error(
                f"Frame analysis failed: {exc}", throttle_duration_sec=2.0
            )
            return

        if found:
            points = corners.reshape(-1, 2)
            self.last_report = analyze_checkerboard(
                points, self.bev["pattern"]
            )
            self.last_report.update(
                {
                    "bev_file": str(self.arguments.bev),
                    "checkerboard_inner_corners": list(self.bev["pattern"]),
                    "square_size_mm": self.bev["square_size_mm"],
                    "bev_size": [self.bev["width"], self.bev["height"]],
                }
            )
            self.last_annotated = draw_report(
                bev_image, points, self.bev["pattern"], self.last_report
            )
            now = time.monotonic()
            if now - self.last_log_time >= 1.0:
                self.get_logger().info(
                    "ratio error="
                    f"{self.last_report['square_ratio_error_percent']:.2f}%, "
                    "spacing CV X/Y="
                    f"{self.last_report['horizontal_spacing_cv_percent']:.2f}/"
                    f"{self.last_report['vertical_spacing_cv_percent']:.2f}%, "
                    "max angle error="
                    f"{self.last_report['max_orthogonality_error_deg']:.2f}deg"
                )
                self.last_log_time = now
        else:
            self.last_report = None
            self.last_annotated = bev_image.copy()
            cv2.putText(
                self.last_annotated,
                f"Checkerboard {self.bev['pattern']} not detected",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("ROS camera (undistorted)", corrected)
        cv2.imshow("BEV checkerboard distortion", self.last_annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()
        elif key == ord("s"):
            self.save_result()

    def save_result(self) -> None:
        if self.last_report is None or self.last_annotated is None:
            self.get_logger().warning(
                "No detected checkerboard result to save"
            )
            return
        self.arguments.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.arguments.output_dir / "bev_distortion_analysis.jpg"
        report_path = self.arguments.output_dir / "bev_distortion_report.json"
        cv2.imwrite(str(image_path), self.last_annotated)
        report_path.write_text(
            json.dumps(self.last_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.get_logger().info(f"Saved {image_path} and {report_path}")

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Check BEV distortion using a ROS camera checkerboard."
    )
    parser.add_argument(
        "--topic", default="/camera/lane/raw/compressed"
    )
    parser.add_argument(
        "--transport", choices=("compressed", "raw"), default="compressed"
    )
    parser.add_argument("--bev", type=Path, default=DEFAULT_BEV)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Use 'none' to disable lens undistortion.",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "bev_distortion_result",
    )
    arguments, ros_arguments = parser.parse_known_args(argv)
    if str(arguments.calibration).lower() == "none":
        arguments.calibration = None
    for name in ("bev", "calibration", "output_dir"):
        value = getattr(arguments, name)
        if value is not None:
            setattr(arguments, name, value.expanduser().resolve())
    if not arguments.bev.is_file():
        parser.error(f"BEV NPZ not found: {arguments.bev}")
    if (
        arguments.calibration is not None
        and not arguments.calibration.is_file()
    ):
        parser.error(
            f"Camera calibration NPZ not found: {arguments.calibration}"
        )
    return arguments, ros_arguments


def main(argv=None) -> None:
    arguments, ros_arguments = parse_arguments(argv)
    rclpy.init(args=ros_arguments)
    node = BevCheckerboardDistortionNode(arguments)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
