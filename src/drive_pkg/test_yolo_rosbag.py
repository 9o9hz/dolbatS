#!/usr/bin/env python3
"""Run the trained YOLO lane model on a ROS2 bag or a USB camera."""

from __future__ import annotations

import argparse
from collections import Counter
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from view_bev_rosbag import image_from_blob
    from view_bev_video import load_bev_parameters, load_calibration, to_bev
except ModuleNotFoundError:
    from src.drive_pkg.view_bev_rosbag import image_from_blob
    from src.drive_pkg.view_bev_video import (
        load_bev_parameters,
        load_calibration,
        to_bev,
    )


DEFAULT_BAG = Path(
    "/home/tak/bag/rosbag2_02/rosbag2_2026_07_23-14_55_06_0.db3"
)
DEFAULT_MODEL = Path(
    "/home/tak/yolo11/data/lane/runs/detect/train/weights/best.pt"
)
DEFAULT_CALIBRATION = Path("/home/tak/dolbatS/camera_calibration.npz")
DEFAULT_BEV_PARAMS = Path("/home/tak/dolbatS/bev_params_parallel_marker.npz.npz")
DEFAULT_TOPIC = "/image_raw/compressed"
DEFAULT_ROS_CAMERA_TOPIC = "/camera1/image_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the trained YOLO lane model on ROS2 bag or USB camera "
            "frames in BEV space."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Open a USB camera by OpenCV index, for example 0 or 1.",
    )
    camera_group.add_argument(
        "--camera-device",
        default=None,
        help="Open a camera device path, for example /dev/video0.",
    )
    camera_group.add_argument(
        "--ros-topic",
        default=None,
        help=(
            "Subscribe to a sensor_msgs/Image topic from usb_cam, "
            f"for example {DEFAULT_ROS_CAMERA_TOPIC}."
        ),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="Requested USB camera width. Use 640 with the supplied calibration.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Requested USB camera height. Use 480 with the supplied calibration.",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=None,
        help="Requested USB camera FPS and output-video FPS fallback.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--bev-params", type=Path, default=DEFAULT_BEV_PARAMS)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display-scale", type=float, default=3.0)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run inference and optionally save output without opening a window.",
    )
    return parser.parse_args()


def annotate_detection(
    model: YOLO,
    bev: np.ndarray,
    confidence: float,
    frame_label: str,
    detections: Counter,
) -> np.ndarray:
    """Run YOLO on one BEV image and return the annotated image."""
    result = model(bev, conf=confidence, verbose=False)[0]
    annotated = result.plot(labels=True, boxes=True, conf=True)

    if result.boxes is not None and len(result.boxes) > 0:
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for class_id in classes:
            detections[model.names[int(class_id)]] += 1

    cv2.putText(
        annotated,
        frame_label,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def image_from_ros_message(message: object) -> np.ndarray:
    """Convert a sensor_msgs/Image message to an OpenCV BGR image."""
    encoding = str(message.encoding).lower()
    if encoding in ("bgr8", "8uc3"):
        channels = 3
        convert_code = None
    elif encoding == "rgb8":
        channels = 3
        convert_code = cv2.COLOR_RGB2BGR
    elif encoding == "bgra8":
        channels = 4
        convert_code = cv2.COLOR_BGRA2BGR
    elif encoding == "rgba8":
        channels = 4
        convert_code = cv2.COLOR_RGBA2BGR
    elif encoding == "mono8":
        channels = 1
        convert_code = cv2.COLOR_GRAY2BGR
    else:
        raise ValueError(f"Unsupported ROS image encoding: {message.encoding}")

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    row_bytes = width * channels
    if height <= 0 or width <= 0 or step < row_bytes:
        raise ValueError(
            f"Invalid ROS image shape: {width}x{height}, step={step}, "
            f"encoding={message.encoding}"
        )

    data = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = height * step
    if data.size < required_bytes:
        raise ValueError(
            f"ROS image data is too short: {data.size} < {required_bytes}"
        )
    rows = data[:required_bytes].reshape(height, step)[:, :row_bytes]
    image = rows.reshape(height, width, channels).copy()
    if convert_code is not None:
        image = cv2.cvtColor(image, convert_code)
    return image


def run_ros_topic(args: argparse.Namespace) -> None:
    """Subscribe to usb_cam's sensor_msgs/Image topic and run YOLO."""
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    if not args.ros_topic:
        raise ValueError("--ros-topic is required")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    calibration = load_calibration(args.calibration)
    bev_parameters = load_bev_parameters(args.bev_params)
    model = YOLO(str(args.model))
    output_fps = args.camera_fps if args.camera_fps is not None else 30.0
    writer = (
        make_writer(
            args.output,
            bev_parameters.width,
            bev_parameters.height,
            output_fps / args.frame_stride,
        )
        if args.output is not None
        else None
    )
    detections = Counter()
    window_name = "YOLO lane test on usb_cam (BEV)"

    class YoloCameraNode(Node):
        def __init__(self) -> None:
            super().__init__("yolo_usb_cam_test")
            self.source_index = 0
            self.processed_frames = 0
            self.stop_requested = False
            self.subscription = self.create_subscription(
                Image,
                args.ros_topic,
                self.on_image,
                10,
            )
            self.stop_timer = self.create_timer(0.1, self.stop_if_requested)

        def on_image(self, message: Image) -> None:
            if self.stop_requested:
                return
            if self.source_index % args.frame_stride != 0:
                self.source_index += 1
                return
            try:
                image = image_from_ros_message(message)
                bev = to_bev(image, calibration, bev_parameters)
                annotated = annotate_detection(
                    model,
                    bev,
                    args.confidence,
                    f"frame {self.source_index + 1}",
                    detections,
                )
                if writer is not None:
                    writer.write(pad_for_video(annotated))
                if not args.no_display:
                    preview = cv2.resize(
                        annotated,
                        None,
                        fx=args.display_scale,
                        fy=args.display_scale,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        self.stop_requested = True
                self.processed_frames += 1
                if (
                    args.max_frames is not None
                    and self.processed_frames >= args.max_frames
                ):
                    self.stop_requested = True
                if self.processed_frames == 1 or self.processed_frames % 30 == 0:
                    print(f"processed {self.processed_frames} frames", flush=True)
            except Exception as exc:
                self.get_logger().error(str(exc))
                self.stop_requested = True
            finally:
                self.source_index += 1

        def stop_if_requested(self) -> None:
            if self.stop_requested:
                rclpy.shutdown()

    rclpy.init()
    node = YoloCameraNode()
    try:
        node.get_logger().info(f"Subscribing to {args.ros_topic}")
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        processed_frames = node.processed_frames
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyWindow(window_name)

    print(f"source: ROS topic {args.ros_topic}")
    print(f"model: {args.model}")
    print(f"processed_frames: {processed_frames}")
    print(f"detections: {dict(detections)}")
    print(f"bev_size: {bev_parameters.width}x{bev_parameters.height}")
    if args.output is not None:
        print(f"output_video: {args.output.resolve()}")


def camera_source(args: argparse.Namespace) -> int | str:
    if args.camera_device is not None:
        return args.camera_device
    if args.camera_index is not None:
        return args.camera_index
    raise ValueError("A camera index or device path is required")


def run_camera(args: argparse.Namespace) -> None:
    """Capture USB frames, transform them to BEV, and run YOLO in real time."""
    if args.camera_width is not None or args.camera_height is not None:
        if args.camera_width is None or args.camera_height is None:
            raise ValueError("--camera-width and --camera-height must be used together")
        if args.camera_width < 1 or args.camera_height < 1:
            raise ValueError("Camera width and height must be positive")
    if args.camera_fps is not None and args.camera_fps <= 0.0:
        raise ValueError("--camera-fps must be greater than zero")

    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    calibration = load_calibration(args.calibration)
    bev_parameters = load_bev_parameters(args.bev_params)
    model = YOLO(str(args.model))

    source = camera_source(args)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open USB camera: {source}")

    if args.camera_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps is not None:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if (actual_width, actual_height) != (calibration.width, calibration.height):
        capture.release()
        raise ValueError(
            "USB camera resolution does not match calibration: "
            f"{actual_width}x{actual_height} != "
            f"{calibration.width}x{calibration.height}. "
            "Use --camera-width 640 --camera-height 480 or recalibrate the camera."
        )

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0.0:
        source_fps = args.camera_fps if args.camera_fps is not None else 30.0

    writer: cv2.VideoWriter | None = None
    if args.output is not None:
        writer = make_writer(
            args.output,
            bev_parameters.width,
            bev_parameters.height,
            source_fps / args.frame_stride,
        )

    window_name = "YOLO lane test on USB camera (BEV)"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            max(1, int(bev_parameters.width * args.display_scale)),
            max(1, int(bev_parameters.height * args.display_scale)),
        )

    source_index = 0
    processed_frames = 0
    detections = Counter()
    try:
        while args.max_frames is None or processed_frames < args.max_frames:
            ok, image = capture.read()
            if not ok:
                print("Failed to read a frame from the USB camera", flush=True)
                break
            if source_index % args.frame_stride != 0:
                source_index += 1
                continue

            bev = to_bev(image, calibration, bev_parameters)
            annotated = annotate_detection(
                model,
                bev,
                args.confidence,
                f"frame {source_index + 1}",
                detections,
            )
            if writer is not None:
                writer.write(pad_for_video(annotated))

            processed_frames += 1
            if not args.no_display:
                preview = cv2.resize(
                    annotated,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            source_index += 1

            if processed_frames == 1 or processed_frames % 30 == 0:
                print(f"processed {processed_frames} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyWindow(window_name)

    print(f"source: USB camera {source}")
    print(f"model: {args.model}")
    print(f"processed_frames: {processed_frames}")
    print(f"detections: {dict(detections)}")
    print(f"camera_size: {actual_width}x{actual_height}")
    print(f"bev_size: {bev_parameters.width}x{bev_parameters.height}")
    if args.output is not None:
        print(f"output_video: {args.output.resolve()}")


def estimate_fps(cursor: sqlite3.Cursor, topic_id: int) -> float:
    timestamps = [
        int(row[0])
        for row in cursor.execute(
            "select timestamp from messages where topic_id = ? order by timestamp limit 30",
            (topic_id,),
        )
    ]
    if len(timestamps) < 2:
        return 15.0
    delta = float(np.median(np.diff(np.asarray(timestamps, dtype=np.int64))))
    return 1.0e9 / delta if delta > 0.0 else 15.0


def make_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_width = width + width % 2
    output_height = height + height % 2
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def pad_for_video(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    output = np.zeros(
        (height + height % 2, width + width % 2, 3),
        dtype=image.dtype,
    )
    output[:height, :width] = image
    return output


def run(args: argparse.Namespace) -> None:
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")
    if args.confidence < 0.0 or args.confidence > 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if args.display_scale <= 0.0:
        raise ValueError("--display-scale must be greater than 0")

    if args.ros_topic is not None:
        run_ros_topic(args)
        return
    if args.camera_index is not None or args.camera_device is not None:
        run_camera(args)
        return

    if not args.bag.is_file():
        raise FileNotFoundError(f"Bag file not found: {args.bag}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    calibration = load_calibration(args.calibration)
    bev_parameters = load_bev_parameters(args.bev_params)
    model = YOLO(str(args.model))
    connection = sqlite3.connect(str(args.bag))
    writer: cv2.VideoWriter | None = None
    window_name = "YOLO lane test on BEV"

    try:
        cursor = connection.cursor()
        topic_row = cursor.execute(
            "select id, type from topics where name = ?",
            (args.topic,),
        ).fetchone()
        if topic_row is None:
            raise KeyError(f"Topic not found: {args.topic}")
        topic_id, topic_type = topic_row
        if topic_type != "sensor_msgs/msg/CompressedImage":
            raise TypeError(f"Unsupported topic type: {topic_type}")

        total_frames = int(cursor.execute(
            "select count(*) from messages where topic_id = ?",
            (topic_id,),
        ).fetchone()[0])
        source_fps = estimate_fps(cursor, topic_id)
        if args.output is not None:
            writer = make_writer(
                args.output,
                bev_parameters.width,
                bev_parameters.height,
                source_fps / args.frame_stride,
            )

        if not args.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                window_name,
                int(bev_parameters.width * args.display_scale),
                int(bev_parameters.height * args.display_scale),
            )

        rows = cursor.execute(
            """
            select timestamp, data
            from messages
            where topic_id = ?
            order by timestamp
            """,
            (topic_id,),
        )
        source_index = 0
        processed_frames = 0
        detections = Counter()
        previous_timestamp: int | None = None

        for timestamp_ns, blob in rows:
            if source_index % args.frame_stride != 0:
                source_index += 1
                continue
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break

            image = image_from_blob(blob)
            bev = to_bev(image, calibration, bev_parameters)
            annotated = annotate_detection(
                model,
                bev,
                args.confidence,
                f"frame {source_index + 1}/{total_frames}",
                detections,
            )
            if writer is not None:
                writer.write(pad_for_video(annotated))

            processed_frames += 1
            if not args.no_display:
                preview = cv2.resize(
                    annotated,
                    None,
                    fx=args.display_scale,
                    fy=args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow(window_name, preview)
                if previous_timestamp is None:
                    delay_ms = 33
                else:
                    delta_ms = max(
                        1.0,
                        (timestamp_ns - previous_timestamp) / 1_000_000.0,
                    )
                    delay_ms = max(1, int(round(delta_ms / args.playback_rate)))
                previous_timestamp = timestamp_ns
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):
                    break
            source_index += 1

            if processed_frames == 1 or processed_frames % 30 == 0:
                print(f"processed {processed_frames} frames", flush=True)

        print(f"model: {args.model}")
        print(f"topic: {args.topic}")
        print(f"processed_frames: {processed_frames}/{total_frames}")
        print(f"detections: {dict(detections)}")
        print(f"bev_size: {bev_parameters.width}x{bev_parameters.height}")
        if args.output is not None:
            print(f"output_video: {args.output.resolve()}")
    finally:
        connection.close()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    run(parse_args())
