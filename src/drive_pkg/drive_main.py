#!/usr/bin/env python3
"""Integrated ROS 2 node: camera image -> lane path -> cmd_vel, one process.

Structurally modeled after ``yolotl_ros2``'s single-node style (argparse +
package-share weight loading, an optional undistort step, and preprocess/
postprocess hooks reserved for future mission logic), but the lane
detection, path generation, and steering *algorithms* are unchanged: this
module only wires together the existing, still-independently-runnable
``lane_detect`` / ``path_plan`` / ``pure_pursuit`` building blocks inside one
image callback instead of three topic-connected nodes.

Input:
    /image_raw/compressed (sensor_msgs/CompressedImage)

Outputs (identical topics/payloads to the split-node pipeline, so
``path_visualizer`` keeps working unchanged):
    /lane/detection/bev/compressed
    /lane/detection/mask/compressed
    /lane/detection/segmentation/compressed
    /lane/detection/status
    /lane/detection/instances
    /lane/path                         (nav_msgs/Path)
    /lane/path/debug/compressed
    /lane/path/status
    /which/lane
    /cmd_vel
    /lane/control/status
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as PathMessage
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from lane_detect import LaneDetectorCore, DetectionOutput
from lane_processing import (
    CameraCalibration,
    DEFAULT_BEV_PARAMS,
    DEFAULT_MODEL,
    LaneConfig,
    PathPlanResult,
    SegmentationLaneProcessor,
    decode_compressed_image,
    load_bev_parameters,
    load_calibration,
    undistort_frame,
)
from pure_pursuit import PurePursuitController, SteeringCommand


def _default_calibration_path() -> Path:
    """Workspace-root ``camera_calibration.npz`` when run from source.

    Not shipped through ``setup.py``'s ``data_files`` (see howtorun.md), so
    this only resolves when running un-installed from the source tree; an
    installed run falls back to an explicit ``--calib-file``/``calib_file``
    override, or simply runs without undistortion.
    """
    return Path(__file__).resolve().parents[2] / "camera_calibration.npz"


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated lane-follow node: detection, path planning, and "
            "Pure Pursuit control in a single ROS 2 node/callback."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLO segmentation weights (.pt); overrides model_path.",
    )
    parser.add_argument(
        "--bev-params",
        type=Path,
        default=DEFAULT_BEV_PARAMS,
        help="NPZ containing src_points, dst_points, warp_w, warp_h.",
    )
    parser.add_argument(
        "--calib-file",
        type=Path,
        default=_default_calibration_path(),
        help=(
            "Optional camera_calibration.npz for undistortion; missing "
            "or invalid files are skipped with a warning."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--enable-drive",
        action="store_true",
        help="Allow non-zero cmd_vel output; default is safe dry-run.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show local cv2.imshow debug windows in addition to topics.",
    )
    return parser.parse_known_args(argv)


class LaneDriveNode(Node):
    """Detect -> plan -> control for one frame, all in one callback."""

    def __init__(self, cli_args: argparse.Namespace) -> None:
        super().__init__("drive_main")

        defaults = {
            "model_path": str(cli_args.weights),
            "bev_params": str(cli_args.bev_params),
            "calib_file": str(cli_args.calib_file),
            "use_undistort": True,
            "image_topic": "/image_raw/compressed",
            "bev_topic": "/lane/detection/bev/compressed",
            "mask_topic": "/lane/detection/mask/compressed",
            "segmentation_topic": "/lane/detection/segmentation/compressed",
            "detection_status_topic": "/lane/detection/status",
            "instances_topic": "/lane/detection/instances",
            "confidence": float(cli_args.confidence),
            "image_size": 640,
            "device": str(cli_args.device),
            "calibration_width": 640,
            "calibration_height": 480,
            "process_every_nth_frame": 1,
            "pixels_per_meter": 600.0,
            "lane_width_m": 0.90,
            "bev_reference_forward_offset_m": 1.04,
            "min_component_area": 250,
            "center_sample_step": 5,
            "same_line_threshold_px": 75.0,
            "min_group_span_px": 80.0,
            "min_group_area": 500,
            "dashed_piece_threshold": 2,
            "prefer_solid_when_dashed": True,
            "lane_track_max_age_frames": 7,
            "lane_track_match_threshold_px": 90.0,
            "solid_enter_frames": 4,
            "solid_exit_frames": 6,
            "path_top_y": 180,
            "path_bottom_margin": 30,
            "path_step_px": 10,
            "path_resample_step_px": 5,
            "path_ema_alpha": 0.35,
            "path_transition_blend_frames": 6,
            "max_path_lateral_step_m": 0.04,
            "max_missing_frames": 8,
            "path_topic": "/lane/path",
            "debug_topic": "/lane/path/debug/compressed",
            "path_status_topic": "/lane/path/status",
            "which_lane_topic": "/which/lane",
            "path_frame_id": "base_link",
            "cmd_vel_topic": "/cmd_vel",
            "control_status_topic": "/lane/control/status",
            "enable_drive": bool(cli_args.enable_drive),
            "speed_mps": 0.18,
            "hold_speed_scale": 0.75,
            "turn_speed_min_scale": 0.50,
            "turn_speed_reduction": 0.70,
            "wheelbase_m": 0.545,
            "max_steer_deg": 18.0,
            "lookahead_m": -1.0,
            "lookahead_min_m": 1.10,
            "lookahead_max_m": 2.00,
            "steering_ema_alpha": 0.35,
            "steering_deadband_deg": 0.8,
            "max_steering_change_deg": 3.0,
            "path_timeout_sec": 0.5,
            "debug_jpeg_quality": 80,
            "local_display": bool(cli_args.display),
            "display_scale": 1.0,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        model_path = Path(str(parameter("model_path")))
        bev_value = str(parameter("bev_params")).strip()
        bev_path = Path(bev_value) if bev_value else DEFAULT_BEV_PARAMS
        bev = load_bev_parameters(bev_path)

        self.detector = LaneDetectorCore(
            model_path=model_path,
            bev_path=bev_path,
            calibration_width=int(parameter("calibration_width")),
            calibration_height=int(parameter("calibration_height")),
            confidence=float(parameter("confidence")),
            image_size=int(parameter("image_size")),
            device_request=str(parameter("device")),
        )

        plan_config = LaneConfig(
            source_points=tuple(bev.source_points.reshape(-1)),
            destination_points=tuple(bev.destination_points.reshape(-1)),
            warp_width=bev.width,
            warp_height=bev.height,
            pixels_per_meter=float(parameter("pixels_per_meter")),
            lane_width_m=float(parameter("lane_width_m")),
            bev_reference_forward_offset_m=float(
                parameter("bev_reference_forward_offset_m")
            ),
            min_component_area=int(parameter("min_component_area")),
            center_sample_step=int(parameter("center_sample_step")),
            same_line_threshold_px=float(
                parameter("same_line_threshold_px")
            ),
            min_group_span_px=float(parameter("min_group_span_px")),
            min_group_area=int(parameter("min_group_area")),
            dashed_piece_threshold=int(
                parameter("dashed_piece_threshold")
            ),
            prefer_solid_when_dashed=bool(
                parameter("prefer_solid_when_dashed")
            ),
            lane_track_max_age_frames=int(
                parameter("lane_track_max_age_frames")
            ),
            lane_track_match_threshold_px=float(
                parameter("lane_track_match_threshold_px")
            ),
            solid_enter_frames=int(parameter("solid_enter_frames")),
            solid_exit_frames=int(parameter("solid_exit_frames")),
            path_top_y=int(parameter("path_top_y")),
            path_bottom_margin=int(parameter("path_bottom_margin")),
            path_step_px=int(parameter("path_step_px")),
            path_resample_step_px=int(
                parameter("path_resample_step_px")
            ),
            path_ema_alpha=float(parameter("path_ema_alpha")),
            path_transition_blend_frames=int(
                parameter("path_transition_blend_frames")
            ),
            max_path_lateral_step_m=float(
                parameter("max_path_lateral_step_m")
            ),
            max_missing_frames=int(parameter("max_missing_frames")),
        )
        self.processor = SegmentationLaneProcessor(None, plan_config)
        self.path_frame_id = str(parameter("path_frame_id"))
        self.debug_canvas_size = (bev.height, bev.width)

        self.enable_drive = bool(parameter("enable_drive"))
        self.controller = PurePursuitController(
            speed_mps=float(parameter("speed_mps")),
            hold_speed_scale=float(parameter("hold_speed_scale")),
            turn_speed_min_scale=float(
                parameter("turn_speed_min_scale")
            ),
            turn_speed_reduction=float(
                parameter("turn_speed_reduction")
            ),
            wheelbase_m=float(parameter("wheelbase_m")),
            lookahead_min_m=float(parameter("lookahead_min_m")),
            lookahead_max_m=float(parameter("lookahead_max_m")),
            fixed_lookahead_m=float(parameter("lookahead_m")),
            max_steer_deg=float(parameter("max_steer_deg")),
            steering_ema_alpha=float(parameter("steering_ema_alpha")),
            steering_deadband_deg=float(
                parameter("steering_deadband_deg")
            ),
            max_steering_change_deg=float(
                parameter("max_steering_change_deg")
            ),
        )
        self.path_timeout_sec = max(
            0.0, float(parameter("path_timeout_sec"))
        )

        self.calibration: Optional[CameraCalibration] = None
        self.use_undistort = bool(parameter("use_undistort"))
        calib_value = str(parameter("calib_file")).strip()
        if self.use_undistort and calib_value:
            try:
                self.calibration = load_calibration(Path(calib_value))
                self.get_logger().info(
                    f"Loaded camera calibration: {calib_value}"
                )
            except Exception as exc:
                self.use_undistort = False
                self.get_logger().warning(
                    "Could not load camera calibration "
                    f"({calib_value}): {exc}. Continuing without "
                    "undistortion."
                )
        else:
            self.use_undistort = False

        self.process_every_nth_frame = max(
            1, int(parameter("process_every_nth_frame"))
        )
        self.jpeg_quality = int(
            np.clip(int(parameter("debug_jpeg_quality")), 1, 100)
        )
        self.local_display = bool(parameter("local_display"))
        self.display_scale = max(0.1, float(parameter("display_scale")))
        self.display_window_ready = False
        self.frame_count = 0
        self.last_path_time = None
        self.timed_out = False

        image_topic = str(parameter("image_topic"))
        self.bev_publisher = self.create_publisher(
            CompressedImage, str(parameter("bev_topic")), qos_profile_sensor_data
        )
        self.mask_publisher = self.create_publisher(
            CompressedImage, str(parameter("mask_topic")), qos_profile_sensor_data
        )
        self.segmentation_publisher = self.create_publisher(
            CompressedImage,
            str(parameter("segmentation_topic")),
            qos_profile_sensor_data,
        )
        self.detection_status_publisher = self.create_publisher(
            String, str(parameter("detection_status_topic")), 10
        )
        self.instances_publisher = self.create_publisher(
            String, str(parameter("instances_topic")), 10
        )
        self.path_publisher = self.create_publisher(
            PathMessage, str(parameter("path_topic")), 10
        )
        self.debug_publisher = self.create_publisher(
            CompressedImage, str(parameter("debug_topic")), qos_profile_sensor_data
        )
        self.path_status_publisher = self.create_publisher(
            String, str(parameter("path_status_topic")), 10
        )
        self.which_lane_publisher = self.create_publisher(
            String, str(parameter("which_lane_topic")), 10
        )
        self.cmd_publisher = self.create_publisher(
            Twist,
            str(parameter("cmd_vel_topic")),
            10,
        )
        self.control_status_publisher = self.create_publisher(
            String, str(parameter("control_status_topic")), 10
        )
        self.image_subscription = self.create_subscription(
            CompressedImage,
            image_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.watchdog = self.create_timer(0.1, self.check_path_timeout)

        self.get_logger().info(
            f"{image_topic} -> /cmd_vel (integrated); model={model_path}, "
            f"bev={bev_path}, device={self.detector.device}, "
            f"undistort={self.use_undistort}, drive_enabled={self.enable_drive}"
        )
        if not self.enable_drive:
            self.get_logger().info(
                "Dry-run mode: pipeline runs end to end, but cmd_vel "
                "remains zero."
            )

    # ------------------------------------------------------------------
    # Mission hooks (structure only; no mission logic implemented yet).
    # ------------------------------------------------------------------
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Called right after undistortion, before BEV warp + detection.

        Future home for competition mission preprocessing (e.g. masking
        out arrow/sign regions before lane inference), mirroring
        yolotl_ros2's arrow-masking step. Currently a pass-through.
        """
        return frame

    def postprocess_result(
        self, command: SteeringCommand
    ) -> SteeringCommand:
        """Called after Pure Pursuit computes a steering command.

        Future home for mission logic that overrides speed/steering (stop
        line, traffic light, parking). Currently a pass-through.
        """
        return command

    # ------------------------------------------------------------------
    # Main pipeline.
    # ------------------------------------------------------------------
    def on_image(self, message: CompressedImage) -> None:
        self.frame_count += 1
        if (self.frame_count - 1) % self.process_every_nth_frame:
            return

        try:
            frame = decode_compressed_image(message)
            if self.use_undistort and self.calibration is not None:
                frame = undistort_frame(self.calibration, frame)
            frame = self.preprocess_frame(frame)
            self.last_path_time = self.get_clock().now()
            self.timed_out = False
            detection = self.detector.detect(frame)
        except Exception as exc:
            self.get_logger().error(
                f"Lane detection failed: {exc}",
                throttle_duration_sec=2.0,
            )
            self._publish_control(self.controller.stop("detection_failed"))
            return

        self._publish_detection(detection, message)

        try:
            plan = self.processor.plan_mask(detection.mask)
        except Exception as exc:
            self.get_logger().error(
                f"Path planning failed: {exc}",
                throttle_duration_sec=2.0,
            )
            self._publish_path(None, message)
            self._publish_control(
                self.controller.stop("path_planning_failed")
            )
            return

        self._publish_path(plan, message)
        self.controller.set_path_fallback(plan.used_fallback)
        points = (
            np.asarray(plan.path_meters, dtype=np.float64)
            if plan.path_meters is not None
            else None
        )
        command = (
            self.controller.compute(points)
            if points is not None
            else self.controller.stop(plan.reason)
        )
        command = self.postprocess_result(command)
        self._publish_control(command)

        if self.local_display:
            self._show_local_debug(detection, plan)

    # ------------------------------------------------------------------
    # Detection publishing (mirrors lane_detect.py's on_image payloads).
    # ------------------------------------------------------------------
    def _publish_detection(
        self,
        detection: DetectionOutput,
        source: CompressedImage,
    ) -> None:
        self._publish_image(
            detection.bev,
            source,
            self.bev_publisher,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        self._publish_image(
            detection.mask, source, self.mask_publisher, ".png", []
        )
        self._publish_image(
            detection.segmentation,
            source,
            self.segmentation_publisher,
            ".jpg",
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        self.detection_status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "lane_instances": len(detection.instances),
                        "mask_pixels": int(
                            np.count_nonzero(detection.mask)
                        ),
                        "inference_ms": round(detection.inference_ms, 1),
                    },
                    ensure_ascii=False,
                )
            )
        )
        timestamp_ns = (
            int(source.header.stamp.sec) * 1_000_000_000
            + int(source.header.stamp.nanosec)
        )
        self.instances_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "timestamp_ns": timestamp_ns,
                        "instances": detection.instances,
                    },
                    ensure_ascii=False,
                )
            )
        )

    # ------------------------------------------------------------------
    # Path publishing (mirrors path_plan.py's on_mask payloads).
    # ------------------------------------------------------------------
    def _publish_path(
        self,
        plan: Optional[PathPlanResult],
        source: CompressedImage,
    ) -> None:
        if plan is None:
            self.path_publisher.publish(self._path_message(None, source))
            return

        self.which_lane_publisher.publish(
            String(data=plan.which_lane or "unknown")
        )
        self.path_status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "path_valid": plan.path_meters is not None,
                        "reason": plan.reason,
                        "fallback": plan.used_fallback,
                        "lane_groups": plan.group_count,
                        "dashed_region_count": plan.dashed_region_count,
                        "selection_mode": plan.selection_mode,
                        "which_lane": plan.which_lane,
                        "detected_lines": plan.detected_lines,
                        "left_boundary_pixels": self._points_payload(
                            plan.left_boundary_pixels
                        ),
                        "right_boundary_pixels": self._points_payload(
                            plan.right_boundary_pixels
                        ),
                        "path_pixels": self._points_payload(
                            plan.path_pixels
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )
        self.path_publisher.publish(
            self._path_message(plan.path_meters, source)
        )
        debug = self._draw_plan_overlay(plan)
        success, encoded = cv2.imencode(
            ".jpg", debug, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if success:
            debug_message = CompressedImage()
            debug_message.header = source.header
            debug_message.format = "jpeg"
            debug_message.data = encoded.tobytes()
            self.debug_publisher.publish(debug_message)

    @staticmethod
    def _points_payload(points: Optional[np.ndarray]) -> list[list[float]]:
        if points is None or len(points) == 0:
            return []
        return np.round(
            np.asarray(points, dtype=np.float64), 1
        ).tolist()

    def _path_message(
        self,
        path_meters: Optional[np.ndarray],
        source: CompressedImage,
    ) -> PathMessage:
        message = PathMessage()
        message.header.stamp = source.header.stamp
        message.header.frame_id = self.path_frame_id
        if path_meters is None:
            return message

        for index, (forward, left) in enumerate(path_meters):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(forward)
            pose.pose.position.y = float(left)
            if len(path_meters) == 1:
                yaw = 0.0
            elif index < len(path_meters) - 1:
                delta = path_meters[index + 1] - path_meters[index]
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            else:
                delta = path_meters[index] - path_meters[index - 1]
                yaw = math.atan2(float(delta[1]), float(delta[0]))
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            message.poses.append(pose)
        return message

    def _draw_plan_overlay(self, plan: PathPlanResult) -> np.ndarray:
        # Reconstructed from the plan output (boundaries/path in pixel
        # space) rather than the raw mask, since drive_main keeps the
        # mask internal instead of round tripping it through a topic.
        debug = np.zeros((*self.debug_canvas_size, 3), dtype=np.uint8)
        for points, color in (
            (plan.left_boundary_pixels, (255, 0, 0)),
            (plan.right_boundary_pixels, (0, 255, 0)),
        ):
            if points is None or len(points) < 2:
                continue
            pts = np.rint(np.asarray(points)).astype(np.int32)
            cv2.polylines(debug, [pts], False, color, 2)
        if plan.path_pixels is not None and len(plan.path_pixels) >= 2:
            pts = np.rint(np.asarray(plan.path_pixels)).astype(np.int32)
            cv2.polylines(debug, [pts], False, (0, 0, 255), 4)
        lines = (
            f"path: {plan.reason}",
            f"selection: {plan.selection_mode}",
            f"groups: {plan.group_count} fallback: {plan.used_fallback}",
            f"which_lane: {plan.which_lane or 'unknown'}",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                debug,
                text,
                (12, 24 + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return debug

    # ------------------------------------------------------------------
    # Control publishing (mirrors pure_pursuit.py's on_path payloads).
    # ------------------------------------------------------------------
    def _publish_control(self, command: SteeringCommand) -> None:
        twist = (
            self.controller.make_twist(command)
            if self.enable_drive and command.path_valid
            else Twist()
        )
        self.cmd_publisher.publish(twist)
        self.control_status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "path_valid": command.path_valid,
                        "reason": command.reason,
                        "fallback": self.controller.path_fallback,
                        "drive_enabled": self.enable_drive,
                        "steering_deg": round(command.steering_deg, 2),
                        "lookahead_m": round(command.lookahead_m, 3),
                        "lookahead_target_m": round(
                            command.target_distance_m, 3
                        ),
                        "lookahead_target_index": int(
                            command.target_index
                        ),
                        "desired_speed_mps": round(
                            command.speed_mps, 3
                        ),
                        "published_linear_x": round(twist.linear.x, 3),
                        "published_angular_z": round(
                            twist.angular.z, 3
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )

    def check_path_timeout(self) -> None:
        if (
            self.last_path_time is None
            or self.timed_out
            or self.path_timeout_sec <= 0.0
        ):
            return
        elapsed = (
            self.get_clock().now() - self.last_path_time
        ).nanoseconds / 1e9
        if elapsed < self.path_timeout_sec:
            return
        self.timed_out = True
        self._publish_control(self.controller.stop("path_timeout"))
        self.get_logger().warning(
            f"No frame processed for {elapsed:.2f}s: vehicle stopped"
        )

    # ------------------------------------------------------------------
    # Optional local debug windows (yolotl-style, off by default).
    # ------------------------------------------------------------------
    def _show_local_debug(
        self,
        detection: DetectionOutput,
        plan: PathPlanResult,
    ) -> None:
        segmentation_preview = cv2.resize(
            detection.segmentation,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        path_preview = cv2.resize(
            self._draw_plan_overlay(plan),
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        if not self.display_window_ready:
            cv2.namedWindow("drive_main: segmentation", cv2.WINDOW_NORMAL)
            cv2.namedWindow("drive_main: path", cv2.WINDOW_NORMAL)
            self.display_window_ready = True
        cv2.imshow("drive_main: segmentation", segmentation_preview)
        cv2.imshow("drive_main: path", path_preview)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            rclpy.shutdown()

    @staticmethod
    def _publish_image(
        image: np.ndarray,
        source: CompressedImage,
        publisher: Any,
        extension: str,
        options: list[int],
    ) -> None:
        success, encoded = cv2.imencode(extension, image, options)
        if not success:
            raise RuntimeError(f"Could not encode {extension} image")
        output = CompressedImage()
        output.header = source.header
        output.format = "png" if extension == ".png" else "jpeg"
        output.data = encoded.tobytes()
        publisher.publish(output)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.cmd_publisher.publish(Twist())
        if self.local_display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    cli_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node: Optional[LaneDriveNode] = None
    try:
        node = LaneDriveNode(cli_args)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
