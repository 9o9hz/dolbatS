#!/usr/bin/env python3
"""ROS 2 node: BEV lane mask -> local path topics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from lane_processing import (
    DEFAULT_BEV_PARAMS,
    LaneConfig,
    PathPlanResult,
    SegmentationLaneProcessor,
    load_bev_parameters,
)


PARAMETER_DEFAULTS = {
    "bev_params": str(DEFAULT_BEV_PARAMS),
    "mask_topic": "/lane/detection/mask/compressed",
    "path_topic": "/lane/path",
    "debug_topic": "/lane/path/debug/compressed",
    "status_topic": "/lane/path/status",
    "which_lane_topic": "/which/lane",
    "path_frame_id": "base_link",
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
    "debug_jpeg_quality": 80,
    "display": False,
    "display_scale": 1.0,
}


class PathPlanNode(Node):
    """Consumes the detector output and publishes a metric nav_msgs/Path."""

    def __init__(self) -> None:
        super().__init__("path_plan")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        bev_value = str(parameter("bev_params")).strip()
        bev_path = Path(bev_value) if bev_value else DEFAULT_BEV_PARAMS
        bev = load_bev_parameters(bev_path)
        config = LaneConfig(
            source_points=tuple(bev.source_points.reshape(-1)),
            destination_points=tuple(
                bev.destination_points.reshape(-1)
            ),
            warp_width=bev.width,
            warp_height=bev.height,
            pixels_per_meter=float(parameter("pixels_per_meter")),
            lane_width_m=float(parameter("lane_width_m")),
            bev_reference_forward_offset_m=float(
                parameter("bev_reference_forward_offset_m")
            ),
            min_component_area=int(
                parameter("min_component_area")
            ),
            center_sample_step=int(
                parameter("center_sample_step")
            ),
            same_line_threshold_px=float(
                parameter("same_line_threshold_px")
            ),
            min_group_span_px=float(
                parameter("min_group_span_px")
            ),
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
            solid_enter_frames=int(
                parameter("solid_enter_frames")
            ),
            solid_exit_frames=int(
                parameter("solid_exit_frames")
            ),
            path_top_y=int(parameter("path_top_y")),
            path_bottom_margin=int(
                parameter("path_bottom_margin")
            ),
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
            max_missing_frames=int(
                parameter("max_missing_frames")
            ),
        )
        self.processor = SegmentationLaneProcessor(None, config)
        self.path_frame_id = str(parameter("path_frame_id"))
        self.jpeg_quality = int(
            np.clip(int(parameter("debug_jpeg_quality")), 1, 100)
        )
        self.display = bool(parameter("display"))
        self.display_scale = max(
            0.1, float(parameter("display_scale"))
        )

        mask_topic = str(parameter("mask_topic"))
        path_topic = str(parameter("path_topic"))
        debug_topic = str(parameter("debug_topic"))
        status_topic = str(parameter("status_topic"))
        which_lane_topic = str(parameter("which_lane_topic"))
        self.path_publisher = self.create_publisher(
            PathMessage,
            path_topic,
            10,
        )
        self.debug_publisher = self.create_publisher(
            CompressedImage,
            debug_topic,
            qos_profile_sensor_data,
        )
        self.status_publisher = self.create_publisher(
            String,
            status_topic,
            10,
        )
        self.which_lane_publisher = self.create_publisher(
            String,
            which_lane_topic,
            10,
        )
        self.mask_subscription = self.create_subscription(
            CompressedImage,
            mask_topic,
            self.on_mask,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"{mask_topic} -> {path_topic}; frame={self.path_frame_id}, "
            f"bev={bev_path}"
        )

    def on_mask(self, message: CompressedImage) -> None:
        try:
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                raise ValueError("Could not decode lane mask")
            output = self.processor.plan_mask(mask)
        except Exception as exc:
            self.path_publisher.publish(self._path_message(None, message))
            self.get_logger().error(
                f"Path planning failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        self.which_lane_publisher.publish(
            String(data=output.which_lane or "unknown")
        )
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "path_valid": output.path_meters is not None,
                        "reason": output.reason,
                        "fallback": output.used_fallback,
                        "lane_groups": output.group_count,
                        "dashed_region_count": (
                            output.dashed_region_count
                        ),
                        "selection_mode": output.selection_mode,
                        "which_lane": output.which_lane,
                        "detected_lines": output.detected_lines,
                        "left_boundary_pixels": self._points_payload(
                            output.left_boundary_pixels
                        ),
                        "right_boundary_pixels": self._points_payload(
                            output.right_boundary_pixels
                        ),
                        "path_pixels": self._points_payload(
                            output.path_pixels
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )
        self.path_publisher.publish(
            self._path_message(output.path_meters, message)
        )
        debug = self._debug_image(mask, output)
        self._publish_debug(debug, message)

        if self.display:
            preview = cv2.resize(
                debug,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow("path_plan: local path", preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()

    @staticmethod
    def _points_payload(points: Optional[np.ndarray]) -> list[list[float]]:
        if points is None or len(points) == 0:
            return []
        return np.round(
            np.asarray(points, dtype=np.float64),
            1,
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

    @staticmethod
    def _debug_image(
        mask: np.ndarray,
        output: PathPlanResult,
    ) -> np.ndarray:
        debug = np.zeros((*mask.shape, 3), dtype=np.uint8)
        debug[mask > 0] = (0, 150, 0)
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        if output.path_pixels is not None:
            points = np.rint(output.path_pixels).astype(np.int32)
            if len(points) >= 2:
                cv2.polylines(
                    debug,
                    [points],
                    False,
                    (0, 0, 255),
                    4,
                )
        cv2.putText(
            debug,
            "GREEN: detected lines",
            (12, debug.shape[0] - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            "RED: reference path",
            (12, debug.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        lines = (
            f"path: {output.reason}",
            f"selection: {output.selection_mode}",
            (
                f"groups: {output.group_count} "
                f"fallback: {output.used_fallback}"
            ),
            f"which_lane: {output.which_lane or 'unknown'}",
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

    def _publish_debug(
        self,
        image: np.ndarray,
        source: CompressedImage,
    ) -> None:
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            return
        message = CompressedImage()
        message.header = source.header
        message.format = "jpeg"
        message.data = encoded.tobytes()
        self.debug_publisher.publish(message)

    def destroy_node(self) -> bool:
        if self.display:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PathPlanNode] = None
    try:
        node = PathPlanNode()
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
