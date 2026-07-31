#!/usr/bin/env python3
"""ROS 2 node: nav_msgs/Path -> lane control candidate topics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Optional, Sequence

from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String

from visualizer import LOW_LATENCY_QOS, DrivingVisualizer


PARAMETER_DEFAULTS = {
    "path_topic": "/lane/path",
    "path_status_topic": "/lane/path/status",
    "steer_angle_topic": "/control/candidate/lane/steer_angle",
    "candidate_valid_topic": "/control/candidate/lane/valid",
    "throttle_feedback_topic": "/auto_throttle",
    "status_topic": "/lane/control/status",
    "wheelbase_m": 0.545,
    "ld_throttle_min": 0.4,
    "ld_throttle_max": 0.8,
    "dynamic_lookahead_enabled": True,
    "lookahead_min_m": 1.10,
    "lookahead_max_m": 2.00,
    "lookahead_m": 1.50,
    "max_steer_deg": 25.0,
    "steering_gain": 1.80,
    "steering_ema_alpha": 0.30,
    "steering_deadband_deg": 0.8,
    "max_steering_change_deg": 5.0,
    # "simple": forward-points-only closest-to-Ld search (yolotl_ros2
    # main5.py's actual default). "arc_length": accumulate arc length from
    # the nearest point until Ld is reached (main5's alternate, disabled by
    # default there too).
    "lookahead_search_mode": "simple",
    # main5.py-style local cv2 window (segmentation + BEV control + lines/
    # path), merged in-process from what used to be the separate
    # path_visualizer node. Set to false for headless runs.
    "local_display": True,
    "segmentation_topic": "/lane/detection/segmentation/compressed",
    "bev_topic": "/lane/detection/bev/compressed",
    "debug_topic": "/lane/path/debug/compressed",
    "instances_topic": "/lane/detection/instances",
    "window_name": (
        "drive visualizer: segmentation | BEV control | lines + path"
    ),
    "window_x": 20,
    "window_y": 60,
    "display_scale": 0.90,
    "box_ema_alpha": 0.15,
    "type_switch_frames": 5,
    "track_max_missed_frames": 12,
    "track_match_distance_px": 140.0,
    "confidence_full_hits": 5,
    "yolo_confidence_aggregation": "mean",
    "lookahead_target_hold_sec": 0.45,
    "reference_path_hold_sec": 0.45,
}


@dataclass
class SteeringCommand:
    """Result of one Pure Pursuit control step."""

    path_valid: bool
    reason: str
    steering_deg: float
    lookahead_m: float
    target_distance_m: float
    target_index: int


class PurePursuitController:
    """Pure Pursuit steering math, independent of ROS.

    Extracted out of ``PurePursuitNode`` so the same steering logic can be
    called directly (no topic round trip) from an integrated node such as
    ``drive_main.LaneDriveNode``, while ``PurePursuitNode`` keeps working
    unchanged as a thin ROS wrapper around it.
    """

    def __init__(
        self,
        wheelbase_m: float,
        ld_throttle_min: float,
        ld_throttle_max: float,
        lookahead_min_m: float,
        lookahead_max_m: float,
        fixed_lookahead_m: float,
        dynamic_lookahead_enabled: bool,
        max_steer_deg: float,
        steering_gain: float,
        steering_ema_alpha: float,
        steering_deadband_deg: float,
        max_steering_change_deg: float,
        lookahead_search_mode: str = "simple",
    ) -> None:
        self.wheelbase_m = float(wheelbase_m)
        self.ld_throttle_min = float(ld_throttle_min)
        self.ld_throttle_max = float(ld_throttle_max)
        if self.ld_throttle_max < self.ld_throttle_min:
            self.ld_throttle_min, self.ld_throttle_max = (
                self.ld_throttle_max,
                self.ld_throttle_min,
            )
        self.current_throttle = self.ld_throttle_min
        self.lookahead_min_m = float(lookahead_min_m)
        self.lookahead_max_m = float(lookahead_max_m)
        self.dynamic_lookahead_enabled = bool(
            dynamic_lookahead_enabled
        )
        if not self.dynamic_lookahead_enabled:
            if float(fixed_lookahead_m) <= 0.0:
                raise ValueError(
                    "lookahead_m must be positive when dynamic "
                    "lookahead is disabled"
                )
            self.lookahead_min_m = float(fixed_lookahead_m)
            self.lookahead_max_m = float(fixed_lookahead_m)
        self.max_steer_deg = float(max_steer_deg)
        self.steering_gain = float(steering_gain)
        self.steering_ema_alpha = float(steering_ema_alpha)
        self.steering_deadband_deg = float(steering_deadband_deg)
        self.max_steering_change_deg = float(max_steering_change_deg)
        self.lookahead_search_mode = str(lookahead_search_mode)
        self._validate_parameters()

        self.last_steering_deg = 0.0
        self.path_fallback = False

    def _validate_parameters(self) -> None:
        if self.wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be positive")
        if (
            self.lookahead_min_m <= 0.0
            or self.lookahead_max_m < self.lookahead_min_m
        ):
            raise ValueError(
                "lookahead must satisfy 0 < min <= max"
            )
        if self.ld_throttle_max <= self.ld_throttle_min:
            raise ValueError(
                "ld throttle range must satisfy min < max"
            )
        if self.max_steer_deg <= 0.0:
            raise ValueError("max_steer_deg must be positive")
        if self.steering_gain <= 0.0:
            raise ValueError("steering_gain must be positive")
        if not 0.0 <= self.steering_ema_alpha <= 1.0:
            raise ValueError(
                "steering_ema_alpha must be between 0 and 1"
            )
        if self.steering_deadband_deg < 0.0:
            raise ValueError("steering_deadband_deg cannot be negative")
        if self.lookahead_search_mode not in ("simple", "arc_length"):
            raise ValueError(
                'lookahead_search_mode must be "simple" or "arc_length"'
            )
        if self.max_steering_change_deg <= 0.0:
            raise ValueError(
                "max_steering_change_deg must be positive"
            )

    def set_path_fallback(self, fallback: bool) -> None:
        self.path_fallback = bool(fallback)

    def set_current_throttle(self, throttle: float) -> None:
        if not math.isfinite(throttle):
            return
        self.current_throttle = float(
            np.clip(
                throttle,
                self.ld_throttle_min,
                self.ld_throttle_max,
            )
        )

    def compute(self, points: np.ndarray) -> SteeringCommand:
        if (
            points is None
            or len(points) == 0
            or points.ndim != 2
            or not np.all(np.isfinite(points))
        ):
            return self.stop("empty_or_invalid_path")

        lookahead_m = self._dynamic_lookahead()
        distances = np.linalg.norm(points, axis=1)
        target_index = self._find_lookahead_index(
            points, distances, lookahead_m
        )
        forward, left = points[target_index]
        target_distance = max(float(distances[target_index]), 1e-3)
        heading_error = math.atan2(
            float(left),
            max(float(forward), 1e-3),
        )
        raw_steering_deg = math.degrees(
            math.atan2(
                2.0 * self.wheelbase_m * math.sin(heading_error),
                target_distance,
            )
        )
        raw_steering_deg = float(
            np.clip(
                raw_steering_deg * self.steering_gain,
                -self.max_steer_deg,
                self.max_steer_deg,
            )
        )
        if abs(raw_steering_deg) < self.steering_deadband_deg:
            raw_steering_deg = 0.0

        filtered = (
            self.steering_ema_alpha * raw_steering_deg
            + (1.0 - self.steering_ema_alpha) * self.last_steering_deg
        )
        steering_step = float(
            np.clip(
                filtered - self.last_steering_deg,
                -self.max_steering_change_deg,
                self.max_steering_change_deg,
            )
        )
        self.last_steering_deg = float(
            np.clip(
                self.last_steering_deg + steering_step,
                -self.max_steer_deg,
                self.max_steer_deg,
            )
        )
        lookahead_m = self._dynamic_lookahead()
        visual_target_index = self._find_lookahead_index(
            points, distances, lookahead_m
        )
        visual_target_distance = max(
            float(distances[visual_target_index]),
            1e-3,
        )
        return SteeringCommand(
            path_valid=True,
            reason="ok",
            steering_deg=self.last_steering_deg,
            lookahead_m=lookahead_m,
            target_distance_m=visual_target_distance,
            target_index=visual_target_index,
        )

    def stop(self, reason: str) -> SteeringCommand:
        return SteeringCommand(
            path_valid=False,
            reason=reason,
            steering_deg=self.last_steering_deg,
            lookahead_m=self._dynamic_lookahead(),
            target_distance_m=0.0,
            target_index=-1,
        )

    def _find_lookahead_index(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        if self.lookahead_search_mode == "arc_length":
            return self._arc_length_lookahead_index(
                points, distances, lookahead_m
            )
        return self._simple_lookahead_index(points, distances, lookahead_m)

    @staticmethod
    def _simple_lookahead_index(
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        """Original local closest-to-Ld point search."""

        del points
        return int(np.argmin(np.abs(distances - lookahead_m)))

    @staticmethod
    def _arc_length_lookahead_index(
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        """Accumulate arc length from the nearest point until Ld is
        reached, ported from yolotl_ros2 main5.py's
        ``use_arc_length_lookahead`` branch.

        dolbatS orders path points far -> near (ascending BEV y); main5's
        own point array is near -> far, so walk the reversed order here to
        match main5's accumulation direction. Snaps to the nearest array
        index instead of main5's sub-segment linear interpolation, since
        ``SteeringCommand.target_index`` is an index into ``points``.
        """

        order = np.arange(len(points) - 1, -1, -1)
        rev_points = points[order]
        rev_distances = distances[order]

        nearest_pos = int(np.argmin(rev_distances))
        accum = float(rev_distances[nearest_pos])
        if accum >= lookahead_m:
            return int(order[nearest_pos])

        for pos in range(nearest_pos, len(rev_points) - 1):
            seg_len = float(
                np.linalg.norm(rev_points[pos + 1] - rev_points[pos])
            )
            if seg_len <= 1e-6:
                continue
            if accum + seg_len >= lookahead_m:
                return int(order[pos + 1])
            accum += seg_len

        return int(order[-1])

    def _dynamic_lookahead(self) -> float:
        if self.lookahead_min_m == self.lookahead_max_m:
            return self.lookahead_min_m
        steering_ratio = float(
            np.clip(
                abs(self.last_steering_deg) / self.max_steer_deg,
                0.0,
                1.0,
            )
        )
        lookahead = self.lookahead_max_m - (
            self.lookahead_max_m - self.lookahead_min_m
        ) * steering_ratio
        return float(
            np.clip(
                lookahead,
                self.lookahead_min_m,
                self.lookahead_max_m,
            )
        )


class PurePursuitNode(Node):
    """Publish lane steering candidates from a metric path."""

    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        self.controller = PurePursuitController(
            wheelbase_m=float(parameter("wheelbase_m")),
            ld_throttle_min=float(parameter("ld_throttle_min")),
            ld_throttle_max=float(parameter("ld_throttle_max")),
            lookahead_min_m=float(parameter("lookahead_min_m")),
            lookahead_max_m=float(parameter("lookahead_max_m")),
            fixed_lookahead_m=float(parameter("lookahead_m")),
            dynamic_lookahead_enabled=bool(
                parameter("dynamic_lookahead_enabled")
            ),
            max_steer_deg=float(parameter("max_steer_deg")),
            steering_gain=float(parameter("steering_gain")),
            steering_ema_alpha=float(
                parameter("steering_ema_alpha")
            ),
            steering_deadband_deg=float(
                parameter("steering_deadband_deg")
            ),
            max_steering_change_deg=float(
                parameter("max_steering_change_deg")
            ),
            lookahead_search_mode=str(
                parameter("lookahead_search_mode")
            ),
        )

        path_topic = str(parameter("path_topic"))
        path_status_topic = str(parameter("path_status_topic"))
        steer_angle_topic = str(parameter("steer_angle_topic"))
        throttle_feedback_topic = str(
            parameter("throttle_feedback_topic")
        )
        candidate_valid_topic = str(parameter("candidate_valid_topic"))
        status_topic = str(parameter("status_topic"))
        self.steer_publisher = self.create_publisher(
            Float32, steer_angle_topic, 10
        )
        self.candidate_valid_publisher = self.create_publisher(
            Bool, candidate_valid_topic, 10
        )
        self.status_publisher = self.create_publisher(
            String,
            status_topic,
            10,
        )
        self.path_subscription = self.create_subscription(
            Path,
            path_topic,
            self.on_path,
            10,
        )
        self.path_status_subscription = self.create_subscription(
            String,
            path_status_topic,
            self.on_path_status,
            10,
        )
        self.throttle_feedback_subscription = self.create_subscription(
            Float32,
            throttle_feedback_topic,
            self.on_throttle_feedback,
            10,
        )

        self.visualizer: Optional[DrivingVisualizer] = None
        if bool(parameter("local_display")):
            self.visualizer = DrivingVisualizer(
                window_name=str(parameter("window_name")),
                window_x=int(parameter("window_x")),
                window_y=int(parameter("window_y")),
                display_scale=float(parameter("display_scale")),
                box_ema_alpha=float(parameter("box_ema_alpha")),
                type_switch_frames=int(
                    parameter("type_switch_frames")
                ),
                track_max_missed_frames=int(
                    parameter("track_max_missed_frames")
                ),
                track_match_distance_px=float(
                    parameter("track_match_distance_px")
                ),
                confidence_full_hits=int(
                    parameter("confidence_full_hits")
                ),
                yolo_confidence_aggregation=str(
                    parameter("yolo_confidence_aggregation")
                ),
                lookahead_target_hold_sec=float(
                    parameter("lookahead_target_hold_sec")
                ),
                reference_path_hold_sec=float(
                    parameter("reference_path_hold_sec")
                ),
                logger=self.get_logger(),
            )
            self.segmentation_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("segmentation_topic")),
                self.visualizer.on_segmentation_image,
                LOW_LATENCY_QOS,
            )
            self.bev_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("bev_topic")),
                self.visualizer.on_bev_image,
                LOW_LATENCY_QOS,
            )
            self.debug_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("debug_topic")),
                self.visualizer.on_debug_image,
                LOW_LATENCY_QOS,
            )
            self.instances_subscription = self.create_subscription(
                String,
                str(parameter("instances_topic")),
                self.visualizer.on_yolo_instances,
                10,
            )

        self.get_logger().info(
            f"{path_topic} -> lane candidates: {steer_angle_topic}, "
            f"{candidate_valid_topic}; throttle feedback: "
            f"{throttle_feedback_topic}"
        )

    def on_throttle_feedback(self, message: Float32) -> None:
        self.controller.set_current_throttle(float(message.data))

    def on_path_status(self, message: String) -> None:
        if self.visualizer is not None:
            self.visualizer.on_path_status(message)
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.controller.set_path_fallback(status.get("fallback", False))

    def on_path(self, message: Path) -> None:
        points = np.asarray(
            [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in message.poses
            ],
            dtype=np.float64,
        )
        command = self.controller.compute(points)
        self._publish_command(command)

    def _publish_command(self, command: SteeringCommand) -> None:
        steering_deg = command.steering_deg  # path_valid=False여도 last_steering_deg 유지
        self.steer_publisher.publish(Float32(data=float(steering_deg)))
        self.candidate_valid_publisher.publish(Bool(data=command.path_valid))
        status = {
            "path_valid": command.path_valid,
            "reason": command.reason,
            "fallback": self.controller.path_fallback,
            "steering_deg": round(command.steering_deg, 2),
            "dynamic_lookahead_enabled": (
                self.controller.dynamic_lookahead_enabled
            ),
            "lookahead_m": round(command.lookahead_m, 3),
            "lookahead_target_m": round(
                command.target_distance_m, 3
            ),
            "lookahead_target_index": int(command.target_index),
            "current_final_throttle": round(
                self.controller.current_throttle, 3
            ),
            "candidate_steer_deg": round(steering_deg, 2),
            "candidate_valid": command.path_valid,
        }
        self.status_publisher.publish(
            String(data=json.dumps(status, ensure_ascii=False))
        )
        if self.visualizer is not None:
            self.visualizer.on_control_status(status)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.steer_publisher.publish(Float32(data=0.0))
            self.candidate_valid_publisher.publish(Bool(data=False))
        if self.visualizer is not None:
            self.visualizer.destroy()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PurePursuitNode] = None
    try:
        node = PurePursuitNode()
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
