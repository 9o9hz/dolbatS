#!/usr/bin/env python3
"""ROS 2 node: nav_msgs/Path -> Ackermann-compatible cmd_vel."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Optional, Sequence

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from visualizer import LOW_LATENCY_QOS, DrivingVisualizer


PARAMETER_DEFAULTS = {
    "path_topic": "/lane/path",
    "path_status_topic": "/lane/path/status",
    "cmd_vel_topic": "/cmd_vel",
    "status_topic": "/lane/control/status",
    "speed_mps": 0.18,
    "hold_speed_scale": 0.75,
    "turn_speed_min_scale": 0.25,
    "turn_speed_reduction": 0.70,
    "wheelbase_m": 0.545,
    "lookahead_min_m": 1.10,
    "lookahead_max_m": 2.50,
    "lookahead_m": -1.0,
    "max_steer_deg": 18.0,
    "steering_ema_alpha": 0.35,
    "max_steering_change_deg": 3.0,
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
    speed_mps: float
    lookahead_m: float
    target_distance_m: float
    target_index: int


class PurePursuitController:
    """Pure Pursuit steering/speed math, independent of ROS.

    Extracted out of ``PurePursuitNode`` so the same steering logic can be
    called directly (no topic round trip) from an integrated node such as
    ``drive_main.LaneDriveNode``, while ``PurePursuitNode`` keeps working
    unchanged as a thin ROS wrapper around it.
    """

    def __init__(
        self,
        speed_mps: float,
        hold_speed_scale: float,
        turn_speed_min_scale: float,
        turn_speed_reduction: float,
        wheelbase_m: float,
        lookahead_min_m: float,
        lookahead_max_m: float,
        fixed_lookahead_m: float,
        max_steer_deg: float,
        steering_ema_alpha: float,
        max_steering_change_deg: float,
        lookahead_search_mode: str = "simple",
    ) -> None:
        self.speed_mps = max(0.0, float(speed_mps))
        self.hold_speed_scale = float(
            np.clip(float(hold_speed_scale), 0.0, 1.0)
        )
        self.turn_speed_min_scale = float(
            np.clip(float(turn_speed_min_scale), 0.0, 1.0)
        )
        self.turn_speed_reduction = float(
            np.clip(float(turn_speed_reduction), 0.0, 1.0)
        )
        self.wheelbase_m = float(wheelbase_m)
        self.lookahead_min_m = float(lookahead_min_m)
        self.lookahead_max_m = float(lookahead_max_m)
        if float(fixed_lookahead_m) > 0.0:
            self.lookahead_min_m = float(fixed_lookahead_m)
            self.lookahead_max_m = float(fixed_lookahead_m)
        self.max_steer_deg = float(max_steer_deg)
        self.steering_ema_alpha = float(steering_ema_alpha)
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
        if self.max_steer_deg <= 0.0:
            raise ValueError("max_steer_deg must be positive")
        if not 0.0 <= self.steering_ema_alpha <= 1.0:
            raise ValueError(
                "steering_ema_alpha must be between 0 and 1"
            )
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

    def compute(self, points: np.ndarray) -> SteeringCommand:
        if (
            points is None
            or len(points) == 0
            or points.ndim != 2
            or not np.all(np.isfinite(points))
        ):
            return self.stop("empty_or_invalid_path")

        lookahead_m = self._dynamic_lookahead(self.last_steering_deg)
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
                raw_steering_deg,
                -self.max_steer_deg,
                self.max_steer_deg,
            )
        )

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
        lookahead_m = self._dynamic_lookahead(self.last_steering_deg)
        visual_target_index = self._find_lookahead_index(
            points, distances, lookahead_m
        )
        visual_target_distance = max(
            float(distances[visual_target_index]),
            1e-3,
        )
        speed = self._target_speed(self.last_steering_deg)
        if self.path_fallback:
            speed *= self.hold_speed_scale

        return SteeringCommand(
            path_valid=True,
            reason="ok",
            steering_deg=self.last_steering_deg,
            speed_mps=speed,
            lookahead_m=lookahead_m,
            target_distance_m=visual_target_distance,
            target_index=visual_target_index,
        )

    def stop(self, reason: str) -> SteeringCommand:
        return SteeringCommand(
            path_valid=False,
            reason=reason,
            steering_deg=self.last_steering_deg,
            speed_mps=0.0,
            lookahead_m=self._dynamic_lookahead(self.last_steering_deg),
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
        """Closest-to-Ld search among forward points only, ported from
        yolotl_ros2 main5.py's default (non-arc-length) lookahead search.
        Falls back to the closest point overall if none are forward (e.g.
        the whole path happens to be behind the vehicle)."""

        forward_mask = points[:, 0] >= 0.0
        if not np.any(forward_mask):
            return int(np.argmin(np.abs(distances - lookahead_m)))
        forward_indices = np.flatnonzero(forward_mask)
        best = forward_indices[
            np.argmin(
                np.abs(distances[forward_indices] - lookahead_m)
            )
        ]
        return int(best)

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

    def _dynamic_lookahead(self, steering_deg: float) -> float:
        ratio = float(
            np.clip(
                abs(steering_deg) / self.max_steer_deg,
                0.0,
                1.0,
            )
        )
        return self.lookahead_max_m - (
            self.lookahead_max_m - self.lookahead_min_m
        ) * ratio

    def _target_speed(self, steering_deg: float) -> float:
        turn_ratio = min(
            1.0,
            abs(steering_deg) / self.max_steer_deg,
        )
        scale = max(
            self.turn_speed_min_scale,
            1.0 - self.turn_speed_reduction * turn_ratio,
        )
        return self.speed_mps * scale

    def make_twist(self, command: SteeringCommand) -> Twist:
        message = Twist()
        message.linear.x = float(command.speed_mps)
        if abs(command.speed_mps) > 1e-6:
            message.angular.z = float(
                command.speed_mps
                / self.wheelbase_m
                * math.tan(math.radians(command.steering_deg))
            )
        return message


class PurePursuitNode(Node):
    """Pure Pursuit controller isolated behind path and cmd_vel topics."""

    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

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
            steering_ema_alpha=float(
                parameter("steering_ema_alpha")
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
        cmd_vel_topic = str(parameter("cmd_vel_topic"))
        status_topic = str(parameter("status_topic"))
        self.cmd_publisher = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10,
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

        self.get_logger().info(f"{path_topic} -> {cmd_vel_topic}")

    # ------------------------------------------------------------------
    # Mission hook (structure only; no mission logic implemented yet).
    # ------------------------------------------------------------------
    def postprocess_command(
        self, command: SteeringCommand
    ) -> SteeringCommand:
        """Called after Pure Pursuit computes a steering command, before
        publishing. Future home for mission logic that overrides speed/
        steering (stop line, traffic light, parking). Currently a
        pass-through."""
        return command

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
        command = self.postprocess_command(command)
        self._publish_command(command)

    def _publish_command(self, command: SteeringCommand) -> None:
        twist = (
            self.controller.make_twist(command)
            if command.path_valid
            else Twist()
        )
        self.cmd_publisher.publish(twist)
        status = {
            "path_valid": command.path_valid,
            "reason": command.reason,
            "fallback": self.controller.path_fallback,
            "steering_deg": round(command.steering_deg, 2),
            "lookahead_m": round(command.lookahead_m, 3),
            "lookahead_target_m": round(
                command.target_distance_m, 3
            ),
            "lookahead_target_index": int(command.target_index),
            "desired_speed_mps": round(command.speed_mps, 3),
            "published_linear_x": round(twist.linear.x, 3),
            "published_angular_z": round(twist.angular.z, 3),
        }
        self.status_publisher.publish(
            String(data=json.dumps(status, ensure_ascii=False))
        )
        if self.visualizer is not None:
            self.visualizer.on_control_status(status)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.cmd_publisher.publish(Twist())
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
