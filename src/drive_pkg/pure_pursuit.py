햣#!/usr/bin/env python3
"""ROS 2 node: nav_msgs/Path -> Ackermann-compatible cmd_vel."""

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
from std_msgs.msg import Bool, String


PARAMETER_DEFAULTS = {
    "path_topic": "/lane/path",
    "path_status_topic": "/lane/path/status",
    "steer_angle_topic": "/control/candidate/lane/steer_angle",
    "candidate_valid_topic": "/control/candidate/lane/valid",
    "throttle_feedback_topic": "/auto_throttle",
    "status_topic": "/lane/control/status",
    "enable_drive": False,
    "keyboard_enable_topic": "/drive/keyboard_enabled",
    "speed_mps": 0.18,
    "hold_speed_scale": 0.75,
    "turn_speed_min_scale": 0.25,
    "turn_speed_reduction": 0.70,
    "wheelbase_m": 0.545,
    "ld_throttle_min": 0.4,
    "ld_throttle_max": 0.8,
    "lookahead_min_m": 2.30,
    "lookahead_max_m": 2.80,
    "lookahead_m": -1.0,
    "max_steer_deg": 18.0,
    "steering_ema_alpha": 0.35,
    "max_steering_change_deg": 3.0,
    "path_timeout_sec": 0.5,
}


class PurePursuitNode(Node):
    """Pure Pursuit controller isolated behind path and cmd_vel topics."""

    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        self.enable_drive = bool(parameter("enable_drive"))
        self.keyboard_drive_enabled = False
        self.speed_mps = max(0.0, float(parameter("speed_mps")))
        self.hold_speed_scale = float(
            np.clip(float(parameter("hold_speed_scale")), 0.0, 1.0)
        )
        self.turn_speed_min_scale = float(
            np.clip(
                float(parameter("turn_speed_min_scale")),
                0.0,
                1.0,
            )
        )
        self.turn_speed_reduction = float(
            np.clip(
                float(parameter("turn_speed_reduction")),
                0.0,
                1.0,
            )
        )
        self.wheelbase_m = float(parameter("wheelbase_m"))
        self.lookahead_min_m = float(parameter("lookahead_min_m"))
        self.lookahead_max_m = float(parameter("lookahead_max_m"))
        fixed_lookahead = float(parameter("lookahead_m"))
        if fixed_lookahead > 0.0:
            self.lookahead_min_m = fixed_lookahead
            self.lookahead_max_m = fixed_lookahead
        self.max_steer_deg = float(parameter("max_steer_deg"))
        self.steering_ema_alpha = float(
            parameter("steering_ema_alpha")
        )
        self.steering_deadband_deg = float(
            parameter("steering_deadband_deg")
        )
        self.max_steering_change_deg = float(
            parameter("max_steering_change_deg")
        )
        self.path_timeout_sec = max(
            0.0, float(parameter("path_timeout_sec"))
        )
        self._validate_parameters()

        self.last_steering_deg = 0.0
        self.path_fallback = False
        self.last_path_time = None
        self.timed_out = False

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
        self.keyboard_subscription = self.create_subscription(
            Bool,
            str(parameter("keyboard_enable_topic")),
            self.on_keyboard_enable,
            10,
        )
        self.watchdog = self.create_timer(0.1, self.check_path_timeout)
        self.get_logger().info(
            f"{path_topic} -> {cmd_vel_topic}; "
            f"drive_enabled={self.enable_drive}"
        )
        if not self.enable_drive:
            self.get_logger().info(
                "Dry-run mode: controller status is calculated, "
                "but cmd_vel remains zero."
            )

    def on_keyboard_enable(self, message: Bool) -> None:
        previous = self.keyboard_drive_enabled
        self.keyboard_drive_enabled = bool(message.data)
        if previous and not self.keyboard_drive_enabled:
            self._publish_stop("keyboard_stop")

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
        lookahead_m = self._dynamic_lookahead(
            self.last_steering_deg
        )
        speed = self._target_speed(self.last_steering_deg)
        if self.path_fallback:
            speed *= self.hold_speed_scale

        command = (
            self._make_twist(
                speed,
                math.radians(self.last_steering_deg),
            )
            if self.enable_drive and self.keyboard_drive_enabled
            else Twist()
        )
        self.cmd_publisher.publish(command)
        self._publish_status(
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

    def _dynamic_lookahead(self) -> float:
        if self.lookahead_min_m == self.lookahead_max_m:
            return self.lookahead_min_m
        throttle_ratio = float(
            np.clip(
                (
                    self.current_throttle - self.ld_throttle_min
                )
                / (
                    self.ld_throttle_max
                    - self.ld_throttle_min
                    + 1e-6
                ),
                0.0,
                1.0,
            )
        )
        lookahead = self.lookahead_min_m + (
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

    def _make_twist(
        self,
        speed_mps: float,
        steering_rad: float,
    ) -> Twist:
        message = Twist()
        message.linear.x = float(speed_mps)
        if abs(speed_mps) > 1e-6:
            message.angular.z = float(
                speed_mps
                / self.wheelbase_m
                * math.tan(steering_rad)
            )
        return message

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
        self._publish_stop("path_timeout")
        self.get_logger().warning(
            f"No path for {elapsed:.2f}s: vehicle stopped"
        )

    def _publish_stop(self, reason: str) -> None:
        command = Twist()
        self.cmd_publisher.publish(command)
        self._publish_status(
            path_valid=False,
            reason=reason,
            desired_speed=0.0,
            command=command,
            lookahead_m=self._dynamic_lookahead(
                self.last_steering_deg
            ),
            target_distance=0.0,
        )

    def _publish_status(
        self,
        *,
        path_valid: bool,
        reason: str,
        desired_speed: float,
        command: Twist,
        lookahead_m: float,
        target_distance: float,
    ) -> None:
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "path_valid": path_valid,
                        "reason": reason,
                        "fallback": self.path_fallback,
                        "drive_enabled": (
                            self.enable_drive
                            and self.keyboard_drive_enabled
                        ),
                        "steering_deg": round(
                            self.last_steering_deg,
                            2,
                        ),
                        "lookahead_m": round(lookahead_m, 3),
                        "lookahead_target_m": round(
                            target_distance,
                            3,
                        ),
                        "desired_speed_mps": round(
                            desired_speed,
                            3,
                        ),
                        "published_linear_x": round(
                            command.linear.x,
                            3,
                        ),
                        "published_angular_z": round(
                            command.angular.z,
                            3,
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )

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
