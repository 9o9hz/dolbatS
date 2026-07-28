#!/usr/bin/env python3
"""ROS 2 node: nav_msgs/Path -> Ackermann-compatible cmd_vel."""

from __future__ import annotations

import json
import math
from typing import Optional, Sequence

from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


PARAMETER_DEFAULTS = {
    "path_topic": "/lane/path",
    "path_status_topic": "/lane/path/status",
    "cmd_vel_topic": "/cmd_vel",
    "status_topic": "/lane/control/status",
    "enable_drive": False,
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
    "steering_deadband_deg": 0.8,
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
        if self.max_steering_change_deg <= 0.0:
            raise ValueError(
                "max_steering_change_deg must be positive"
            )

    def on_path_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.path_fallback = bool(status.get("fallback", False))

    def on_path(self, message: Path) -> None:
        self.last_path_time = self.get_clock().now()
        self.timed_out = False
        points = np.asarray(
            [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in message.poses
            ],
            dtype=np.float64,
        )
        if (
            len(points) == 0
            or points.ndim != 2
            or not np.all(np.isfinite(points))
        ):
            self._publish_stop("empty_or_invalid_path")
            return

        lookahead_m = self._dynamic_lookahead(
            self.last_steering_deg
        )
        distances = np.linalg.norm(points, axis=1)
        target_index = int(
            np.argmin(np.abs(distances - lookahead_m))
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
        if abs(raw_steering_deg) < self.steering_deadband_deg:
            raw_steering_deg = 0.0

        filtered = (
            self.steering_ema_alpha * raw_steering_deg
            + (1.0 - self.steering_ema_alpha)
            * self.last_steering_deg
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
            if self.enable_drive
            else Twist()
        )
        self.cmd_publisher.publish(command)
        self._publish_status(
            path_valid=True,
            reason="ok",
            desired_speed=speed,
            command=command,
            lookahead_m=lookahead_m,
            target_distance=target_distance,
        )

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
                        "drive_enabled": self.enable_drive,
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
            self.cmd_publisher.publish(Twist())
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
