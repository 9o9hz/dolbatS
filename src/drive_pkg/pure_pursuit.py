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
from std_msgs.msg import String


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
    "steering_deadband_deg": 0.8,
    "max_steering_change_deg": 3.0,
    "path_timeout_sec": 0.5,
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
        steering_deadband_deg: float,
        max_steering_change_deg: float,
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
        self.steering_deadband_deg = float(steering_deadband_deg)
        self.max_steering_change_deg = float(max_steering_change_deg)
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
        visual_target_index = int(
            np.argmin(np.abs(distances - lookahead_m))
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
        self.get_logger().info(f"{path_topic} -> {cmd_vel_topic}")

    def on_path_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.controller.set_path_fallback(status.get("fallback", False))

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
        command = self.controller.compute(points)
        self._publish_command(command)

    def _publish_command(self, command: SteeringCommand) -> None:
        twist = (
            self.controller.make_twist(command)
            if command.path_valid
            else Twist()
        )
        self.cmd_publisher.publish(twist)
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "path_valid": command.path_valid,
                        "reason": command.reason,
                        "fallback": self.controller.path_fallback,
                        "steering_deg": round(
                            command.steering_deg,
                            2,
                        ),
                        "lookahead_m": round(command.lookahead_m, 3),
                        "lookahead_target_m": round(
                            command.target_distance_m,
                            3,
                        ),
                        "lookahead_target_index": int(
                            command.target_index
                        ),
                        "desired_speed_mps": round(
                            command.speed_mps,
                            3,
                        ),
                        "published_linear_x": round(
                            twist.linear.x,
                            3,
                        ),
                        "published_angular_z": round(
                            twist.angular.z,
                            3,
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
        self._publish_command(self.controller.stop("path_timeout"))
        self.get_logger().warning(
            f"No path for {elapsed:.2f}s: vehicle stopped"
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
