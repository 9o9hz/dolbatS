#!/usr/bin/env python3
"""Publish an obstacle steering candidate from side ultrasonic sensors."""

import json
import math
from enum import Enum, auto
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String


class AvoidanceState(Enum):
    LANE_FOLLOW = auto()
    APPROACH = auto()
    TURN = auto()
    REARM = auto()
    FAULT = auto()


class YoloObstacleTurn(Node):
    """Gate YOLO and publish obstacle candidates for mission_manager."""

    def __init__(self) -> None:
        super().__init__("yolo_obstacle_turn")

        defaults = {
            "left_distance_topic": "/ultrasonic/left_distance",
            "right_distance_topic": "/ultrasonic/right_distance",
            "yolo_enable_topic": "/detect/obstacle/enable",
            "avoidance_active_topic": (
                "/detect/obstacle/avoidance_active"
            ),
            "candidate_steer_topic": (
                "/control/candidate/obstacle/steer_angle"
            ),
            "candidate_valid_topic": (
                "/control/candidate/obstacle/valid"
            ),
            "status_topic": "/detect/avoidance/status",
            "detect_threshold_cm": 40.0,
            "clear_threshold_cm": 45.0,
            "consecutive_frames": 3,
            "full_steer_angle_deg": 20.0,
            "trend_epsilon_cm": 0.5,
            "minimum_drop_cm": 2.0,
            "rise_from_minimum_cm": 3.0,
            "trend_consecutive_frames": 3,
            "approach_timeout_sec": 8.0,
            "turn_timeout_sec": 8.0,
            "publish_rate_hz": 30.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        parameter = lambda name: self.get_parameter(name).value

        self.detect_threshold_cm = float(parameter("detect_threshold_cm"))
        self.clear_threshold_cm = float(parameter("clear_threshold_cm"))
        self.consecutive_frames = max(
            1, int(parameter("consecutive_frames"))
        )
        self.full_steer_angle_deg = abs(
            float(parameter("full_steer_angle_deg"))
        )
        self.trend_epsilon_cm = max(
            0.0, float(parameter("trend_epsilon_cm"))
        )
        self.minimum_drop_cm = max(
            0.0, float(parameter("minimum_drop_cm"))
        )
        self.rise_from_minimum_cm = max(
            0.0, float(parameter("rise_from_minimum_cm"))
        )
        self.trend_consecutive_frames = max(
            1, int(parameter("trend_consecutive_frames"))
        )
        self.approach_timeout_sec = max(
            0.0, float(parameter("approach_timeout_sec"))
        )
        self.turn_timeout_sec = max(
            0.0, float(parameter("turn_timeout_sec"))
        )
        publish_rate_hz = max(
            1.0, float(parameter("publish_rate_hz"))
        )

        if self.detect_threshold_cm <= 0.0:
            raise ValueError("detect_threshold_cm must be positive")
        if self.clear_threshold_cm < self.detect_threshold_cm:
            raise ValueError(
                "clear_threshold_cm must be >= detect_threshold_cm"
            )
        if self.full_steer_angle_deg <= 0.0:
            raise ValueError("full_steer_angle_deg must be positive")

        self.state = AvoidanceState.LANE_FOLLOW
        self.left_distance: Optional[float] = None
        self.latched_side: Optional[str] = None
        self.pending_side: Optional[str] = None
        self.side_frames = 0
        self.clear_frames = 0
        self.rearm_frames = 0
        self.state_started = self.get_clock().now()
        self.turn_entry_distance: Optional[float] = None
        self.last_turn_distance: Optional[float] = None
        self.minimum_turn_distance: Optional[float] = None
        self.decrease_frames = 0
        self.increase_frames = 0
        self.saw_decrease = False

        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.yolo_enable_pub = self.create_publisher(
            Bool, str(parameter("yolo_enable_topic")), latched_qos
        )
        self.avoidance_active_pub = self.create_publisher(
            Bool, str(parameter("avoidance_active_topic")), 10
        )
        self.candidate_steer_pub = self.create_publisher(
            Float32, str(parameter("candidate_steer_topic")), 10
        )
        self.candidate_valid_pub = self.create_publisher(
            Bool, str(parameter("candidate_valid_topic")), 10
        )
        self.status_pub = self.create_publisher(
            String, str(parameter("status_topic")), 10
        )
        self.left_sub = self.create_subscription(
            Float32,
            str(parameter("left_distance_topic")),
            self.on_left_distance,
            10,
        )
        self.right_sub = self.create_subscription(
            Float32,
            str(parameter("right_distance_topic")),
            self.on_right_distance,
            10,
        )
        self.publish_timer = self.create_timer(
            1.0 / publish_rate_hz, self.publish_candidate
        )
        self.yolo_enable_pub.publish(Bool(data=False))
        self.publish_candidate()
        self.publish_status("ready")
        self.get_logger().info(
            "Obstacle candidate ready: ultrasonic gates YOLO; "
            f"detect <= {self.detect_threshold_cm:.1f} cm, "
            f"clear > {self.clear_threshold_cm:.1f} cm"
        )

    @staticmethod
    def valid_distance(value: Optional[float]) -> bool:
        return (
            value is not None
            and math.isfinite(value)
            and value >= 0.0
        )

    def on_left_distance(self, msg: Float32) -> None:
        self.left_distance = float(msg.data)

    def on_right_distance(self, msg: Float32) -> None:
        # serial_bridge publishes the left sample immediately before right.
        self.process_sensor_frame(self.left_distance, float(msg.data))

    def process_sensor_frame(
        self, left: Optional[float], right: Optional[float]
    ) -> None:
        if self.state == AvoidanceState.LANE_FOLLOW:
            self.detect_obstacle_side(left, right)
            return

        if self.latched_side is None:
            return
        watched = left if self.latched_side == "left" else right

        if self.state == AvoidanceState.APPROACH:
            cleared = (
                self.valid_distance(watched)
                and watched > self.clear_threshold_cm
            )
            self.clear_frames = self.clear_frames + 1 if cleared else 0
            if self.clear_frames >= self.consecutive_frames:
                self.start_turn(float(watched))
            return

        if self.state == AvoidanceState.TURN:
            if self.valid_distance(watched):
                self.update_turn_trend(float(watched))
            return

        if self.state == AvoidanceState.REARM:
            left_clear = (
                self.valid_distance(left)
                and left > self.clear_threshold_cm
            )
            right_clear = (
                self.valid_distance(right)
                and right > self.clear_threshold_cm
            )
            self.rearm_frames = (
                self.rearm_frames + 1
                if left_clear and right_clear
                else 0
            )
            if self.rearm_frames >= self.consecutive_frames:
                self.reset_for_next_obstacle()

    def detect_obstacle_side(
        self, left: Optional[float], right: Optional[float]
    ) -> None:
        left_close = (
            self.valid_distance(left)
            and left <= self.detect_threshold_cm
        )
        right_close = (
            self.valid_distance(right)
            and right <= self.detect_threshold_cm
        )
        side = None
        if left_close and right_close:
            side = "left" if float(left) <= float(right) else "right"
        elif left_close:
            side = "left"
        elif right_close:
            side = "right"

        if side is None:
            self.pending_side = None
            self.side_frames = 0
            return
        if side != self.pending_side:
            self.pending_side = side
            self.side_frames = 1
        else:
            self.side_frames += 1
        if self.side_frames < self.consecutive_frames:
            return

        self.latched_side = side
        self.state = AvoidanceState.APPROACH
        self.state_started = self.get_clock().now()
        self.clear_frames = 0
        self.yolo_enable_pub.publish(Bool(data=True))
        self.publish_candidate()
        self.publish_status("ultrasonic_detected")
        self.get_logger().warning(
            f"{side} ultrasonic obstacle detected: "
            "YOLO on, obstacle candidate active"
        )

    def start_turn(self, distance: float) -> None:
        self.state = AvoidanceState.TURN
        self.state_started = self.get_clock().now()
        self.turn_entry_distance = distance
        self.last_turn_distance = distance
        self.minimum_turn_distance = distance
        self.decrease_frames = 0
        self.increase_frames = 0
        self.saw_decrease = False
        self.publish_candidate()
        self.publish_status("sensor_cleared_start_turn")
        self.get_logger().warning(
            f"{self.latched_side} obstacle no longer detected: "
            f"publishing full-steering {self.latched_side} candidate"
        )

    def update_turn_trend(self, distance: float) -> None:
        previous = self.last_turn_distance
        if previous is None or self.minimum_turn_distance is None:
            self.last_turn_distance = distance
            self.minimum_turn_distance = distance
            return

        if distance < self.minimum_turn_distance:
            self.minimum_turn_distance = distance

        if distance <= previous - self.trend_epsilon_cm:
            self.decrease_frames += 1
            self.increase_frames = 0
        elif distance >= previous + self.trend_epsilon_cm:
            self.increase_frames += 1
            self.decrease_frames = 0
        else:
            self.decrease_frames = 0
            self.increase_frames = 0

        drop = float(self.turn_entry_distance) - self.minimum_turn_distance
        if (
            self.decrease_frames >= self.trend_consecutive_frames
            and drop >= self.minimum_drop_cm
        ):
            self.saw_decrease = True
        rise = distance - self.minimum_turn_distance
        self.last_turn_distance = distance

        if (
            self.saw_decrease
            and self.increase_frames >= self.trend_consecutive_frames
            and rise >= self.rise_from_minimum_cm
        ):
            self.finish_turn(distance)

    def finish_turn(self, distance: float) -> None:
        self.state = AvoidanceState.REARM
        self.state_started = self.get_clock().now()
        self.rearm_frames = 0
        self.yolo_enable_pub.publish(Bool(data=False))
        self.publish_candidate()
        self.publish_status("distance_increased_lane_resume")
        self.get_logger().warning(
            f"{self.latched_side} distance increased to {distance:.1f} cm "
            f"after minimum {self.minimum_turn_distance:.1f} cm: "
            "obstacle candidate released"
        )

    def reset_for_next_obstacle(self) -> None:
        self.state = AvoidanceState.LANE_FOLLOW
        self.state_started = self.get_clock().now()
        self.latched_side = None
        self.pending_side = None
        self.side_frames = 0
        self.clear_frames = 0
        self.rearm_frames = 0
        self.publish_candidate()
        self.publish_status("rearmed")

    def publish_candidate(self) -> None:
        elapsed = (
            self.get_clock().now() - self.state_started
        ).nanoseconds / 1e9
        if (
            self.state == AvoidanceState.APPROACH
            and self.approach_timeout_sec > 0.0
            and elapsed >= self.approach_timeout_sec
        ):
            self.enter_fault("approach_timeout")
            return
        if (
            self.state == AvoidanceState.TURN
            and self.turn_timeout_sec > 0.0
            and elapsed >= self.turn_timeout_sec
        ):
            self.enter_fault("turn_timeout")
            return

        active = self.state in (
            AvoidanceState.APPROACH,
            AvoidanceState.TURN,
            AvoidanceState.FAULT,
        )
        valid = self.state in (
            AvoidanceState.APPROACH,
            AvoidanceState.TURN,
        )
        steer = 0.0
        if self.state == AvoidanceState.TURN:
            steer = (
                self.full_steer_angle_deg
                if self.latched_side == "left"
                else -self.full_steer_angle_deg
            )
        self.candidate_steer_pub.publish(Float32(data=steer))
        self.candidate_valid_pub.publish(Bool(data=valid))
        self.avoidance_active_pub.publish(Bool(data=active))

    def enter_fault(self, reason: str) -> None:
        if self.state == AvoidanceState.FAULT:
            return
        self.state = AvoidanceState.FAULT
        self.state_started = self.get_clock().now()
        self.yolo_enable_pub.publish(Bool(data=False))
        self.publish_candidate()
        self.publish_status(reason)
        self.get_logger().error(
            f"Avoidance candidate invalidated: {reason}. "
            "mission_manager will stop the vehicle."
        )

    def publish_status(self, reason: str) -> None:
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "state": self.state.name.lower(),
                        "reason": reason,
                        "obstacle_side": self.latched_side,
                        "avoidance_active": self.state
                        in (
                            AvoidanceState.APPROACH,
                            AvoidanceState.TURN,
                            AvoidanceState.FAULT,
                        ),
                        "candidate_valid": self.state
                        in (
                            AvoidanceState.APPROACH,
                            AvoidanceState.TURN,
                        ),
                        "yolo_enabled": self.state
                        in (
                            AvoidanceState.APPROACH,
                            AvoidanceState.TURN,
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )

    def destroy_node(self) -> bool:
        if hasattr(self, "candidate_valid_pub"):
            self.state = AvoidanceState.FAULT
            self.publish_candidate()
            self.yolo_enable_pub.publish(Bool(data=False))
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleTurn()
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
