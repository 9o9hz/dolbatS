#!/usr/bin/env python3
"""Publish obstacle steering candidates from YOLO and four ultrasonics."""

import json
import math
from enum import Enum, auto
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class AvoidanceState(Enum):
    LANE_FOLLOW = auto()
    APPROACH = auto()
    TURN = auto()
    REARM = auto()
    FAULT = auto()


class YoloObstacleTurn(Node):
    """Generate an obstacle candidate for mission_manager.

    A YOLO detection plus a close rear sensor starts avoidance. Once the
    latched rear sensor clears, steer fully toward that side. End avoidance
    after the opposite-side front sensor decreases and then increases.

    candidate_valid is only True in TURN, when a full-steer avoidance
    command is actually being published. During APPROACH and REARM,
    avoidance is merely armed/watching, so candidate_valid stays False and
    mission_manager falls back to lane steering.
    """

    def __init__(self) -> None:
        super().__init__("yolo_obstacle_turn")

        defaults = {
            "yolo_detected_topic": "/detect/obstacle/detected",
            "left_front_distance_topic": (
                "/ultrasonic/left/front"
            ),
            "left_rear_distance_topic": (
                "/ultrasonic/left/rear"
            ),
            "right_front_distance_topic": (
                "/ultrasonic/right/front"
            ),
            "right_rear_distance_topic": (
                "/ultrasonic/right/rear"
            ),
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
            "rear_no_echo_is_clear": True,
            "yolo_timeout_sec": 0.5,
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
        self.rear_no_echo_is_clear = bool(
            parameter("rear_no_echo_is_clear")
        )
        self.yolo_timeout_sec = max(
            0.0, float(parameter("yolo_timeout_sec"))
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
        self.state_started = self.get_clock().now()
        self.yolo_detected = False
        self.last_yolo_time = None
        self.yolo_clear_frames = 0

        self.left_front_distance: Optional[float] = None
        self.left_rear_distance: Optional[float] = None
        self.right_front_distance: Optional[float] = None
        self.right_rear_distance: Optional[float] = None

        self.latched_side: Optional[str] = None
        self.trend_sensor_side: Optional[str] = None
        self.pending_side: Optional[str] = None
        self.side_frames = 0
        self.clear_frames = 0

        self.turn_entry_distance: Optional[float] = None
        self.last_turn_distance: Optional[float] = None
        self.minimum_turn_distance: Optional[float] = None
        self.decrease_frames = 0
        self.increase_frames = 0
        self.saw_decrease = False

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

        self.yolo_sub = self.create_subscription(
            Bool,
            str(parameter("yolo_detected_topic")),
            self.on_yolo_detected,
            10,
        )
        self.left_front_sub = self.create_subscription(
            Float32,
            str(parameter("left_front_distance_topic")),
            self.on_left_front_distance,
            10,
        )
        self.left_rear_sub = self.create_subscription(
            Float32,
            str(parameter("left_rear_distance_topic")),
            self.on_left_rear_distance,
            10,
        )
        self.right_front_sub = self.create_subscription(
            Float32,
            str(parameter("right_front_distance_topic")),
            self.on_right_front_distance,
            10,
        )
        self.right_rear_sub = self.create_subscription(
            Float32,
            str(parameter("right_rear_distance_topic")),
            self.on_right_rear_distance,
            10,
        )
        self.publish_timer = self.create_timer(
            1.0 / publish_rate_hz, self.publish_candidate
        )
        self.publish_candidate()
        self.publish_status("ready")
        self.get_logger().info(
            "Obstacle candidate ready: YOLO + rear ultrasonic trigger; "
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

    def rear_obstacle_cleared(self, value: Optional[float]) -> bool:
        if value is None or not math.isfinite(value):
            return False
        if value > self.clear_threshold_cm:
            return True
        return self.rear_no_echo_is_clear and value < 0.0

    def on_yolo_detected(self, msg: Bool) -> None:
        self.yolo_detected = bool(msg.data)
        self.last_yolo_time = self.get_clock().now()
        if self.state != AvoidanceState.REARM:
            return
        self.yolo_clear_frames = (
            0 if self.yolo_detected else self.yolo_clear_frames + 1
        )
        if self.yolo_clear_frames >= self.consecutive_frames:
            self.reset_for_next_obstacle()

    def on_left_front_distance(self, msg: Float32) -> None:
        self.left_front_distance = float(msg.data)

    def on_left_rear_distance(self, msg: Float32) -> None:
        self.left_rear_distance = float(msg.data)

    def on_right_front_distance(self, msg: Float32) -> None:
        self.right_front_distance = float(msg.data)

    def on_right_rear_distance(self, msg: Float32) -> None:
        self.right_rear_distance = float(msg.data)
        # Arduino and serial_bridge publish LF, LR, RF, RR in this order.
        # The RR callback therefore completes one four-sensor frame.
        self.process_sensor_frame(
            self.left_front_distance,
            self.left_rear_distance,
            self.right_front_distance,
            self.right_rear_distance,
        )

    def yolo_is_active(self) -> bool:
        if not self.yolo_detected or self.last_yolo_time is None:
            return False
        if self.yolo_timeout_sec <= 0.0:
            return True
        age = (
            self.get_clock().now() - self.last_yolo_time
        ).nanoseconds / 1e9
        return age <= self.yolo_timeout_sec

    def process_sensor_frame(
        self,
        left_front: Optional[float],
        left_rear: Optional[float],
        right_front: Optional[float],
        right_rear: Optional[float],
    ) -> None:
        if self.state == AvoidanceState.LANE_FOLLOW:
            self.detect_obstacle_side(left_rear, right_rear)
            return

        if self.latched_side is None:
            return

        if self.state == AvoidanceState.APPROACH:
            watched_rear = (
                left_rear
                if self.latched_side == "left"
                else right_rear
            )
            cleared = self.rear_obstacle_cleared(watched_rear)
            self.clear_frames = self.clear_frames + 1 if cleared else 0
            if self.clear_frames >= self.consecutive_frames:
                opposite_front = (
                    right_front
                    if self.latched_side == "left"
                    else left_front
                )
                self.start_turn(opposite_front)
            return

        if self.state == AvoidanceState.TURN:
            opposite_front = (
                right_front
                if self.latched_side == "left"
                else left_front
            )
            if self.valid_distance(opposite_front):
                self.update_turn_trend(float(opposite_front))

    def detect_obstacle_side(
        self,
        left_rear: Optional[float],
        right_rear: Optional[float],
    ) -> None:
        if not self.yolo_is_active():
            self.pending_side = None
            self.side_frames = 0
            return

        left_close = (
            self.valid_distance(left_rear)
            and left_rear <= self.detect_threshold_cm
        )
        right_close = (
            self.valid_distance(right_rear)
            and right_rear <= self.detect_threshold_cm
        )
        side = None
        if left_close and right_close:
            side = (
                "left"
                if float(left_rear) <= float(right_rear)
                else "right"
            )
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
        self.trend_sensor_side = (
            "right_front" if side == "left" else "left_front"
        )
        self.state = AvoidanceState.APPROACH
        self.state_started = self.get_clock().now()
        self.clear_frames = 0
        self.publish_candidate()
        self.publish_status("yolo_and_rear_ultrasonic_detected")
        self.get_logger().warning(
            f"YOLO + {side} rear ultrasonic obstacle: "
            "obstacle mode active, straight candidate"
        )

    def start_turn(self, opposite_front: Optional[float]) -> None:
        self.state = AvoidanceState.TURN
        self.state_started = self.get_clock().now()
        initial = (
            float(opposite_front)
            if self.valid_distance(opposite_front)
            else None
        )
        self.turn_entry_distance = initial
        self.last_turn_distance = initial
        self.minimum_turn_distance = initial
        self.decrease_frames = 0
        self.increase_frames = 0
        self.saw_decrease = False
        self.publish_candidate()
        self.publish_status("latched_rear_cleared_start_turn")
        self.get_logger().warning(
            f"{self.latched_side} rear obstacle cleared: "
            f"full-steering {self.latched_side}; "
            f"tracking {self.trend_sensor_side}"
        )

    def update_turn_trend(self, distance: float) -> None:
        if (
            self.turn_entry_distance is None
            or self.last_turn_distance is None
            or self.minimum_turn_distance is None
        ):
            self.turn_entry_distance = distance
            self.last_turn_distance = distance
            self.minimum_turn_distance = distance
            return

        previous = self.last_turn_distance
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

        drop = self.turn_entry_distance - self.minimum_turn_distance
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
        self.yolo_clear_frames = 0
        self.publish_candidate()
        self.publish_status("opposite_front_distance_increased")
        self.get_logger().warning(
            f"{self.trend_sensor_side} increased to {distance:.1f} cm "
            f"after minimum {self.minimum_turn_distance:.1f} cm: "
            "obstacle mode off, lane mode resumed"
        )

    def reset_for_next_obstacle(self) -> None:
        self.state = AvoidanceState.LANE_FOLLOW
        self.state_started = self.get_clock().now()
        self.latched_side = None
        self.trend_sensor_side = None
        self.pending_side = None
        self.side_frames = 0
        self.clear_frames = 0
        self.yolo_clear_frames = 0
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
        # Only TURN actually publishes a full-steer command; APPROACH and
        # REARM leave candidate_valid False so mission_manager falls back
        # to lane steering while avoidance is merely armed/watching.
        valid = self.state == AvoidanceState.TURN
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
                        "yolo_detected": self.yolo_detected,
                        "obstacle_side": self.latched_side,
                        "trend_sensor": self.trend_sensor_side,
                        "avoidance_active": self.state
                        in (
                            AvoidanceState.APPROACH,
                            AvoidanceState.TURN,
                            AvoidanceState.FAULT,
                        ),
                        "candidate_valid": self.state
                        == AvoidanceState.TURN,
                    },
                    ensure_ascii=False,
                )
            )
        )

    def destroy_node(self) -> bool:
        if hasattr(self, "candidate_valid_pub"):
            self.state = AvoidanceState.FAULT
            self.publish_candidate()
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
