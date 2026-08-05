#!/usr/bin/env python3
"""Publish obstacle steering candidates from YOLO and four ultrasonics."""

import json
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class AvoidanceState(Enum):
    LANE_FOLLOW = auto()
    TURN = auto()
    REARM = auto()
    FAULT = auto()


class RearObstacleState(Enum):
    WAIT_DETECTION = auto()
    DETECTED = auto()
    CLEARED_AFTER_DETECTION = auto()


def should_monitor_ultrasonic(
    yolo_trigger_required: bool,
    yolo_gate_open: bool,
) -> bool:
    """Monitor immediately when the optional YOLO gate is disabled."""
    return not yolo_trigger_required or yolo_gate_open


def turn_end_threshold_reached(distance: float, threshold: float) -> bool:
    """Return whether a valid distance strictly exceeds the turn-end limit."""
    return (
        math.isfinite(distance)
        and math.isfinite(threshold)
        and distance >= 0.0
        and distance > threshold
    )


@dataclass
class LeftHalfDisappearanceTrigger:
    """Enable ultrasonic processing after a left-half YOLO target vanishes."""

    boundary_ratio: float = 0.5
    missing_frames_required: int = 3
    seen_in_left_half: bool = False
    missing_frames: int = 0
    enabled: bool = False

    def observe_bbox(self, center_x: float, image_width: int) -> bool:
        """Record a valid detection and return whether left-half was newly seen."""
        if self.enabled or image_width <= 0 or not math.isfinite(center_x):
            return False
        newly_seen = (
            not self.seen_in_left_half
            and center_x < image_width * self.boundary_ratio
        )
        if newly_seen:
            self.seen_in_left_half = True
        self.missing_frames = 0
        return newly_seen

    def observe_detection(self, detected: bool) -> bool:
        """Return True only on the frame that opens the ultrasonic gate."""
        if self.enabled:
            return False
        if detected:
            self.missing_frames = 0
            return False
        if not self.seen_in_left_half:
            return False

        self.missing_frames += 1
        if self.missing_frames < self.missing_frames_required:
            return False
        self.enabled = True
        return True

    def reset(self) -> None:
        self.seen_in_left_half = False
        self.missing_frames = 0
        self.enabled = False


class YoloObstacleTurn(Node):
    """Generate an obstacle candidate for mission_manager.

    A YOLO bbox must first appear in the left half of the image and then
    disappear for the configured number of frames. That sequence opens the
    ultrasonic gate. A rear-ultrasonic detect-then-clear event then starts
    full steering toward the rear sensor side. Avoidance ends as soon as the
    opposite-side front sensor exceeds the configured threshold.

    candidate_valid and avoidance_active are only True in TURN. While either
    condition is still pending, mission_manager continues lane driving.
    """

    def __init__(self) -> None:
        super().__init__("yolo_obstacle_turn")

        defaults = {
            "yolo_detected_topic": "/detect/obstacle/detected",
            "bbox_topic": "/detect/obstacle/bbox",
            "fallback_image_width_px": 640,
            "left_half_boundary_ratio": 0.5,
            "yolo_disappear_consecutive_frames": 3,
            "yolo_ultrasonic_trigger_enabled": True,
            "ultrasonic_enable_topic": (
                "/detect/obstacle/ultrasonic_enabled"
            ),
            "left_front_distance_topic": (
                "/sonic/left/front"
            ),
            "left_rear_distance_topic": (
                "/sonic/left/rear"
            ),
            "right_front_distance_topic": (
                "/sonic/right/front"
            ),
            "right_rear_distance_topic": (
                "/sonic/right/rear"
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
            "full_steer_angle_deg": 20.0,
            "turn_end_threshold_cm": 70.0,
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
        self.fallback_image_width_px = max(
            1, int(parameter("fallback_image_width_px"))
        )
        left_half_boundary_ratio = float(
            parameter("left_half_boundary_ratio")
        )
        if not 0.0 < left_half_boundary_ratio < 1.0:
            raise ValueError(
                "left_half_boundary_ratio must be between 0 and 1"
            )
        self.yolo_trigger = LeftHalfDisappearanceTrigger(
            boundary_ratio=left_half_boundary_ratio,
            missing_frames_required=max(
                1,
                int(parameter("yolo_disappear_consecutive_frames")),
            ),
        )
        self.yolo_ultrasonic_trigger_enabled = bool(
            parameter("yolo_ultrasonic_trigger_enabled")
        )
        self.yolo_image_width = self.fallback_image_width_px
        self.rear_no_echo_is_clear = bool(
            parameter("rear_no_echo_is_clear")
        )
        self.full_steer_angle_deg = abs(
            float(parameter("full_steer_angle_deg"))
        )
        self.turn_end_threshold_cm = float(
            parameter("turn_end_threshold_cm")
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
        if self.turn_end_threshold_cm <= 0.0:
            raise ValueError("turn_end_threshold_cm must be positive")

        self.state = AvoidanceState.LANE_FOLLOW
        self.state_started = self.get_clock().now()
        self.rear_obstacle_state = RearObstacleState.WAIT_DETECTION
        self.yolo_detected = False
        self.yolo_latched = False
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
        self.ultrasonic_enable_pub = self.create_publisher(
            Bool, str(parameter("ultrasonic_enable_topic")), 10
        )

        self.yolo_sub = self.create_subscription(
            Bool,
            str(parameter("yolo_detected_topic")),
            self.on_yolo_detected,
            10,
        )
        self.bbox_sub = self.create_subscription(
            Float32MultiArray,
            str(parameter("bbox_topic")),
            self.on_bbox,
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
            "Obstacle candidate ready: YOLO-to-ultrasonic trigger "
            f"{'enabled' if self.yolo_ultrasonic_trigger_enabled else 'bypassed'}; "
            "ultrasonic monitoring "
            f"{'waits for YOLO clear' if self.yolo_ultrasonic_trigger_enabled else 'always on'}; "
            f"detect <= {self.detect_threshold_cm:.1f} cm, "
            f"clear > {self.clear_threshold_cm:.1f} cm; "
            f"turn end: opposite-front > {self.turn_end_threshold_cm:.1f} cm"
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

        if not self.yolo_ultrasonic_trigger_enabled:
            return

        if self.state in (AvoidanceState.REARM, AvoidanceState.FAULT):
            self.yolo_clear_frames = (
                0 if self.yolo_detected else self.yolo_clear_frames + 1
            )
            if self.yolo_clear_frames >= self.consecutive_frames:
                self.reset_for_next_obstacle()
            return

        if self.state != AvoidanceState.LANE_FOLLOW:
            return

        if self.yolo_trigger.observe_detection(self.yolo_detected):
            self.yolo_latched = True
            self.publish_status(
                "left_half_yolo_disappeared_ultrasonic_enabled"
            )
            self.get_logger().warning(
                "Left-half YOLO target disappeared: ultrasonic obstacle "
                "detection enabled"
            )
            self.try_start_turn()

    def on_bbox(self, msg: Float32MultiArray) -> None:
        if not self.yolo_ultrasonic_trigger_enabled:
            return
        if len(msg.data) < 4:
            self.get_logger().warning(
                "Ignoring malformed obstacle bbox; expected [cx, cy, w, h]"
            )
            return
        center_x = float(msg.data[0])
        if self.yolo_trigger.observe_bbox(
            center_x, self.yolo_image_width
        ):
            self.publish_status("yolo_seen_in_left_half")
            boundary_x = (
                self.yolo_image_width
                * self.yolo_trigger.boundary_ratio
            )
            self.get_logger().info(
                f"YOLO target entered left half: x={center_x:.1f}px, "
                f"boundary={boundary_x:.1f}px"
            )

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

    def process_sensor_frame(
        self,
        left_front: Optional[float],
        left_rear: Optional[float],
        right_front: Optional[float],
        right_rear: Optional[float],
    ) -> None:
        if self.state == AvoidanceState.LANE_FOLLOW:
            if not self.is_ultrasonic_monitoring_enabled():
                return
            self.update_rear_obstacle_state(left_rear, right_rear)
            return

        if self.latched_side is None:
            return

        if self.state == AvoidanceState.TURN:
            opposite_front = (
                right_front
                if self.latched_side == "left"
                else left_front
            )
            if self.valid_distance(opposite_front):
                self.update_turn_end(float(opposite_front))

    def update_rear_obstacle_state(
        self,
        left_rear: Optional[float],
        right_rear: Optional[float],
    ) -> None:
        if (
            self.rear_obstacle_state
            == RearObstacleState.CLEARED_AFTER_DETECTION
        ):
            return

        if self.rear_obstacle_state == RearObstacleState.WAIT_DETECTION:
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
            self.rear_obstacle_state = RearObstacleState.DETECTED
            self.clear_frames = 0
            self.publish_status("rear_ultrasonic_detection_latched")
            self.get_logger().warning(
                f"{side} rear ultrasonic obstacle latched; waiting for "
                "that sensor to clear"
            )
            return

        if self.latched_side is None:
            return

        watched_rear = (
            left_rear if self.latched_side == "left" else right_rear
        )
        cleared = self.rear_obstacle_cleared(watched_rear)
        self.clear_frames = self.clear_frames + 1 if cleared else 0
        if self.clear_frames < self.consecutive_frames:
            return

        self.rear_obstacle_state = (
            RearObstacleState.CLEARED_AFTER_DETECTION
        )
        self.publish_status("rear_ultrasonic_cleared_after_detection")
        self.get_logger().warning(
            f"{self.latched_side} rear ultrasonic cleared after "
            "detection; state latched"
        )
        self.try_start_turn()

    def try_start_turn(self) -> None:
        if (
            self.state != AvoidanceState.LANE_FOLLOW
            or (
                self.yolo_ultrasonic_trigger_enabled
                and not self.yolo_latched
            )
            or self.rear_obstacle_state
            != RearObstacleState.CLEARED_AFTER_DETECTION
            or self.latched_side is None
        ):
            return

        opposite_front = (
            self.right_front_distance
            if self.latched_side == "left"
            else self.left_front_distance
        )
        self.start_turn(opposite_front)

    def start_turn(self, opposite_front: Optional[float]) -> None:
        del opposite_front
        self.state = AvoidanceState.TURN
        self.state_started = self.get_clock().now()
        self.publish_candidate()
        self.publish_status("latched_conditions_ready_start_turn")
        self.get_logger().warning(
            "YOLO and rear detect-then-clear states ready: "
            f"full-steering {self.latched_side}; "
            f"tracking {self.trend_sensor_side} until > "
            f"{self.turn_end_threshold_cm:.1f} cm"
        )

    def update_turn_end(self, distance: float) -> None:
        if turn_end_threshold_reached(
            distance, self.turn_end_threshold_cm
        ):
            self.finish_turn(distance)

    def finish_turn(self, distance: float) -> None:
        self.state = AvoidanceState.REARM
        self.state_started = self.get_clock().now()
        self.yolo_clear_frames = 0
        self.publish_candidate()
        self.publish_status("opposite_front_threshold_exceeded")
        self.get_logger().warning(
            f"{self.trend_sensor_side}={distance:.1f} cm exceeded "
            f"turn-end threshold {self.turn_end_threshold_cm:.1f} cm: "
            "obstacle mode off, lane mode resumed"
        )

    def reset_for_next_obstacle(self) -> None:
        self.state = AvoidanceState.LANE_FOLLOW
        self.state_started = self.get_clock().now()
        self.rear_obstacle_state = RearObstacleState.WAIT_DETECTION
        self.yolo_latched = False
        self.latched_side = None
        self.trend_sensor_side = None
        self.pending_side = None
        self.side_frames = 0
        self.clear_frames = 0
        self.yolo_clear_frames = 0
        self.yolo_trigger.reset()
        self.publish_candidate()
        self.publish_status("rearmed")

    def is_ultrasonic_monitoring_enabled(self) -> bool:
        return should_monitor_ultrasonic(
            self.yolo_ultrasonic_trigger_enabled,
            self.yolo_trigger.enabled,
        )

    def publish_candidate(self) -> None:
        elapsed = (
            self.get_clock().now() - self.state_started
        ).nanoseconds / 1e9
        if (
            self.state == AvoidanceState.TURN
            and self.turn_timeout_sec > 0.0
            and elapsed >= self.turn_timeout_sec
        ):
            self.enter_fault("turn_timeout")
            return

        active = self.state in (
            AvoidanceState.TURN,
            AvoidanceState.FAULT,
        )
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
        self.ultrasonic_enable_pub.publish(
            Bool(data=self.is_ultrasonic_monitoring_enabled())
        )

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
                        "yolo_latched": self.yolo_latched,
                        "yolo_seen_in_left_half": (
                            self.yolo_trigger.seen_in_left_half
                        ),
                        "yolo_missing_frames": (
                            self.yolo_trigger.missing_frames
                        ),
                        "ultrasonic_enabled": (
                            self.is_ultrasonic_monitoring_enabled()
                        ),
                        "yolo_ultrasonic_trigger_enabled": (
                            self.yolo_ultrasonic_trigger_enabled
                        ),
                        "rear_obstacle_state": (
                            self.rear_obstacle_state.name.lower()
                        ),
                        "obstacle_side": self.latched_side,
                        "trend_sensor": self.trend_sensor_side,
                        "avoidance_active": self.state
                        in (
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
        if rclpy.ok() and hasattr(self, "candidate_valid_pub"):
            self.state = AvoidanceState.FAULT
            self.publish_candidate()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleTurn()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
