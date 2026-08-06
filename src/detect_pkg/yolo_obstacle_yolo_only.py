#!/usr/bin/env python3
"""Alternating timed avoidance triggered by a centered, large YOLO bbox."""

import json
import math
from enum import Enum, auto
from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class TimedTurnState(Enum):
    WAIT_TRIGGER = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    COUNTERSTEER_LEFT = auto()
    COUNTERSTEER_RIGHT = auto()
    WAIT_CLEAR = auto()


def next_timed_turn_state(
    state: TimedTurnState,
    elapsed_sec: float,
    duration_sec: float,
    enable_opposite_steer: bool = False,
) -> TimedTurnState:
    """Advance a timed turn, optionally through an equal countersteer."""
    if elapsed_sec < duration_sec:
        return state
    if state == TimedTurnState.TURN_LEFT:
        return (
            TimedTurnState.COUNTERSTEER_RIGHT
            if enable_opposite_steer
            else TimedTurnState.WAIT_CLEAR
        )
    if state == TimedTurnState.TURN_RIGHT:
        return (
            TimedTurnState.COUNTERSTEER_LEFT
            if enable_opposite_steer
            else TimedTurnState.WAIT_CLEAR
        )
    if state in (
        TimedTurnState.COUNTERSTEER_LEFT,
        TimedTurnState.COUNTERSTEER_RIGHT,
    ):
        return TimedTurnState.WAIT_CLEAR
    return state


def opposite_turn_state(state: TimedTurnState) -> TimedTurnState:
    """Return the direction to use for the next obstacle."""
    if state == TimedTurnState.TURN_LEFT:
        return TimedTurnState.TURN_RIGHT
    return TimedTurnState.TURN_LEFT


def centered_bbox_is_large_enough(
    center_x: float,
    bbox_width: float,
    bbox_height: float,
    image_width: int,
    image_height: int,
    middle_left_ratio: float,
    middle_right_ratio: float,
    min_bbox_area_ratio: float,
) -> bool:
    """Return whether the bbox center and area satisfy the trigger settings."""
    values = (center_x, bbox_width, bbox_height, min_bbox_area_ratio)
    if image_width <= 0 or image_height <= 0:
        return False
    if not all(math.isfinite(value) for value in values):
        return False
    if bbox_width <= 0.0 or bbox_height <= 0.0:
        return False

    center_is_middle = (
        image_width * middle_left_ratio
        <= center_x
        <= image_width * middle_right_ratio
    )
    bbox_area_ratio = (
        bbox_width * bbox_height / float(image_width * image_height)
    )
    return center_is_middle and bbox_area_ratio >= min_bbox_area_ratio


class YoloObstacleYoloOnly(Node):
    def __init__(self) -> None:
        super().__init__("yolo_obstacle_yolo_only")

        defaults = {
            "yolo_detected_topic": "/detect/obstacle/detected",
            "bbox_topic": "/detect/obstacle/bbox",
            "raw_image_topic": "/camera/lane/raw",
            "fallback_image_width_px": 640,
            "fallback_image_height_px": 480,
            "middle_left_ratio": 1.0 / 3.0,
            "middle_right_ratio": 2.0 / 3.0,
            "min_bbox_area_ratio": 0.08,
            "full_steer_angle_deg": 25.0,
            "left_steer_duration_sec": 1.0,
            "right_steer_duration_sec": 1.0,
            "enable_opposite_steer": False,
            "rearm_clear_frames": 3,
            "avoidance_active_topic": "/detect/obstacle/avoidance_active",
            "candidate_steer_topic": (
                "/control/candidate/obstacle/steer_angle"
            ),
            "candidate_valid_topic": "/control/candidate/obstacle/valid",
            "status_topic": "/detect/avoidance/status",
            "publish_rate_hz": 30.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        parameter = lambda name: self.get_parameter(name).value

        self.image_width = max(
            1, int(parameter("fallback_image_width_px"))
        )
        self.image_height = max(
            1, int(parameter("fallback_image_height_px"))
        )
        self.middle_left_ratio = float(parameter("middle_left_ratio"))
        self.middle_right_ratio = float(parameter("middle_right_ratio"))
        self.min_bbox_area_ratio = float(parameter("min_bbox_area_ratio"))
        self.full_steer_angle_deg = abs(
            float(parameter("full_steer_angle_deg"))
        )
        self.left_steer_duration_sec = float(
            parameter("left_steer_duration_sec")
        )
        self.right_steer_duration_sec = float(
            parameter("right_steer_duration_sec")
        )
        self.enable_opposite_steer = bool(
            parameter("enable_opposite_steer")
        )
        self.rearm_clear_frames = max(
            1, int(parameter("rearm_clear_frames"))
        )
        publish_rate_hz = max(1.0, float(parameter("publish_rate_hz")))

        if not (
            0.0 <= self.middle_left_ratio
            < self.middle_right_ratio
            <= 1.0
        ):
            raise ValueError(
                "middle ratios must satisfy 0 <= left < right <= 1"
            )
        if not 0.0 < self.min_bbox_area_ratio <= 1.0:
            raise ValueError("min_bbox_area_ratio must be in (0, 1]")
        if self.full_steer_angle_deg <= 0.0:
            raise ValueError("full_steer_angle_deg must be positive")
        if self.left_steer_duration_sec <= 0.0:
            raise ValueError("left_steer_duration_sec must be positive")
        if self.right_steer_duration_sec <= 0.0:
            raise ValueError("right_steer_duration_sec must be positive")

        self.state = TimedTurnState.WAIT_TRIGGER
        self.state_started = self.get_clock().now()
        self.next_turn_state = TimedTurnState.TURN_LEFT
        self.yolo_detected = False
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
        self.raw_image_sub = self.create_subscription(
            Image,
            str(parameter("raw_image_topic")),
            self.on_raw_image,
            10,
        )
        self.publish_timer = self.create_timer(
            1.0 / publish_rate_hz, self.on_publish_timer
        )

        self.publish_candidate()
        self.publish_status("ready")
        opposite_steer_status = (
            "on" if self.enable_opposite_steer else "off"
        )
        self.get_logger().info(
            "Centered-bbox timed avoidance ready: "
            f"middle={self.middle_left_ratio:.3f}W.."
            f"{self.middle_right_ratio:.3f}W, "
            f"min area={self.min_bbox_area_ratio:.3f}, "
            f"alternating turn={self.full_steer_angle_deg:.1f} deg for "
            f"left={self.left_steer_duration_sec:.2f} sec / "
            f"right={self.right_steer_duration_sec:.2f} sec per obstacle; "
            f"equal opposite steer={opposite_steer_status}; "
            "first direction=left"
        )

    def on_raw_image(self, msg: Image) -> None:
        if msg.width > 0 and msg.height > 0:
            self.image_width = int(msg.width)
            self.image_height = int(msg.height)

    def on_yolo_detected(self, msg: Bool) -> None:
        self.yolo_detected = bool(msg.data)
        if self.state != TimedTurnState.WAIT_CLEAR:
            return
        self.clear_frames = (
            0 if self.yolo_detected else self.clear_frames + 1
        )
        if self.clear_frames >= self.rearm_clear_frames:
            self.state = TimedTurnState.WAIT_TRIGGER
            self.state_started = self.get_clock().now()
            self.clear_frames = 0
            self.publish_status("rearmed_after_yolo_clear")

    def on_bbox(self, msg: Float32MultiArray) -> None:
        if self.state != TimedTurnState.WAIT_TRIGGER:
            return
        if len(msg.data) < 4:
            self.get_logger().warning(
                "Ignoring malformed bbox; expected [cx, cy, w, h]"
            )
            return

        center_x, _, bbox_width, bbox_height = (
            float(value) for value in msg.data[:4]
        )
        if not centered_bbox_is_large_enough(
            center_x,
            bbox_width,
            bbox_height,
            self.image_width,
            self.image_height,
            self.middle_left_ratio,
            self.middle_right_ratio,
            self.min_bbox_area_ratio,
        ):
            return

        self.state = self.next_turn_state
        self.state_started = self.get_clock().now()
        self.publish_candidate()
        direction = (
            "left" if self.state == TimedTurnState.TURN_LEFT else "right"
        )
        self.publish_status(f"centered_bbox_triggered_{direction}_turn")
        self.get_logger().warning(
            "Centered bbox reached configured size: starting "
            f"full-{direction} turn"
        )

    def duration_for_state(self, state: TimedTurnState) -> float:
        if state in (
            TimedTurnState.TURN_LEFT,
            TimedTurnState.COUNTERSTEER_LEFT,
        ):
            return self.left_steer_duration_sec
        if state in (
            TimedTurnState.TURN_RIGHT,
            TimedTurnState.COUNTERSTEER_RIGHT,
        ):
            return self.right_steer_duration_sec
        return self.left_steer_duration_sec

    def on_publish_timer(self) -> None:
        elapsed = (
            self.get_clock().now() - self.state_started
        ).nanoseconds / 1e9
        next_state = next_timed_turn_state(
            self.state,
            elapsed,
            self.duration_for_state(self.state),
            self.enable_opposite_steer,
        )
        if next_state != self.state and self.state in (
            TimedTurnState.TURN_LEFT,
            TimedTurnState.TURN_RIGHT,
        ):
            completed_turn = self.state
            completed_direction = (
                "left"
                if completed_turn == TimedTurnState.TURN_LEFT
                else "right"
            )
            self.next_turn_state = opposite_turn_state(completed_turn)
            self.state = next_state
            self.state_started = self.get_clock().now()
            self.clear_frames = 0
            if next_state == TimedTurnState.WAIT_CLEAR:
                self.publish_status(
                    f"timed_{completed_direction}_turn_complete_lane_resumed"
                )
                self.get_logger().warning(
                    f"Full-{completed_direction} duration complete: "
                    "obstacle mode off, lane mode resumed"
                )
            else:
                opposite_direction = (
                    "left"
                    if next_state == TimedTurnState.COUNTERSTEER_LEFT
                    else "right"
                )
                self.publish_status(
                    f"timed_{completed_direction}_turn_complete_"
                    f"countersteer_{opposite_direction}_started"
                )
                self.get_logger().warning(
                    f"Full-{completed_direction} duration complete: "
                    f"starting equal full-{opposite_direction} countersteer"
                )
        elif next_state != self.state and self.state in (
            TimedTurnState.COUNTERSTEER_LEFT,
            TimedTurnState.COUNTERSTEER_RIGHT,
        ):
            completed_direction = (
                "left"
                if self.state == TimedTurnState.COUNTERSTEER_LEFT
                else "right"
            )
            self.state = next_state
            self.state_started = self.get_clock().now()
            self.clear_frames = 0
            self.publish_status(
                f"countersteer_{completed_direction}_complete_lane_resumed"
            )
            self.get_logger().warning(
                f"Full-{completed_direction} countersteer complete: "
                "obstacle mode off, lane mode resumed"
            )
        self.publish_candidate()

    def publish_candidate(self) -> None:
        active = self.state in (
            TimedTurnState.TURN_LEFT,
            TimedTurnState.TURN_RIGHT,
            TimedTurnState.COUNTERSTEER_LEFT,
            TimedTurnState.COUNTERSTEER_RIGHT,
        )
        if self.state in (
            TimedTurnState.TURN_LEFT,
            TimedTurnState.COUNTERSTEER_LEFT,
        ):
            steer = self.full_steer_angle_deg
        elif self.state in (
            TimedTurnState.TURN_RIGHT,
            TimedTurnState.COUNTERSTEER_RIGHT,
        ):
            steer = -self.full_steer_angle_deg
        else:
            steer = 0.0
        self.candidate_steer_pub.publish(Float32(data=steer))
        self.candidate_valid_pub.publish(Bool(data=active))
        self.avoidance_active_pub.publish(Bool(data=active))

    def publish_status(self, reason: str) -> None:
        self.status_pub.publish(
            String(
                data=json.dumps(
                    {
                        "state": self.state.name.lower(),
                        "reason": reason,
                        "yolo_detected": self.yolo_detected,
                        "image_width": self.image_width,
                        "image_height": self.image_height,
                        "min_bbox_area_ratio": self.min_bbox_area_ratio,
                        "next_turn_direction": (
                            "left"
                            if self.next_turn_state
                            == TimedTurnState.TURN_LEFT
                            else "right"
                        ),
                        "avoidance_active": self.state
                        in (
                            TimedTurnState.TURN_LEFT,
                            TimedTurnState.TURN_RIGHT,
                            TimedTurnState.COUNTERSTEER_LEFT,
                            TimedTurnState.COUNTERSTEER_RIGHT,
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )

    def destroy_node(self) -> bool:
        if rclpy.ok() and hasattr(self, "candidate_valid_pub"):
            self.state = TimedTurnState.WAIT_CLEAR
            self.publish_candidate()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleYoloOnly()
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
