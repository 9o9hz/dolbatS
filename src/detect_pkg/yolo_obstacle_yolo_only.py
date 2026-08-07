#!/usr/bin/env python3
"""Direction-selectable YOLO avoidance ending at a bbox exit line."""

import json
import math
from enum import Enum, auto
from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from yolo_bbox_utils import bbox_crossed_exit_boundary


class YoloOnlyState(Enum):
    WAIT_TRIGGER = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    WAIT_CLEAR = auto()


def turn_state_from_direction(direction: str) -> YoloOnlyState:
    """Convert an L/R direction parameter into its turn state."""
    normalized = direction.strip().upper()
    if normalized == "L":
        return YoloOnlyState.TURN_LEFT
    if normalized == "R":
        return YoloOnlyState.TURN_RIGHT
    raise ValueError("avoid_direction must be 'L' or 'R'")


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
            "avoid_direction": "L",
            "bbox_left_boundary_ratio": 0.25,
            "bbox_right_boundary_ratio": 0.75,
            "bbox_exit_consecutive_frames": 3,
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
        self.avoid_direction = str(
            parameter("avoid_direction")
        ).strip().upper()
        self.configured_turn_state = turn_state_from_direction(
            self.avoid_direction
        )
        self.bbox_left_boundary_ratio = float(
            parameter("bbox_left_boundary_ratio")
        )
        self.bbox_right_boundary_ratio = float(
            parameter("bbox_right_boundary_ratio")
        )
        self.bbox_exit_consecutive_frames = max(
            1, int(parameter("bbox_exit_consecutive_frames"))
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
        if not (
            0.0 <= self.bbox_left_boundary_ratio
            < self.bbox_right_boundary_ratio
            <= 1.0
        ):
            raise ValueError(
                "bbox boundary ratios must satisfy "
                "0 <= left < right <= 1"
            )

        self.state = YoloOnlyState.WAIT_TRIGGER
        self.yolo_detected = False
        self.clear_frames = 0
        self.bbox_exit_frames = 0
        self.last_bbox_center_x: Optional[float] = None

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
        self.get_logger().info(
            "Centered-bbox avoidance ready: "
            f"middle={self.middle_left_ratio:.3f}W.."
            f"{self.middle_right_ratio:.3f}W, "
            f"min area={self.min_bbox_area_ratio:.3f}, "
            f"turn={self.avoid_direction} "
            f"({self.full_steer_angle_deg:.1f} deg); "
            f"exit lines={self.bbox_left_boundary_ratio:.3f}W/"
            f"{self.bbox_right_boundary_ratio:.3f}W, "
            f"debounce={self.bbox_exit_consecutive_frames} frames"
        )

    def on_raw_image(self, msg: Image) -> None:
        if msg.width > 0 and msg.height > 0:
            self.image_width = int(msg.width)
            self.image_height = int(msg.height)

    def on_yolo_detected(self, msg: Bool) -> None:
        self.yolo_detected = bool(msg.data)
        if self.state != YoloOnlyState.WAIT_CLEAR:
            return
        self.clear_frames = (
            0 if self.yolo_detected else self.clear_frames + 1
        )
        if self.clear_frames >= self.rearm_clear_frames:
            self.state = YoloOnlyState.WAIT_TRIGGER
            self.clear_frames = 0
            self.bbox_exit_frames = 0
            self.last_bbox_center_x = None
            self.publish_status("rearmed_after_yolo_clear")

    def on_bbox(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.get_logger().warning(
                "Ignoring malformed bbox; expected [cx, cy, w, h]"
            )
            return

        center_x, _, bbox_width, bbox_height = (
            float(value) for value in msg.data[:4]
        )
        if not math.isfinite(center_x):
            return
        self.last_bbox_center_x = center_x

        if self.state in (YoloOnlyState.TURN_LEFT, YoloOnlyState.TURN_RIGHT):
            direction = (
                "left" if self.state == YoloOnlyState.TURN_LEFT else "right"
            )
            crossed = bbox_crossed_exit_boundary(
                center_x,
                self.image_width,
                direction,
                self.bbox_left_boundary_ratio,
                self.bbox_right_boundary_ratio,
            )
            self.bbox_exit_frames = (
                self.bbox_exit_frames + 1 if crossed else 0
            )
            if self.bbox_exit_frames >= self.bbox_exit_consecutive_frames:
                self.finish_turn_from_bbox(center_x, direction)
            return

        if self.state != YoloOnlyState.WAIT_TRIGGER:
            return

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

        self.state = self.configured_turn_state
        self.bbox_exit_frames = 0
        self.publish_candidate()
        direction = (
            "left" if self.state == YoloOnlyState.TURN_LEFT else "right"
        )
        self.publish_status(f"centered_bbox_triggered_{direction}_turn")
        self.get_logger().warning(
            "Centered bbox reached configured size: starting "
            f"full-{direction} turn"
        )

    def finish_turn_from_bbox(self, center_x: float, direction: str) -> None:
        boundary_ratio = (
            self.bbox_right_boundary_ratio
            if direction == "left"
            else self.bbox_left_boundary_ratio
        )
        boundary_x = self.image_width * boundary_ratio
        crossed_side = "right" if direction == "left" else "left"

        self.state = YoloOnlyState.WAIT_CLEAR
        self.clear_frames = 0
        self.publish_candidate()
        self.publish_status(f"bbox_crossed_{crossed_side}_boundary")
        self.get_logger().warning(
            f"BBox center x={center_x:.1f}px crossed {crossed_side} "
            f"boundary x={boundary_x:.1f}px: obstacle mode off, "
            "lane mode resumed"
        )

    def on_publish_timer(self) -> None:
        self.publish_candidate()

    def publish_candidate(self) -> None:
        active = self.state in (
            YoloOnlyState.TURN_LEFT,
            YoloOnlyState.TURN_RIGHT,
        )
        if self.state == YoloOnlyState.TURN_LEFT:
            steer = self.full_steer_angle_deg
        elif self.state == YoloOnlyState.TURN_RIGHT:
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
                        "avoid_direction": self.avoid_direction,
                        "bbox_center_x": self.last_bbox_center_x,
                        "bbox_exit_frames": self.bbox_exit_frames,
                        "avoidance_active": self.state
                        in (
                            YoloOnlyState.TURN_LEFT,
                            YoloOnlyState.TURN_RIGHT,
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        )

    def destroy_node(self) -> bool:
        if rclpy.ok() and hasattr(self, "candidate_valid_pub"):
            self.state = YoloOnlyState.WAIT_CLEAR
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
