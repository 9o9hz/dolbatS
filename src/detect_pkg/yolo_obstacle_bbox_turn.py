#!/usr/bin/env python3
"""Obstacle avoidance that finishes when the detected bbox exits center view."""

import math
from typing import Optional, Sequence

import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from yolo_obstacle_turn import AvoidanceState, YoloObstacleTurn


def bbox_crossed_exit_boundary(
    center_x: float,
    image_width: int,
    turn_side: str,
    left_boundary_ratio: float,
    right_boundary_ratio: float,
) -> bool:
    """Return whether the bbox center crossed the turn-specific exit line.

    During a left turn the obstacle moves toward the right side of the raw
    image, so it exits at the right boundary. A right turn is symmetric.
    """
    if image_width <= 0 or not math.isfinite(center_x):
        return False
    if turn_side == "left":
        return center_x >= image_width * right_boundary_ratio
    if turn_side == "right":
        return center_x <= image_width * left_boundary_ratio
    return False


class YoloObstacleBboxTurn(YoloObstacleTurn):
    """Keep the existing trigger states but finish TURN from bbox position."""

    def __init__(self) -> None:
        # The launch file remaps the inherited node name to
        # yolo_obstacle_bbox_turn, so it receives its independent YAML block.
        super().__init__()

        defaults = {
            "bbox_topic": "/detect/obstacle/bbox",
            "raw_image_topic": "/camera/lane/raw",
            "fallback_image_width_px": 640,
            "bbox_left_boundary_ratio": 0.25,
            "bbox_right_boundary_ratio": 0.75,
            "bbox_exit_consecutive_frames": 3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        parameter = lambda name: self.get_parameter(name).value

        self.fallback_image_width_px = max(
            1, int(parameter("fallback_image_width_px"))
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
        if not (
            0.0 <= self.bbox_left_boundary_ratio
            < self.bbox_right_boundary_ratio
            <= 1.0
        ):
            raise ValueError(
                "bbox boundary ratios must satisfy "
                "0 <= left < right <= 1"
            )

        self.raw_image_width = self.fallback_image_width_px
        self.bbox_exit_frames = 0
        self.last_bbox_center_x: Optional[float] = None

        self.raw_image_sub = self.create_subscription(
            Image,
            str(parameter("raw_image_topic")),
            self.on_raw_image,
            10,
        )
        self.bbox_sub = self.create_subscription(
            Float32MultiArray,
            str(parameter("bbox_topic")),
            self.on_bbox,
            10,
        )
        self.publish_status("bbox_exit_node_ready")
        self.get_logger().info(
            "BBox-exit avoidance ready: "
            f"left line={self.bbox_left_boundary_ratio:.3f}W, "
            f"right line={self.bbox_right_boundary_ratio:.3f}W, "
            f"debounce={self.bbox_exit_consecutive_frames} frames"
        )

    def on_raw_image(self, msg: Image) -> None:
        if msg.width > 0:
            self.raw_image_width = int(msg.width)

    def on_bbox(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 4:
            self.get_logger().warning(
                "Ignoring malformed obstacle bbox; expected [cx, cy, w, h]"
            )
            return

        center_x = float(msg.data[0])
        if not math.isfinite(center_x):
            return
        self.last_bbox_center_x = center_x

        if self.state != AvoidanceState.TURN:
            self.bbox_exit_frames = 0
            return

        crossed = bbox_crossed_exit_boundary(
            center_x,
            self.raw_image_width,
            self.latched_side or "",
            self.bbox_left_boundary_ratio,
            self.bbox_right_boundary_ratio,
        )
        self.bbox_exit_frames = (
            self.bbox_exit_frames + 1 if crossed else 0
        )
        if self.bbox_exit_frames >= self.bbox_exit_consecutive_frames:
            self.finish_turn_from_bbox(center_x)

    def update_turn_trend(self, distance: float) -> None:
        # This variant intentionally does not use the opposite-front
        # ultrasonic decrease/increase trend as its TURN exit condition.
        del distance

    def start_turn(self, opposite_front: Optional[float]) -> None:
        del opposite_front
        self.state = AvoidanceState.TURN
        self.state_started = self.get_clock().now()
        self.bbox_exit_frames = 0
        self.last_bbox_center_x = None
        self.trend_sensor_side = (
            "bbox_right_boundary"
            if self.latched_side == "left"
            else "bbox_left_boundary"
        )
        self.publish_candidate()
        self.publish_status("latched_conditions_ready_start_bbox_turn")
        self.get_logger().warning(
            "YOLO and rear detect-then-clear states ready: "
            f"full-steering {self.latched_side}; "
            f"tracking {self.trend_sensor_side}"
        )

    def finish_turn_from_bbox(self, center_x: float) -> None:
        boundary_ratio = (
            self.bbox_right_boundary_ratio
            if self.latched_side == "left"
            else self.bbox_left_boundary_ratio
        )
        boundary_x = self.raw_image_width * boundary_ratio
        crossed_side = "right" if self.latched_side == "left" else "left"

        self.state = AvoidanceState.REARM
        self.state_started = self.get_clock().now()
        self.yolo_clear_frames = 0
        self.publish_candidate()
        self.publish_status(f"bbox_crossed_{crossed_side}_boundary")
        self.get_logger().warning(
            f"BBox center x={center_x:.1f}px crossed {crossed_side} "
            f"boundary x={boundary_x:.1f}px: obstacle mode off, "
            "lane mode resumed"
        )

    def reset_for_next_obstacle(self) -> None:
        self.bbox_exit_frames = 0
        self.last_bbox_center_x = None
        super().reset_for_next_obstacle()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleBboxTurn()
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
