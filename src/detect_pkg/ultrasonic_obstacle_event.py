#!/usr/bin/env python3
import math
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int8MultiArray


EVENT_CLEARED = 0
EVENT_DETECTED = 1
DIRECTION_RIGHT = -1
DIRECTION_NONE = 0
DIRECTION_LEFT = 1


class UltrasonicObstacleEvent(Node):
    """Turn noisy left/right ultrasonic samples into debounced state events."""

    def __init__(self) -> None:
        super().__init__("ultrasonic_obstacle_event")

        self.declare_parameter("left_distance_topic", "/ultrasonic/left_distance")
        self.declare_parameter("right_distance_topic", "/ultrasonic/right_distance")
        self.declare_parameter("event_topic", "/detect/obstacle_event")
        self.declare_parameter("detect_threshold_cm", 40.0)
        self.declare_parameter("clear_threshold_cm", 45.0)
        self.declare_parameter("consecutive_frames", 3)
        self.declare_parameter("default_avoid_direction", DIRECTION_LEFT)

        left_topic = str(self.get_parameter("left_distance_topic").value)
        right_topic = str(self.get_parameter("right_distance_topic").value)
        event_topic = str(self.get_parameter("event_topic").value)
        self.detect_threshold_cm = float(
            self.get_parameter("detect_threshold_cm").value
        )
        self.clear_threshold_cm = float(
            self.get_parameter("clear_threshold_cm").value
        )
        self.consecutive_frames = max(
            1, int(self.get_parameter("consecutive_frames").value)
        )
        default_direction = int(self.get_parameter("default_avoid_direction").value)
        self.default_avoid_direction = (
            default_direction
            if default_direction in (DIRECTION_LEFT, DIRECTION_RIGHT)
            else DIRECTION_LEFT
        )

        if self.detect_threshold_cm <= 0.0:
            raise ValueError("detect_threshold_cm must be greater than zero")
        if self.clear_threshold_cm < self.detect_threshold_cm:
            raise ValueError(
                "clear_threshold_cm must be greater than or equal to "
                "detect_threshold_cm"
            )

        self.left_distance: Optional[float] = None
        self.obstacle_detected = False
        self.pending_frames = 0
        self.last_avoid_direction = DIRECTION_NONE

        self.event_pub = self.create_publisher(Int8MultiArray, event_topic, 10)
        self.left_subscription = self.create_subscription(
            Float32, left_topic, self.on_left_distance, 10
        )
        self.right_subscription = self.create_subscription(
            Float32, right_topic, self.on_right_distance, 10
        )

        self.get_logger().info(
            f"Ultrasonic event filter: {left_topic}, {right_topic} -> {event_topic}; "
            f"detect <= {self.detect_threshold_cm:.1f} cm, "
            f"clear > {self.clear_threshold_cm:.1f} cm, "
            f"{self.consecutive_frames} consecutive frames"
        )

    @staticmethod
    def is_valid_distance(distance: Optional[float]) -> bool:
        return (
            distance is not None
            and math.isfinite(distance)
            and distance >= 0.0
        )

    def on_left_distance(self, msg: Float32) -> None:
        self.left_distance = float(msg.data)

    def on_right_distance(self, msg: Float32) -> None:
        # serial_bridge publishes left immediately before right, so a right sample
        # completes one left/right sensor frame.
        self.process_frame(self.left_distance, float(msg.data))

    def process_frame(
        self, left_distance: Optional[float], right_distance: Optional[float]
    ) -> None:
        left_valid = self.is_valid_distance(left_distance)
        right_valid = self.is_valid_distance(right_distance)

        if not self.obstacle_detected:
            left_close = left_valid and left_distance <= self.detect_threshold_cm
            right_close = right_valid and right_distance <= self.detect_threshold_cm
            desired_state = left_close or right_close
        else:
            # Do not announce "cleared" unless both sensors returned valid,
            # sufficiently distant readings.
            desired_state = not (
                left_valid
                and right_valid
                and left_distance > self.clear_threshold_cm
                and right_distance > self.clear_threshold_cm
            )

        if desired_state == self.obstacle_detected:
            self.pending_frames = 0
            return

        self.pending_frames += 1
        if self.pending_frames < self.consecutive_frames:
            return

        self.pending_frames = 0
        self.obstacle_detected = desired_state
        if desired_state:
            self.last_avoid_direction = self.choose_avoid_direction(
                left_distance, right_distance
            )
            self.publish_event(EVENT_DETECTED, self.last_avoid_direction)
        else:
            # The clear event carries the same direction as its detect event.
            self.publish_event(EVENT_CLEARED, self.last_avoid_direction)

    def choose_avoid_direction(
        self, left_distance: Optional[float], right_distance: Optional[float]
    ) -> int:
        left_close = (
            self.is_valid_distance(left_distance)
            and left_distance <= self.detect_threshold_cm
        )
        right_close = (
            self.is_valid_distance(right_distance)
            and right_distance <= self.detect_threshold_cm
        )

        if left_close and not right_close:
            return DIRECTION_RIGHT
        if right_close and not left_close:
            return DIRECTION_LEFT

        if left_close and right_close and left_distance != right_distance:
            # When both sides are blocked, steer toward the side with more room.
            return (
                DIRECTION_LEFT
                if left_distance > right_distance
                else DIRECTION_RIGHT
            )
        return self.default_avoid_direction

    def publish_event(self, event: int, direction: int) -> None:
        self.event_pub.publish(Int8MultiArray(data=[event, direction]))
        event_name = "detected" if event == EVENT_DETECTED else "cleared"
        direction_name = "left" if direction == DIRECTION_LEFT else "right"
        self.get_logger().info(
            f"Obstacle {event_name}; avoid_direction={direction_name}"
        )


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = UltrasonicObstacleEvent()
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
