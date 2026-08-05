#!/usr/bin/env python3
"""Obstacle avoidance ending on either YOLO bbox or sonic threshold."""

from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException

from yolo_obstacle_bbox_turn import YoloObstacleBboxTurn
from yolo_obstacle_turn import YoloObstacleTurn


class YoloObstacleFusionEnd(YoloObstacleBboxTurn):
    """Finish TURN when either inherited bbox or sonic condition succeeds."""

    def update_turn_end(self, distance: float) -> None:
        # YoloObstacleBboxTurn intentionally disables this inherited sonic
        # condition. Call the base implementation to combine it with the bbox
        # callback using OR semantics: whichever finishes first changes state.
        YoloObstacleTurn.update_turn_end(self, distance)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloObstacleFusionEnd()
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
