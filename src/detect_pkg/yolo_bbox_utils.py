"""Shared YOLO bbox boundary helpers for obstacle avoidance nodes."""

import math


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
