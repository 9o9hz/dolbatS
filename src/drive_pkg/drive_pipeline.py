#!/usr/bin/env python3
"""Compatibility runner for the three topic-connected drive nodes."""

from __future__ import annotations

from typing import Optional, Sequence

import rclpy
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
)

from lane_detect import LaneDetectNode
from path_plan import PathPlanNode
from pure_pursuit import PurePursuitNode


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    nodes = []
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        for node_type in (
            LaneDetectNode,
            PathPlanNode,
            PurePursuitNode,
        ):
            node = node_type()
            nodes.append(node)
            executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
