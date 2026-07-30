#!/usr/bin/env python3
"""Spacebar start/stop switch for the vehicle drive-enable gate."""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool


class KeyboardDriveToggle(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_drive_toggle")
        self.declare_parameter("enable_topic", "/drive/enabled")
        self.declare_parameter("poll_period_sec", 0.05)
        self.enable_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("enable_topic").value),
            10,
        )
        self.enabled = False
        self._terminal_settings = None
        if sys.stdin.isatty():
            self._terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            self.get_logger().warning(
                "stdin is not a terminal; keyboard toggle is unavailable"
            )
        self.enable_publisher.publish(Bool(data=False))
        self.timer = self.create_timer(
            max(
                0.01,
                float(self.get_parameter("poll_period_sec").value),
            ),
            self.poll_keyboard,
        )
        self.get_logger().info(
            "DRIVE DISABLED — press SPACE to start/stop; Ctrl+C exits"
        )

    def poll_keyboard(self) -> None:
        if self._terminal_settings is None:
            return
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return
        key = sys.stdin.read(1)
        if key != " ":
            return
        self.enabled = not self.enabled
        self.enable_publisher.publish(Bool(data=self.enabled))
        state = "ENABLED" if self.enabled else "DISABLED"
        self.get_logger().warning(f"DRIVE {state}")

    def destroy_node(self) -> bool:
        if hasattr(self, "enable_publisher") and rclpy.ok():
            self.enable_publisher.publish(Bool(data=False))
        if self._terminal_settings is not None:
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self._terminal_settings,
            )
            self._terminal_settings = None
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[KeyboardDriveToggle] = None
    try:
        node = KeyboardDriveToggle()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
