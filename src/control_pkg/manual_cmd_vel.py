#!/usr/bin/env python3
"""Keyboard teleoperation node that publishes Ackermann-compatible cmd_vel."""

import curses
import math
import time
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ManualCmdVel(Node):
    """Convert arrow-key input into geometry_msgs/Twist commands."""

    def __init__(self) -> None:
        super().__init__("manual_cmd_vel")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("speed", 0.9)
        self.declare_parameter("wheelbase", 0.545)
        self.declare_parameter("steer_step_deg", 10.0)
        self.declare_parameter("max_steer_deg", 20.0)
        self.declare_parameter("key_timeout", 0.25)
        self.declare_parameter("publish_rate", 20.0)

        topic = str(self.get_parameter("cmd_vel_topic").value)
        self.speed = abs(float(self.get_parameter("speed").value))
        self.wheelbase = max(
            1e-6, abs(float(self.get_parameter("wheelbase").value))
        )
        self.steer_step_deg = abs(
            float(self.get_parameter("steer_step_deg").value)
        )
        self.max_steer_deg = abs(
            float(self.get_parameter("max_steer_deg").value)
        )
        self.key_timeout = max(
            0.05, float(self.get_parameter("key_timeout").value)
        )
        publish_rate = max(
            1.0, float(self.get_parameter("publish_rate").value)
        )

        self.publisher = self.create_publisher(Twist, topic, 10)
        self.steering_deg = 0.0
        self.direction = 0
        self.last_drive_key_time = 0.0
        self.last_steer_key_time = 0.0
        self.quit_requested = False
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_command)

        self.get_logger().info(
            f"Manual driving ready: publishing to {topic} at {publish_rate:.1f} Hz"
        )

    def handle_key(self, key: int) -> None:
        """Update the desired speed and steering from one curses key code."""
        now = time.monotonic()

        if key in (ord("q"), ord("Q")):
            self.quit_requested = True
        elif key == curses.KEY_UP:
            self.direction = 1
            self.last_drive_key_time = now
        elif key == curses.KEY_DOWN:
            self.direction = -1
            self.last_drive_key_time = now
        elif key in (ord(" "), ord("s"), ord("S")):
            self.direction = 0
            self.last_drive_key_time = 0.0
        elif key in (ord("c"), ord("C")):
            self.steering_deg = 0.0
        elif key == curses.KEY_LEFT:
            self._change_steering(self.steer_step_deg, now)
        elif key == curses.KEY_RIGHT:
            self._change_steering(-self.steer_step_deg, now)

    def _change_steering(self, change_deg: float, now: float) -> None:
        # Limit changes while a held key is generating terminal key-repeat events.
        if now - self.last_steer_key_time < 0.08:
            return
        self.steering_deg = max(
            -self.max_steer_deg,
            min(self.max_steer_deg, self.steering_deg + change_deg),
        )
        self.last_steer_key_time = now

    def current_command(self, now: Optional[float] = None) -> Twist:
        """Build the current command, stopping when drive-key input expires."""
        if now is None:
            now = time.monotonic()

        if now - self.last_drive_key_time > self.key_timeout:
            self.direction = 0

        message = Twist()
        message.linear.x = self.direction * self.speed
        if self.direction:
            steering_rad = math.radians(self.steering_deg)
            message.angular.z = (
                message.linear.x / self.wheelbase * math.tan(steering_rad)
            )
        return message

    def publish_command(self) -> None:
        self.publisher.publish(self.current_command())

    def publish_stop(self) -> None:
        self.direction = 0
        self.publisher.publish(Twist())


def _run_terminal(screen: "curses.window", node: ManualCmdVel) -> None:
    curses.curs_set(0)
    screen.keypad(True)
    screen.nodelay(True)

    while rclpy.ok() and not node.quit_requested:
        screen.erase()
        screen.addstr(0, 0, "Arrow-key manual driving (/cmd_vel)")
        screen.addstr(2, 0, "UP/DOWN : forward/reverse (hold)")
        screen.addstr(3, 0, "LEFT/RIGHT : steering +/-")
        screen.addstr(4, 0, "SPACE or S : stop")
        screen.addstr(5, 0, "C : center steering")
        screen.addstr(6, 0, "Q : stop and quit")
        screen.addstr(
            8,
            0,
            f"speed={node.direction * node.speed:+.2f} m/s  "
            f"steering={node.steering_deg:+.1f} deg",
        )
        screen.refresh()

        while True:
            key = screen.getch()
            if key == -1:
                break
            node.handle_key(key)

        rclpy.spin_once(node, timeout_sec=0.02)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = ManualCmdVel()
    try:
        curses.wrapper(_run_terminal, node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send several zero commands so the bridge receives a stop before shutdown.
        for _ in range(3):
            node.publish_stop()
            rclpy.spin_once(node, timeout_sec=0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
