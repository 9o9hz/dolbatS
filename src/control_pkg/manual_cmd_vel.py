#!/usr/bin/env python3
"""Keyboard teleoperation node for final steer/throttle Float32 topics."""

import curses
import time
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class ManualCmdVel(Node):
    """Convert arrow-key input into final normalized vehicle commands."""

    def __init__(self) -> None:
        super().__init__("manual_cmd_vel")

        self.declare_parameter("steer_topic", "/auto_steer_angle")
        self.declare_parameter("throttle_topic", "/auto_throttle")
        self.declare_parameter("throttle", 0.9)
        self.declare_parameter("steer_step_deg", 10.0)
        self.declare_parameter("max_steer_deg", 20.0)
        self.declare_parameter("key_timeout", 0.25)
        self.declare_parameter("publish_rate", 20.0)

        steer_topic = str(self.get_parameter("steer_topic").value)
        throttle_topic = str(self.get_parameter("throttle_topic").value)
        self.throttle = min(
            1.0, abs(float(self.get_parameter("throttle").value))
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

        self.steer_publisher = self.create_publisher(
            Float32, steer_topic, 10
        )
        self.throttle_publisher = self.create_publisher(
            Float32, throttle_topic, 10
        )
        self.steering_deg = 0.0
        self.direction = 0
        self.last_drive_key_time = 0.0
        self.last_steer_key_time = 0.0
        self.quit_requested = False
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_command)

        self.get_logger().info(
            f"Manual driving ready: publishing {steer_topic}, "
            f"{throttle_topic} at {publish_rate:.1f} Hz"
        )
        self.get_logger().warning(
            "Do not run mission_manager while manual driving; both publish "
            "the final /auto_* command topics"
        )

    def handle_key(self, key: int) -> None:
        """Update commands from a terminal key without privileged input access."""
        now = time.monotonic()

        if key in (ord("q"), ord("Q")):
            self.quit_requested = True
        elif key == curses.KEY_LEFT:
            if now - self.last_steer_key_time > 0.08:
                self._change_steering(self.steer_step_deg)
                self.last_steer_key_time = now
        elif key == curses.KEY_RIGHT:
            if now - self.last_steer_key_time > 0.08:
                self._change_steering(-self.steer_step_deg)
                self.last_steer_key_time = now
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
            self.publish_steering()

    def _change_steering(self, change_deg: float) -> None:
        self.steering_deg = max(
            -self.max_steer_deg,
            min(self.max_steer_deg, self.steering_deg + change_deg),
        )
        self.publish_steering()

    def publish_steering(self) -> None:
        self.steer_publisher.publish(Float32(data=self.steering_deg))

    def current_command(
        self, now: Optional[float] = None
    ) -> tuple[float, float]:
        """Return steer and throttle, stopping when drive input expires."""
        if now is None:
            now = time.monotonic()

        if now - self.last_drive_key_time > self.key_timeout:
            self.direction = 0

        return self.steering_deg, self.direction * self.throttle

    def publish_command(self) -> None:
        steering_deg, throttle = self.current_command()
        self.steer_publisher.publish(Float32(data=steering_deg))
        self.throttle_publisher.publish(Float32(data=throttle))

    def publish_stop(self) -> None:
        self.direction = 0
        self.steer_publisher.publish(Float32(data=self.steering_deg))
        self.throttle_publisher.publish(Float32(data=0.0))


def _run_terminal(screen: "curses.window", node: ManualCmdVel) -> None:
    curses.curs_set(0)
    screen.keypad(True)
    screen.nodelay(True)

    while rclpy.ok() and not node.quit_requested:
        screen.erase()
        screen.addstr(0, 0, "Arrow-key manual driving (/auto_*)")
        screen.addstr(2, 0, "UP/DOWN : forward/reverse (hold)")
        screen.addstr(3, 0, "LEFT/RIGHT : steering +/-")
        screen.addstr(4, 0, "SPACE or S : stop")
        screen.addstr(5, 0, "C : center steering")
        screen.addstr(6, 0, "Q : stop and quit")
        screen.addstr(
            8,
            0,
            f"throttle={node.direction * node.throttle:+.2f}  "
            f"steering={node.steering_deg:+.1f} deg",
        )
        screen.refresh()

        while True:
            key = screen.getch()
            if key == -1:
                break
            node.handle_key(key)

        rclpy.spin_once(node, timeout_sec=0.01)


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
