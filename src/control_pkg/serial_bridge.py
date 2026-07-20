#!/usr/bin/env python3
import argparse
import math
import time
from typing import Optional, Sequence, Tuple

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


class SerialBridge(Node):
    """Exchange cmd_vel commands and vehicle telemetry with the Arduino."""

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baudrate: Optional[int] = None,
    ) -> None:
        super().__init__("serial_bridge")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("serial_port", serial_port or "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200 if baudrate is None else baudrate)
        self.declare_parameter("serial_startup_delay", 2.0)
        self.declare_parameter("max_linear_speed", 1.0)
        self.declare_parameter("wheelbase", 0.545)
        self.declare_parameter("max_drive_pwm", 255)
        self.declare_parameter("max_steer_angle", 20.0)
        self.declare_parameter("linear_deadband", 0.01)
        self.declare_parameter("angular_deadband", 0.01)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("center_on_timeout", True)
        self.declare_parameter("invert_angular", False)
        self.declare_parameter(
            "steering_angle_topic", "/vehicle/current_steering_angle"
        )
        self.declare_parameter("speed_topic", "/vehicle/current_speed")
        self.declare_parameter(
            "left_distance_topic", "/ultrasonic/left_distance"
        )
        self.declare_parameter(
            "right_distance_topic", "/ultrasonic/right_distance"
        )

        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        serial_port = str(self.get_parameter("serial_port").value)
        baud_rate = int(self.get_parameter("baud_rate").value)
        startup_delay = max(
            0.0, float(self.get_parameter("serial_startup_delay").value)
        )
        self.max_linear_speed = max(
            1e-6, abs(float(self.get_parameter("max_linear_speed").value))
        )
        self.wheelbase = max(1e-6, abs(float(self.get_parameter("wheelbase").value)))
        self.max_drive_pwm = max(
            0, min(255, int(self.get_parameter("max_drive_pwm").value))
        )
        self.max_steer_angle = abs(
            float(self.get_parameter("max_steer_angle").value)
        )
        self.linear_deadband = max(
            0.0, float(self.get_parameter("linear_deadband").value)
        )
        self.angular_deadband = max(
            0.0, float(self.get_parameter("angular_deadband").value)
        )
        self.command_timeout = max(
            0.0, float(self.get_parameter("command_timeout").value)
        )
        self.center_on_timeout = bool(self.get_parameter("center_on_timeout").value)
        self.invert_angular = bool(self.get_parameter("invert_angular").value)
        steering_angle_topic = str(
            self.get_parameter("steering_angle_topic").value
        )
        speed_topic = str(self.get_parameter("speed_topic").value)
        left_distance_topic = str(self.get_parameter("left_distance_topic").value)
        right_distance_topic = str(
            self.get_parameter("right_distance_topic").value
        )

        self.serial = serial.Serial(serial_port, baud_rate, timeout=0.1)
        time.sleep(startup_delay)
        if hasattr(self.serial, "reset_input_buffer"):
            self.serial.reset_input_buffer()
        self.serial_rx_buffer = bytearray()
        self.last_drive_command = None
        self.last_steer_command = None
        self.received_cmd_vel = False
        self.timed_out = False
        self.last_cmd_time = self.get_clock().now()

        self.send_steer(0.0)
        self.send_drive("S", 0)

        self.subscription = self.create_subscription(
            Twist, cmd_vel_topic, self.on_cmd_vel, 10
        )
        self.steering_angle_pub = self.create_publisher(
            Float32, steering_angle_topic, 10
        )
        self.speed_pub = self.create_publisher(Float32, speed_topic, 10)
        self.left_distance_pub = self.create_publisher(
            Float32, left_distance_topic, 10
        )
        self.right_distance_pub = self.create_publisher(
            Float32, right_distance_topic, 10
        )
        self.timeout_timer = self.create_timer(0.1, self.check_command_timeout)
        self.serial_read_timer = self.create_timer(0.005, self.read_serial_telemetry)

        self.get_logger().info(
            f"Listening on {cmd_vel_topic}; serial={serial_port}, "
            f"linear +/-{self.max_linear_speed:.2f} m/s -> PWM {self.max_drive_pwm}, "
            f"Ackermann wheelbase={self.wheelbase:.3f}m, "
            f"max steer +/-{self.max_steer_angle:.1f} deg"
        )
        self.get_logger().info(
            "Publishing Arduino telemetry: "
            f"{steering_angle_topic}, {speed_topic}, "
            f"{left_distance_topic}, {right_distance_topic}"
        )

    def on_cmd_vel(self, msg: Twist) -> None:
        linear_x = msg.linear.x if math.isfinite(msg.linear.x) else 0.0
        angular_z = msg.angular.z if math.isfinite(msg.angular.z) else 0.0

        direction, pwm = self.convert_drive(linear_x)
        steer_angle = self.convert_steer(linear_x, angular_z)

        self.send_steer(steer_angle)
        self.send_drive(direction, pwm)
        self.last_cmd_time = self.get_clock().now()
        self.received_cmd_vel = True
        self.timed_out = False

    def convert_drive(self, linear_x: float) -> Tuple[str, int]:
        if abs(linear_x) <= self.linear_deadband:
            return "S", 0

        ratio = min(1.0, abs(linear_x) / self.max_linear_speed)
        pwm = int(round(ratio * self.max_drive_pwm))
        return ("F" if linear_x > 0.0 else "R"), pwm

    def convert_steer(self, linear_x: float, angular_z: float) -> float:
        if (
            abs(linear_x) <= self.linear_deadband
            or abs(angular_z) <= self.angular_deadband
        ):
            return 0.0

        angle = math.degrees(math.atan(self.wheelbase * angular_z / linear_x))
        if self.invert_angular:
            angle = -angle
        return max(-self.max_steer_angle, min(self.max_steer_angle, angle))

    def check_command_timeout(self) -> None:
        if not self.received_cmd_vel or self.timed_out or self.command_timeout <= 0.0:
            return

        elapsed = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if elapsed < self.command_timeout:
            return

        self.send_drive("S", 0)
        if self.center_on_timeout:
            self.send_steer(0.0)
        self.timed_out = True
        self.get_logger().warning(
            f"No cmd_vel for {elapsed:.2f}s: vehicle stopped"
        )

    def read_serial_telemetry(self) -> None:
        waiting = self.serial.in_waiting
        if waiting <= 0:
            return

        self.serial_rx_buffer.extend(self.serial.read(waiting))
        while True:
            newline_index = self.serial_rx_buffer.find(b"\n")
            if newline_index < 0:
                break

            raw_line = bytes(self.serial_rx_buffer[:newline_index])
            del self.serial_rx_buffer[: newline_index + 1]
            self.publish_telemetry_line(raw_line)

        if len(self.serial_rx_buffer) > 1024:
            self.serial_rx_buffer.clear()
            self.get_logger().warning("Discarded oversized Arduino serial input")

    def publish_telemetry_line(self, raw_line: bytes) -> None:
        try:
            fields = raw_line.decode("ascii").strip().split(",")
            if len(fields) != 4:
                raise ValueError("expected four comma-separated fields")

            speed, steering_angle, left_distance, right_distance = [
                float(field) for field in fields
            ]
            if not all(
                math.isfinite(value)
                for value in (
                    speed,
                    steering_angle,
                    left_distance,
                    right_distance,
                )
            ):
                raise ValueError("telemetry contains a non-finite value")
        except (UnicodeDecodeError, ValueError) as exc:
            self.get_logger().warning(
                f"Ignoring invalid Arduino telemetry {raw_line!r}: {exc}"
            )
            return

        self.steering_angle_pub.publish(Float32(data=steering_angle))
        self.speed_pub.publish(Float32(data=speed))
        self.left_distance_pub.publish(Float32(data=left_distance))
        self.right_distance_pub.publish(Float32(data=right_distance))

    def send_steer(self, angle: float, force: bool = False) -> None:
        command = f"S,{angle:.1f}\n"
        if force or command != self.last_steer_command:
            self.serial.write(command.encode("ascii"))
            self.serial.flush()
            self.last_steer_command = command

    def send_drive(self, direction: str, speed: int, force: bool = False) -> None:
        command = f"D,{direction},{speed}\n"
        if force or command != self.last_drive_command:
            self.serial.write(command.encode("ascii"))
            self.serial.flush()
            self.last_drive_command = command

    def destroy_node(self) -> bool:
        if hasattr(self, "serial") and self.serial.is_open:
            self.send_drive("S", 0, force=True)
            self.send_steer(0.0, force=True)
            self.serial.close()
        return super().destroy_node()


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description="Bridge ROS 2 cmd_vel messages to Arduino serial commands."
    )
    parser.add_argument(
        "--serial-port",
        default=None,
        help="Arduino serial device, for example /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--baudrate",
        "--baud-rate",
        dest="baudrate",
        type=int,
        default=None,
        help="Arduino serial baudrate, for example 115200.",
    )
    return parser.parse_known_args(argv)


def main(args: Optional[Sequence[str]] = None) -> None:
    cli_args, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = SerialBridge(
            serial_port=cli_args.serial_port,
            baudrate=cli_args.baudrate,
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
