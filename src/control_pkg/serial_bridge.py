#!/usr/bin/env python3
"""Bridge final ROS control commands to the Dolsoi Arduino protocol."""

import argparse
import math
import threading
import time
from typing import Optional, Sequence, Tuple

import rclpy
import serial
from rclpy.node import Node
from std_msgs.msg import Float32

from serial_protocol import (
    ULTRASONIC_FRAME,
    VEHICLE_FRAME,
    parse_telemetry_line,
)


class SerialBridge(Node):
    """Forward only fresh steer/throttle pairs and reconnect after I/O loss."""

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baudrate: Optional[int] = None,
    ) -> None:
        super().__init__("serial_bridge")

        self.declare_parameter("steer_command_topic", "/auto_steer_angle")
        self.declare_parameter("throttle_command_topic", "/auto_throttle")
        self.declare_parameter("serial_port", serial_port or "/dev/ttyACM0")
        self.declare_parameter("baud_rate", 115200 if baudrate is None else baudrate)
        self.declare_parameter("serial_startup_delay", 2.0)
        self.declare_parameter("reconnect_interval_sec", 1.0)
        self.declare_parameter("reconnect_log_interval_sec", 5.0)
        self.declare_parameter("control_output_rate_hz", 20.0)
        self.declare_parameter("max_drive_pwm", 255)
        self.declare_parameter("max_steer_angle", 20.0)
        self.declare_parameter("throttle_deadband", 0.02)
        self.declare_parameter("max_throttle", 1.0)
        self.declare_parameter("command_timeout_sec", 0.3)
        self.declare_parameter("center_on_timeout", False)
        self.declare_parameter("invert_steer", False)
        self.declare_parameter(
            "steering_angle_topic", "/vehicle/angle"
        )
        self.declare_parameter("drive_pwm_topic", "/vehicle/drive_pwm")
        self.declare_parameter(
            "left_front_distance_topic", "/sonic/left/front"
        )
        self.declare_parameter(
            "left_rear_distance_topic", "/sonic/left/rear"
        )
        self.declare_parameter(
            "right_front_distance_topic", "/sonic/right/front"
        )
        self.declare_parameter(
            "right_rear_distance_topic", "/sonic/right/rear"
        )

        steer_command_topic = str(
            self.get_parameter("steer_command_topic").value
        )
        throttle_command_topic = str(
            self.get_parameter("throttle_command_topic").value
        )
        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baud_rate = int(self.get_parameter("baud_rate").value)
        self.serial_startup_delay = max(
            0.0, float(self.get_parameter("serial_startup_delay").value)
        )
        reconnect_interval = max(
            0.1, float(self.get_parameter("reconnect_interval_sec").value)
        )
        self.reconnect_log_interval = max(
            0.1, float(self.get_parameter("reconnect_log_interval_sec").value)
        )
        control_output_rate = max(
            1.0, float(self.get_parameter("control_output_rate_hz").value)
        )
        self.max_drive_pwm = max(
            0, min(255, int(self.get_parameter("max_drive_pwm").value))
        )
        self.max_steer_angle = abs(
            float(self.get_parameter("max_steer_angle").value)
        )
        self.throttle_deadband = max(
            0.0, float(self.get_parameter("throttle_deadband").value)
        )
        self.max_throttle = min(
            1.0, abs(float(self.get_parameter("max_throttle").value))
        )
        self.command_timeout_sec = max(
            0.0, float(self.get_parameter("command_timeout_sec").value)
        )
        self.center_on_timeout = bool(self.get_parameter("center_on_timeout").value)
        self.invert_steer = bool(self.get_parameter("invert_steer").value)
        steering_angle_topic = str(
            self.get_parameter("steering_angle_topic").value
        )
        drive_pwm_topic = str(self.get_parameter("drive_pwm_topic").value)
        left_front_distance_topic = str(
            self.get_parameter("left_front_distance_topic").value
        )
        left_rear_distance_topic = str(
            self.get_parameter("left_rear_distance_topic").value
        )
        right_front_distance_topic = str(
            self.get_parameter("right_front_distance_topic").value
        )
        right_rear_distance_topic = str(
            self.get_parameter("right_rear_distance_topic").value
        )

        self.serial = None
        self.serial_lock = threading.RLock()
        self.shutting_down = False
        self.serial_rx_buffer = bytearray()
        self.last_drive_command = None
        self.last_steer_command = None
        self.last_reconnect_log_monotonic = -math.inf

        self.received_steer = False
        self.received_throttle = False
        self.last_steer_time = self.get_clock().now()
        self.last_throttle_time = self.get_clock().now()
        self.desired_steer = 0.0
        self.desired_throttle = 0.0
        self.control_was_valid = False
        self.stale_stop_sent = False

        self.steer_subscription = self.create_subscription(
            Float32, steer_command_topic, self.on_steer_command, 10
        )
        self.throttle_subscription = self.create_subscription(
            Float32, throttle_command_topic, self.on_throttle_command, 10
        )
        self.steering_angle_pub = self.create_publisher(
            Float32, steering_angle_topic, 10
        )
        self.drive_pwm_pub = self.create_publisher(
            Float32, drive_pwm_topic, 10
        )
        self.left_front_distance_pub = self.create_publisher(
            Float32, left_front_distance_topic, 10
        )
        self.left_rear_distance_pub = self.create_publisher(
            Float32, left_rear_distance_topic, 10
        )
        self.right_front_distance_pub = self.create_publisher(
            Float32, right_front_distance_topic, 10
        )
        self.right_rear_distance_pub = self.create_publisher(
            Float32, right_rear_distance_topic, 10
        )

        self.reconnect_timer = self.create_timer(
            reconnect_interval, self.try_connect
        )
        self.control_timer = self.create_timer(
            1.0 / control_output_rate, self.publish_fresh_control
        )
        self.serial_read_timer = self.create_timer(
            0.005, self.read_serial_telemetry
        )

        self.get_logger().info(
            f"Final control: {steer_command_topic}, {throttle_command_topic}; "
            f"serial={self.serial_port}@{self.baud_rate}, "
            f"pair timeout={self.command_timeout_sec:.2f}s, "
            f"heartbeat={control_output_rate:.1f}Hz"
        )
        self.get_logger().info(
            "Publishing Arduino telemetry: "
            f"{steering_angle_topic}, {drive_pwm_topic} (commanded -255..255, "
            "not measured speed), "
            f"{left_front_distance_topic}, {left_rear_distance_topic}, "
            f"{right_front_distance_topic}, {right_rear_distance_topic}"
        )

        # A missing device must not abort node construction.
        self.try_connect()

    def try_connect(self) -> None:
        if self.shutting_down:
            return
        with self.serial_lock:
            if self.serial is not None and self.serial.is_open:
                return

        now = time.monotonic()
        should_log = (
            now - self.last_reconnect_log_monotonic
            >= self.reconnect_log_interval
        )
        if should_log:
            self.get_logger().info(
                f"Opening serial {self.serial_port}@{self.baud_rate}"
            )
            self.last_reconnect_log_monotonic = now

        try:
            connection = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=0.05,
                write_timeout=0.2,
            )
            time.sleep(self.serial_startup_delay)
            if hasattr(connection, "reset_input_buffer"):
                connection.reset_input_buffer()
            with self.serial_lock:
                if self.shutting_down:
                    connection.close()
                    return
                self.serial = connection
                self.serial_rx_buffer.clear()
                self.last_drive_command = None
                self.last_steer_command = None
            # Reconnection never replays an old moving command.
            self.send_drive("S", 0, force=True)
            self.get_logger().info("Serial connected; safe stop sent")
        except (serial.SerialException, OSError) as exc:
            try:
                connection.close()
            except (UnboundLocalError, serial.SerialException, OSError):
                pass
            if should_log:
                self.get_logger().warning(
                    f"Serial open failed; retrying: {exc}"
                )

    def _disconnect(self, operation: str, exc: BaseException) -> None:
        with self.serial_lock:
            connection = self.serial
            self.serial = None
            self.serial_rx_buffer.clear()
            self.last_drive_command = None
            self.last_steer_command = None
        if connection is not None:
            try:
                connection.close()
            except (serial.SerialException, OSError):
                pass
        self.control_was_valid = False
        self.stale_stop_sent = False
        self.get_logger().warning(
            f"Serial {operation} failed; disconnected and will retry: {exc}",
            throttle_duration_sec=self.reconnect_log_interval,
        )

    def on_steer_command(self, msg: Float32) -> None:
        angle = float(msg.data) if math.isfinite(msg.data) else 0.0
        if self.invert_steer:
            angle = -angle
        self.desired_steer = max(
            -self.max_steer_angle, min(self.max_steer_angle, angle)
        )
        self.last_steer_time = self.get_clock().now()
        self.received_steer = True

    def on_throttle_command(self, msg: Float32) -> None:
        throttle = float(msg.data) if math.isfinite(msg.data) else 0.0
        self.desired_throttle = max(
            -self.max_throttle, min(self.max_throttle, throttle)
        )
        self.last_throttle_time = self.get_clock().now()
        self.received_throttle = True

    def _control_is_fresh(self, now) -> Tuple[bool, str]:
        if not self.received_steer or not self.received_throttle:
            return False, "waiting for both steer and throttle"
        if self.command_timeout_sec <= 0.0:
            return True, ""

        steer_age = (now - self.last_steer_time).nanoseconds / 1e9
        throttle_age = (now - self.last_throttle_time).nanoseconds / 1e9
        stale = []
        if steer_age >= self.command_timeout_sec:
            stale.append(f"steer={steer_age:.2f}s")
        if throttle_age >= self.command_timeout_sec:
            stale.append(f"throttle={throttle_age:.2f}s")
        return not stale, ", ".join(stale)

    def publish_fresh_control(self) -> None:
        fresh, stale_reason = self._control_is_fresh(self.get_clock().now())
        if not fresh:
            if not self.stale_stop_sent:
                self.send_drive("S", 0, force=True)
                if self.control_was_valid:
                    self.get_logger().warning(
                        f"Control pair stale ({stale_reason}); drive stopped"
                    )
                if self.center_on_timeout and self.received_steer:
                    self.send_steer(0.0, force=True)
                self.stale_stop_sent = True
            self.control_was_valid = False
            return

        direction, pwm = self.convert_throttle(self.desired_throttle)
        # Drive is intentionally forced every cycle: it is the Arduino
        # watchdog heartbeat even when the command value has not changed.
        steer_sent = self.send_steer(self.desired_steer)
        drive_sent = self.send_drive(direction, pwm, force=True)
        if not (steer_sent and drive_sent):
            self.control_was_valid = False
            return
        if not self.control_was_valid:
            self.get_logger().info("Fresh steer/throttle pair; control enabled")
        self.control_was_valid = True
        self.stale_stop_sent = False

    def convert_throttle(self, throttle: float) -> Tuple[str, int]:
        if abs(throttle) <= self.throttle_deadband:
            return "S", 0
        pwm = int(round(abs(throttle) * self.max_drive_pwm))
        return ("F" if throttle > 0.0 else "R"), pwm

    def read_serial_telemetry(self) -> None:
        with self.serial_lock:
            connection = self.serial
            if connection is None:
                return
            try:
                waiting = connection.in_waiting
                data = connection.read(waiting) if waiting > 0 else b""
            except (serial.SerialException, OSError) as exc:
                self._disconnect("read", exc)
                return

        if not data:
            return
        self.serial_rx_buffer.extend(data)
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
            frame_type, values = parse_telemetry_line(raw_line)
        except ValueError as exc:
            self.get_logger().warning(
                f"Ignoring invalid Arduino telemetry {raw_line!r}: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        if frame_type == VEHICLE_FRAME:
            drive_pwm, steering_angle = values
            self.drive_pwm_pub.publish(Float32(data=drive_pwm))
            self.steering_angle_pub.publish(Float32(data=steering_angle))
            return

        if frame_type == ULTRASONIC_FRAME:
            side, position, distance = values
            ultrasonic_publishers = {
                ("L", "F"): self.left_front_distance_pub,
                ("L", "R"): self.left_rear_distance_pub,
                ("R", "F"): self.right_front_distance_pub,
                ("R", "R"): self.right_rear_distance_pub,
            }
            ultrasonic_publishers[(side, position)].publish(
                Float32(data=distance)
            )

    def _write_command(self, command: str, force: bool, kind: str) -> bool:
        with self.serial_lock:
            connection = self.serial
            if connection is None:
                return False
            previous = (
                self.last_steer_command
                if kind == "steer"
                else self.last_drive_command
            )
            if not force and command == previous:
                return True
            try:
                connection.write(command.encode("ascii"))
                connection.flush()
            except (serial.SerialException, OSError) as exc:
                self._disconnect("write", exc)
                return False
            if kind == "steer":
                self.last_steer_command = command
            else:
                self.last_drive_command = command
            return True

    def send_steer(self, angle: float, force: bool = False) -> bool:
        return self._write_command(
            f"S,{angle:.1f}\n", force=force, kind="steer"
        )

    def send_drive(
        self, direction: str, speed: int, force: bool = False
    ) -> bool:
        return self._write_command(
            f"D,{direction},{speed}\n", force=force, kind="drive"
        )

    def destroy_node(self) -> bool:
        self.shutting_down = True
        with self.serial_lock:
            connection = self.serial
        if connection is not None:
            self.send_drive("S", 0, force=True)
        with self.serial_lock:
            self.serial = None
        if connection is not None:
            try:
                connection.close()
            except (serial.SerialException, OSError):
                pass
        return super().destroy_node()


def parse_args(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[argparse.Namespace, list]:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge final steer angle / normalized throttle Float32 commands "
            "to Arduino serial commands."
        )
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
