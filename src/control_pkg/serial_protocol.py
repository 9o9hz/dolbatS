"""Parsing helpers for telemetry emitted by the Dolsoi Arduino."""

import math
from typing import Tuple, Union


VEHICLE_FRAME = "VEH"
ULTRASONIC_FRAME = "ULT"
TelemetryValues = Union[
    Tuple[float, float],
    Tuple[str, str, float],
]


def parse_telemetry_line(raw_line: bytes) -> Tuple[str, TelemetryValues]:
    """Parse one tagged Arduino telemetry line.

    Supported frames are ``VEH,drive_pwm,steering_deg`` and
    ``ULT,side,position,distance_cm``.
    """
    try:
        fields = raw_line.decode("ascii").strip().split(",")
    except UnicodeDecodeError as exc:
        raise ValueError("telemetry is not ASCII") from exc

    frame_type = fields[0] if fields else ""
    expected_fields = {
        VEHICLE_FRAME: 3,
        ULTRASONIC_FRAME: 4,
    }.get(frame_type)
    if expected_fields is None:
        raise ValueError(f"unknown telemetry frame type {frame_type!r}")
    if len(fields) != expected_fields:
        raise ValueError(
            f"{frame_type} telemetry expected {expected_fields} fields"
        )

    if frame_type == VEHICLE_FRAME:
        try:
            drive_pwm, steering_angle = (
                float(field) for field in fields[1:]
            )
        except ValueError as exc:
            raise ValueError("telemetry contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in (drive_pwm, steering_angle)):
            raise ValueError("telemetry contains a non-finite value")
        if not -255.0 <= drive_pwm <= 255.0:
            raise ValueError("drive PWM is outside -255..255")
        return frame_type, (drive_pwm, steering_angle)

    side, position = fields[1:3]
    if side not in ("L", "R"):
        raise ValueError("ultrasonic side must be L or R")
    if position not in ("F", "R"):
        raise ValueError("ultrasonic position must be F or R")
    try:
        distance = float(fields[3])
    except ValueError as exc:
        raise ValueError("telemetry contains a non-numeric value") from exc
    if not math.isfinite(distance):
        raise ValueError("telemetry contains a non-finite value")
    return frame_type, (side, position, distance)
