#!/usr/bin/env python3
"""Select steering candidates and publish final vehicle commands."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Optional, Sequence

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


DEFAULTS = {
    "publish_rate_hz": 30.0,
    "log_rate_hz": 1.0,
    "lane_steer_topic": "/control/candidate/lane/steer_angle",
    "lane_valid_topic": "/control/candidate/lane/valid",
    "obstacle_steer_topic": "/control/candidate/obstacle/steer_angle",
    "obstacle_valid_topic": "/control/candidate/obstacle/valid",
    "obstacle_active_topic": "/detect/obstacle/avoidance_active",
    "traffic_detected_topic": "/detect/traffic_light/detected",
    "traffic_color_topic": "/detect/traffic_light/color",
    "traffic_confidence_topic": "/detect/traffic_light/confidence",
    "final_steer_topic": "/auto_steer_angle",
    "final_throttle_topic": "/auto_throttle",
    "mission_state_topic": "/mission_state",
    "status_topic": "/mission_manager/status",
    "auto_steer_angle_abs_max": 23.0,
    "auto_throttle_max": 1.0,
    "throttle_curve_k": 0.0,
    "lane_throttle_min": 0.3,
    "lane_throttle_max": 0.6,
    "obstacle_throttle_min": 0.4,
    "obstacle_throttle_max": 0.8,
    "yellow_deceleration_sec": 3.0,
    "green_forward_duration_sec": 3.0,
    "green_forward_throttle": 0.4,
    "green_forward_steer_deg": 0.0,
}

MISSION_LANE = "lane"
MISSION_TRAFFIC = "traffic_light"
MISSION_OBSTACLE = "obstacle"

TRAFFIC_IDLE = "idle"
TRAFFIC_RED_HOLD = "red_hold"
TRAFFIC_YELLOW_DECELERATING = "yellow_decelerating"
TRAFFIC_YELLOW_HOLD = "yellow_hold"
TRAFFIC_GREEN_FORWARD = "green_forward"
TRAFFIC_GREEN_WAIT_CLEAR = "green_completed_wait_clear"

VALID_TRAFFIC_COLORS = {"red", "yellow", "green"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def map_throttle_by_steer(
    steer_deg: float,
    steer_abs_max: float,
    throttle_min: float,
    throttle_max: float,
    curve_k: float,
) -> float:
    """Map small absolute steering to high throttle and vice versa."""
    if throttle_max < throttle_min:
        throttle_min, throttle_max = throttle_max, throttle_min
    ratio = clamp(
        abs(float(steer_deg)) / max(float(steer_abs_max), 1e-6),
        0.0,
        1.0,
    )
    k = max(0.0, float(curve_k))
    if k < 1e-6:
        shaped = ratio
    else:
        shaped = math.expm1(-k * ratio) / math.expm1(-k)
    return throttle_max - shaped * (throttle_max - throttle_min)


@dataclass
class CandidateMemory:
    latest_steer: float = 0.0
    steer_received: bool = False
    current_valid: bool = False
    last_valid_steer: float = 0.0
    has_last_valid: bool = False

    def update_steer(self, steer: float) -> None:
        if not math.isfinite(steer):
            return
        self.latest_steer = float(steer)
        self.steer_received = True
        self._remember_if_valid()

    def update_valid(self, valid: bool) -> None:
        self.current_valid = bool(valid)
        self._remember_if_valid()

    def _remember_if_valid(self) -> None:
        if self.current_valid and self.steer_received:
            self.last_valid_steer = self.latest_steer
            self.has_last_valid = True


@dataclass(frozen=True)
class MissionOutput:
    mission_state: str
    traffic_substate: str
    selected_steer_source: str
    steer_deg: float
    throttle: float


class MissionLogic:
    """Deterministic state and output logic with caller-supplied time."""

    def __init__(
        self,
        *,
        auto_steer_angle_abs_max: float = 23.0,
        auto_throttle_max: float = 1.0,
        throttle_curve_k: float = 0.0,
        lane_throttle_min: float = 0.3,
        lane_throttle_max: float = 0.6,
        obstacle_throttle_min: float = 0.4,
        obstacle_throttle_max: float = 0.8,
        yellow_deceleration_sec: float = 3.0,
        green_forward_duration_sec: float = 3.0,
        green_forward_throttle: float = 0.4,
        green_forward_steer_deg: float = 0.0,
    ) -> None:
        self.auto_steer_angle_abs_max = auto_steer_angle_abs_max
        self.auto_throttle_max = auto_throttle_max
        self.throttle_curve_k = throttle_curve_k
        self.lane_throttle_min = lane_throttle_min
        self.lane_throttle_max = lane_throttle_max
        self.obstacle_throttle_min = obstacle_throttle_min
        self.obstacle_throttle_max = obstacle_throttle_max
        self.yellow_deceleration_sec = yellow_deceleration_sec
        self.green_forward_duration_sec = green_forward_duration_sec
        self.green_forward_throttle = green_forward_throttle
        self.green_forward_steer_deg = green_forward_steer_deg
                                                                                                                                                                     
        self.lane = CandidateMemory()
        self.obstacle = CandidateMemory()
        self.obstacle_active = False
        self.traffic_detected = False
        self.traffic_color = "none"
        self.traffic_confidence = 0.0
        self.traffic_substate = TRAFFIC_IDLE
        self.traffic_green_completed = False
        self.yellow_start_time = 0.0
        self.yellow_start_throttle = 0.0
        self.green_start_time = 0.0
        self.current_output_throttle = 0.0

    def _traffic_is_valid(self) -> bool:
        return (
            self.traffic_detected
            and self.traffic_color in VALID_TRAFFIC_COLORS
        )

    def _start_yellow(self, now_sec: float) -> None:
        self.traffic_substate = TRAFFIC_YELLOW_DECELERATING
        self.yellow_start_time = now_sec
        self.yellow_start_throttle = self.current_output_throttle

    def _start_green(self, now_sec: float) -> None:
        self.traffic_substate = TRAFFIC_GREEN_FORWARD
        self.green_start_time = now_sec

    def _update_traffic_state(self, now_sec: float) -> None:
        if self.traffic_green_completed:
            if not self.traffic_detected:
                self.traffic_green_completed = False
                self.traffic_substate = TRAFFIC_IDLE
            else:
                self.traffic_substate = TRAFFIC_GREEN_WAIT_CLEAR
            return

        valid = self._traffic_is_valid()
        color = self.traffic_color if valid else "none"

        if self.traffic_substate == TRAFFIC_IDLE:
            if color == "red":
                self.traffic_substate = TRAFFIC_RED_HOLD
            elif color == "yellow":
                self._start_yellow(now_sec)
            elif color == "green":
                self._start_green(now_sec)
            return

        if self.traffic_substate == TRAFFIC_RED_HOLD:
            if color == "green":
                self._start_green(now_sec)
            return

        if self.traffic_substate == TRAFFIC_YELLOW_DECELERATING:
            if color == "red":
                self.traffic_substate = TRAFFIC_RED_HOLD
                return
            elapsed = now_sec - self.yellow_start_time
            if elapsed >= self.yellow_deceleration_sec:
                self.traffic_substate = TRAFFIC_YELLOW_HOLD
                if color == "green":
                    self._start_green(now_sec)
            return

        if self.traffic_substate == TRAFFIC_YELLOW_HOLD:
            if color == "red":
                self.traffic_substate = TRAFFIC_RED_HOLD
            elif color == "green":
                self._start_green(now_sec)
            return

        if self.traffic_substate == TRAFFIC_GREEN_FORWARD:
            if color == "red":
                self.traffic_substate = TRAFFIC_RED_HOLD
                return
            elapsed = now_sec - self.green_start_time
            if elapsed >= self.green_forward_duration_sec:
                self.traffic_green_completed = True
                self.traffic_substate = TRAFFIC_GREEN_WAIT_CLEAR

    def _lane_output(self) -> MissionOutput:
        if not self.lane.has_last_valid:
            return MissionOutput(
                MISSION_LANE, self.traffic_substate, "none", 0.0, 0.0
            )
        steer = self.lane.last_valid_steer
        throttle = map_throttle_by_steer(
            steer,
            self.auto_steer_angle_abs_max,
            self.lane_throttle_min,
            self.lane_throttle_max,
            self.throttle_curve_k,
        )
        return MissionOutput(
            MISSION_LANE, self.traffic_substate, "lane", steer, throttle
        )

    def _obstacle_output(self) -> MissionOutput:
        # Unlike lane tracking, an active avoidance candidate must be valid
        # now. Holding a stale full-steering obstacle command after a sensor
        # fault or controller timeout would be unsafe.
        if (
            not self.obstacle.current_valid
            or not self.obstacle.has_last_valid
        ):
            return MissionOutput(
                MISSION_OBSTACLE,
                self.traffic_substate,
                "none",
                0.0,
                0.0,
            )
        steer = self.obstacle.last_valid_steer
        throttle = map_throttle_by_steer(
            steer,
            self.auto_steer_angle_abs_max,
            self.obstacle_throttle_min,
            self.obstacle_throttle_max,
            self.throttle_curve_k,
        )
        return MissionOutput(
            MISSION_OBSTACLE,
            self.traffic_substate,
            "obstacle",
            steer,
            throttle,
        )

    def _traffic_output(self, now_sec: float) -> MissionOutput:
        lane_steer = (
            self.lane.last_valid_steer if self.lane.has_last_valid else 0.0
        )
        if self.traffic_substate == TRAFFIC_GREEN_FORWARD:
            return MissionOutput(
                MISSION_TRAFFIC,
                self.traffic_substate,
                "fixed_green",
                self.green_forward_steer_deg,
                self.green_forward_throttle,
            )
        if self.traffic_substate == TRAFFIC_YELLOW_DECELERATING:
            if self.yellow_deceleration_sec <= 0.0:
                throttle = 0.0
            else:
                progress = clamp(
                    (now_sec - self.yellow_start_time)
                    / self.yellow_deceleration_sec,
                    0.0,
                    1.0,
                )
                throttle = self.yellow_start_throttle * (1.0 - progress)
            return MissionOutput(
                MISSION_TRAFFIC,
                self.traffic_substate,
                "lane" if self.lane.has_last_valid else "none",
                lane_steer,
                throttle,
            )
        return MissionOutput(
            MISSION_TRAFFIC,
            self.traffic_substate,
            "lane" if self.lane.has_last_valid else "none",
            lane_steer,
            0.0,
        )

    def step(self, now_sec: float) -> MissionOutput:
        self._update_traffic_state(now_sec)
        traffic_active = self.traffic_substate in {
            TRAFFIC_RED_HOLD,
            TRAFFIC_YELLOW_DECELERATING,
            TRAFFIC_YELLOW_HOLD,
            TRAFFIC_GREEN_FORWARD,
        }
        if traffic_active:
            output = self._traffic_output(now_sec)
        elif self.obstacle_active:
            output = self._obstacle_output()
        else:
            output = self._lane_output()

        steer_limit = max(0.0, self.auto_steer_angle_abs_max - 1e-3)
        output = MissionOutput(
            output.mission_state,
            output.traffic_substate,
            output.selected_steer_source,
            clamp(output.steer_deg, -steer_limit, steer_limit),
            clamp(output.throttle, 0.0, self.auto_throttle_max),
        )
        self.current_output_throttle = output.throttle
        return output


class MissionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_manager")
        for name, default in DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        publish_rate_hz = float(parameter("publish_rate_hz"))
        if publish_rate_hz <= 0.0:
            self.get_logger().warning(
                "publish_rate_hz must be positive; using 30.0"
            )
            publish_rate_hz = 30.0
        self.log_rate_hz = float(parameter("log_rate_hz"))
        if self.log_rate_hz < 0.0:
            self.get_logger().warning(
                "log_rate_hz must be non-negative; using 1.0"
            )
            self.log_rate_hz = 1.0
        self.next_log_time = 0.0

        steer_abs_max = float(parameter("auto_steer_angle_abs_max"))
        if steer_abs_max <= 0.0:
            self.get_logger().warning(
                "auto_steer_angle_abs_max must be positive; using 23.0"
            )
            steer_abs_max = 23.0
        throttle_max = max(
            0.0, float(parameter("auto_throttle_max"))
        )
        lane_min, lane_max = self._validated_throttle_range(
            "lane",
            float(parameter("lane_throttle_min")),
            float(parameter("lane_throttle_max")),
            throttle_max,
        )
        obstacle_min, obstacle_max = self._validated_throttle_range(
            "obstacle",
            float(parameter("obstacle_throttle_min")),
            float(parameter("obstacle_throttle_max")),
            throttle_max,
        )
        yellow_duration = max(
            0.0, float(parameter("yellow_deceleration_sec"))
        )
        green_duration = max(
            0.0, float(parameter("green_forward_duration_sec"))
        )
        self.logic = MissionLogic(
            auto_steer_angle_abs_max=steer_abs_max,
            auto_throttle_max=throttle_max,
            throttle_curve_k=max(
                0.0, float(parameter("throttle_curve_k"))
            ),
            lane_throttle_min=lane_min,
            lane_throttle_max=lane_max,
            obstacle_throttle_min=obstacle_min,
            obstacle_throttle_max=obstacle_max,
            yellow_deceleration_sec=yellow_duration,
            green_forward_duration_sec=green_duration,
            green_forward_throttle=clamp(
                float(parameter("green_forward_throttle")),
                0.0,
                throttle_max,
            ),
            green_forward_steer_deg=float(
                parameter("green_forward_steer_deg")
            ),
        )

        self._create_subscriptions(parameter)
        self.final_steer_pub = self.create_publisher(
            Float32, str(parameter("final_steer_topic")), 10
        )
        self.final_throttle_pub = self.create_publisher(
            Float32, str(parameter("final_throttle_topic")), 10
        )
        self.mission_state_pub = self.create_publisher(
            String, str(parameter("mission_state_topic")), 10
        )
        self.status_pub = self.create_publisher(
            String, str(parameter("status_topic")), 10
        )
        self.timer = self.create_timer(
            1.0 / publish_rate_hz, self.on_timer
        )
        self.get_logger().info(
            f"mission_manager publishing final control at "
            f"{publish_rate_hz:.1f}Hz"
        )

    def _validated_throttle_range(
        self, label: str, lower: float, upper: float, limit: float
    ):
        lower = clamp(lower, 0.0, limit)
        upper = clamp(upper, 0.0, limit)
        if upper < lower:
            self.get_logger().warning(
                f"{label}_throttle_max is below min; swapping values"
            )
            lower, upper = upper, lower
        return lower, upper

    def _create_subscriptions(self, parameter) -> None:
        self.lane_steer_sub = self.create_subscription(
            Float32,
            str(parameter("lane_steer_topic")),
            lambda msg: self.logic.lane.update_steer(float(msg.data)),
            10,
        )
        self.lane_valid_sub = self.create_subscription(
            Bool,
            str(parameter("lane_valid_topic")),
            lambda msg: self.logic.lane.update_valid(bool(msg.data)),
            10,
        )
        self.obstacle_steer_sub = self.create_subscription(
            Float32,
            str(parameter("obstacle_steer_topic")),
            lambda msg: self.logic.obstacle.update_steer(float(msg.data)),
            10,
        )
        self.obstacle_valid_sub = self.create_subscription(
            Bool,
            str(parameter("obstacle_valid_topic")),
            lambda msg: self.logic.obstacle.update_valid(bool(msg.data)),
            10,
        )
        self.obstacle_active_sub = self.create_subscription(
            Bool,
            str(parameter("obstacle_active_topic")),
            self.on_obstacle_active,
            10,
        )
        self.traffic_detected_sub = self.create_subscription(
            Bool,
            str(parameter("traffic_detected_topic")),
            self.on_traffic_detected,
            10,
        )
        self.traffic_color_sub = self.create_subscription(
            String,
            str(parameter("traffic_color_topic")),
            self.on_traffic_color,
            10,
        )
        self.traffic_confidence_sub = self.create_subscription(
            Float32,
            str(parameter("traffic_confidence_topic")),
            self.on_traffic_confidence,
            10,
        )

    def on_obstacle_active(self, message: Bool) -> None:
        self.logic.obstacle_active = bool(message.data)

    def on_traffic_detected(self, message: Bool) -> None:
        self.logic.traffic_detected = bool(message.data)

    def on_traffic_color(self, message: String) -> None:
        color = str(message.data).strip().lower()
        self.logic.traffic_color = (
            color if color in VALID_TRAFFIC_COLORS else "none"
        )

    def on_traffic_confidence(self, message: Float32) -> None:
        confidence = float(message.data)
        self.logic.traffic_confidence = (
            confidence if math.isfinite(confidence) else 0.0
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def on_timer(self) -> None:
        now_sec = self._now_sec()
        output = self.logic.step(now_sec)
        self.final_steer_pub.publish(Float32(data=output.steer_deg))
        self.final_throttle_pub.publish(Float32(data=output.throttle))
        self.mission_state_pub.publish(
            String(data=output.mission_state)
        )
        status = {
            "mission_state": output.mission_state,
            "traffic_substate": output.traffic_substate,
            "traffic_detected": self.logic.traffic_detected,
            "traffic_color": self.logic.traffic_color,
            "traffic_confidence": round(
                self.logic.traffic_confidence, 3
            ),
            "obstacle_active": self.logic.obstacle_active,
            "selected_steer_source": output.selected_steer_source,
            "selected_steer_deg": round(output.steer_deg, 3),
            "output_throttle": round(output.throttle, 3),
            "lane_candidate_received": self.logic.lane.steer_received,
            "lane_candidate_valid": self.logic.lane.current_valid,
            "obstacle_candidate_received": (
                self.logic.obstacle.steer_received
            ),
            "obstacle_candidate_valid": (
                self.logic.obstacle.current_valid
            ),
        }
        status_json = json.dumps(status, ensure_ascii=False)
        self.status_pub.publish(String(data=status_json))
        if self.log_rate_hz == 0.0 or now_sec >= self.next_log_time:
            if self.log_rate_hz > 0.0:
                self.next_log_time = now_sec + 1.0 / self.log_rate_hz
            self.get_logger().info(status_json)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.final_steer_pub.publish(Float32(data=0.0))
            self.final_throttle_pub.publish(Float32(data=0.0))
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[MissionManagerNode] = None
    try:
        node = MissionManagerNode()
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
