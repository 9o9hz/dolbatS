#!/usr/bin/env python3
"""
차선 노드 켜지 않고 mission_manager에 차선같은 입력값을 줄 수 

# 터미널 1
ros2 run control_pkg serial_bridge

# 터미널 2 — 처음엔 lane_throttle을 낮게 override해서 여유를 두는 걸 추천
ros2 run mission_manager_pkg mission_manager --ros-args \
  -p lane_throttle_min:=0.2 -p lane_throttle_max:=0.25

# 터미널 3
ros2 run detect_pkg yolo_obstacle_turn
# (+ 초음파/YOLO detected를 실제로 발행하는 노드들도 같이)

# 터미널 4 — 이 테스트 스크립트
python3 manual_lane_candidate_test.py



"""

import curses
import json
import time
from typing import Optional, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class ManualLaneCandidateTest(Node):
    """Publish keyboard steering as the lane candidate, for hardware testing."""

    def __init__(self) -> None:
        super().__init__("manual_lane_candidate_test")

        self.declare_parameter("lane_steer_topic", "/control/candidate/lane/steer_angle")
        self.declare_parameter("lane_valid_topic", "/control/candidate/lane/valid")
        self.declare_parameter("status_topic", "/mission_manager/status")
        self.declare_parameter("steer_step_deg", 5.0)
        self.declare_parameter("max_steer_deg", 20.0)
        self.declare_parameter("publish_rate", 20.0)

        lane_steer_topic = str(self.get_parameter("lane_steer_topic").value)
        lane_valid_topic = str(self.get_parameter("lane_valid_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        self.steer_step_deg = abs(float(self.get_parameter("steer_step_deg").value))
        self.max_steer_deg = abs(float(self.get_parameter("max_steer_deg").value))
        publish_rate = max(1.0, float(self.get_parameter("publish_rate").value))

        self.steer_publisher = self.create_publisher(Float32, lane_steer_topic, 10)
        self.valid_publisher = self.create_publisher(Bool, lane_valid_topic, 10)
        self.status_subscription = self.create_subscription(
            String, status_topic, self.on_status, 10
        )

        self.steering_deg = 0.0
        self.quit_requested = False
        self.last_status: dict = {}
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_candidate)

        self.get_logger().warning(
            "TEST NODE: publishing FAKE lane candidate on "
            f"{lane_steer_topic}/{lane_valid_topic}. "
            "mission_manager, yolo_obstacle_turn, serial_bridge must already "
            "be running for real. This node cannot force throttle=0 — "
            "Ctrl+C mission_manager for a real stop."
        )

    def on_status(self, message: String) -> None:
        try:
            self.last_status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), ord("Q")):
            self.quit_requested = True
        elif key == curses.KEY_LEFT:
            self.steering_deg = max(
                -self.max_steer_deg, self.steering_deg - self.steer_step_deg
            )
        elif key == curses.KEY_RIGHT:
            self.steering_deg = min(
                self.max_steer_deg, self.steering_deg + self.steer_step_deg
            )
        elif key in (ord(" "), ord("c"), ord("C")):
            self.steering_deg = 0.0

    def publish_candidate(self) -> None:
        self.steer_publisher.publish(Float32(data=self.steering_deg))
        self.valid_publisher.publish(Bool(data=True))


def _run_terminal(screen: "curses.window", node: ManualLaneCandidateTest) -> None:
    curses.curs_set(0)
    screen.keypad(True)
    screen.nodelay(True)

    while rclpy.ok() and not node.quit_requested:
        screen.erase()
        screen.addstr(0, 0, "[TEST] manual steering -> lane candidate")
        screen.addstr(2, 0, "LEFT/RIGHT : steering candidate +/-")
        screen.addstr(3, 0, "SPACE/C    : center steering candidate (NOT a real stop)")
        screen.addstr(4, 0, "Q          : quit this test node")
        screen.addstr(
            6, 0, f"candidate steer_deg = {node.steering_deg:+.1f}"
        )

        status = node.last_status
        screen.addstr(8, 0, "-- mission_manager/status (live) --")
        screen.addstr(
            9, 0,
            f"mission_state        = {status.get('mission_state', '?')}"
        )
        screen.addstr(
            10, 0,
            f"selected_steer_source = {status.get('selected_steer_source', '?')}"
        )
        screen.addstr(
            11, 0,
            f"selected_steer_deg    = {status.get('selected_steer_deg', '?')}"
        )
        screen.addstr(
            12, 0,
            f"output_throttle       = {status.get('output_throttle', '?')}"
        )
        screen.addstr(
            13, 0,
            f"obstacle_candidate_valid = "
            f"{status.get('obstacle_candidate_valid', '?')}"
        )
        if status.get("selected_steer_source") == "obstacle":
            screen.addstr(15, 0, "*** OBSTACLE TAKEOVER ACTIVE ***")

        screen.refresh()

        while True:
            key = screen.getch()
            if key == -1:
                break
            node.handle_key(key)

        rclpy.spin_once(node, timeout_sec=0.01)


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = ManualLaneCandidateTest()
    try:
        curses.wrapper(_run_terminal, node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()