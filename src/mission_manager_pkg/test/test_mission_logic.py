import math
import unittest

from mission_manager import (
    MISSION_LANE,
    MISSION_OBSTACLE,
    MISSION_TRAFFIC,
    TRAFFIC_IDLE,
    TRAFFIC_RED_HOLD,
    TRAFFIC_YELLOW_DECELERATING,
    TRAFFIC_YELLOW_HOLD,
    MissionLogic,
    MissionOutput,
    apply_drive_enable_gate,
    map_throttle_by_steer,
)


class ThrottleMappingTest(unittest.TestCase):
    def test_linear_mapping_and_clamp(self):
        values = [
            (0.0, 0.8),
            (26.5, 0.4),
            (-26.5, 0.4),
            (30.0, 0.4),
        ]
        for steer, expected in values:
            with self.subTest(steer=steer):
                actual = map_throttle_by_steer(
                    steer, 26.5, 0.4, 0.8, 0.0
                )
                self.assertAlmostEqual(actual, expected)

    def test_drive_enable_gate_defaults_to_zero_output(self):
        output = MissionOutput(
            MISSION_LANE,
            "idle",
            "lane",
            12.0,
            0.5,
        )
        self.assertEqual(
            apply_drive_enable_gate(output, False),
            (0.0, 0.0, "drive_disabled"),
        )
        self.assertEqual(
            apply_drive_enable_gate(output, True),
            (12.0, 0.5, "lane"),
        )


class CandidateAndModeTest(unittest.TestCase):
    def test_lane_requires_one_valid_candidate_and_holds_it(self):
        logic = MissionLogic()
        output = logic.step(0.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual((output.steer_deg, output.throttle), (0.0, 0.0))

        logic.lane.update_steer(10.0)
        logic.lane.update_valid(True)
        output = logic.step(1.0)
        self.assertEqual(output.selected_steer_source, "lane")
        self.assertEqual(output.steer_deg, 10.0)

        logic.lane.update_valid(False)
        logic.lane.update_steer(-5.0)
        output = logic.step(10000.0)
        self.assertEqual(output.steer_deg, 10.0)

    def test_obstacle_trigger_valid_now_wins_over_lane(self):
        logic = MissionLogic()
        logic.obstacle_active = True
        logic.lane.update_steer(7.0)
        logic.lane.update_valid(True)
        output = logic.step(0.0)
        self.assertEqual(output.mission_state, MISSION_OBSTACLE)

        # TURN trigger currently valid: obstacle steering wins over lane.
        logic.obstacle.update_steer(-12.0)
        logic.obstacle.update_valid(True)
        output = logic.step(1.0)
        self.assertEqual(output.selected_steer_source, "obstacle")
        self.assertEqual(output.steer_deg, -12.0)
        self.assertGreater(output.throttle, 0.4)
        self.assertLess(output.throttle, 0.8)

        # Trigger ends (TURN -> REARM): fall back to lane
        # immediately rather than holding the stale obstacle steer, but
        # keep using the obstacle throttle range while still active.
        logic.obstacle.update_valid(False)
        output = logic.step(1.5)
        self.assertEqual(output.selected_steer_source, "lane")
        self.assertEqual(output.steer_deg, 7.0)
        self.assertGreater(output.throttle, 0.4)
        self.assertLess(output.throttle, 0.8)

        logic.obstacle_active = False
        self.assertEqual(logic.step(2.0).mission_state, MISSION_LANE)

    def test_obstacle_active_without_trigger_follows_lane(self):
        logic = MissionLogic()
        logic.obstacle_active = True
        logic.lane.update_steer(9.0)
        logic.lane.update_valid(True)
        # Obstacle candidate never becomes valid.
        output = logic.step(0.0)
        self.assertEqual(output.mission_state, MISSION_OBSTACLE)
        self.assertEqual(output.selected_steer_source, "lane")
        self.assertEqual(output.steer_deg, 9.0)
        self.assertGreaterEqual(output.throttle, 0.4)
        self.assertLessEqual(output.throttle, 0.8)

    def test_obstacle_active_with_no_candidates_is_safe(self):
        logic = MissionLogic()
        logic.obstacle_active = True
        output = logic.step(0.0)
        self.assertEqual(output.selected_steer_source, "none")
        self.assertEqual((output.steer_deg, output.throttle), (0.0, 0.0))


class TrafficStateTest(unittest.TestCase):
    def setUp(self):
        self.logic = MissionLogic()
        self.logic.lane.update_steer(0.0)
        self.logic.lane.update_valid(True)
        self.assertAlmostEqual(self.logic.step(0.0).throttle, 0.6)

    def set_traffic(self, detected, color):
        self.logic.update_traffic_detected(detected)
        self.logic.traffic_color = color

    def test_red_hold_releases_after_three_missed_frames(self):
        self.set_traffic(True, "red")
        output = self.logic.step(1.0)
        self.assertEqual(output.mission_state, MISSION_TRAFFIC)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)
        self.assertEqual(output.throttle, 0.0)

        for frame in range(1, 3):
            self.set_traffic(False, "none")
            output = self.logic.step(1.0 + frame)
            self.assertEqual(output.mission_state, MISSION_TRAFFIC)
            self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)

        self.set_traffic(False, "none")
        output = self.logic.step(4.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.assertEqual(output.selected_steer_source, "lane")

    def test_red_missed_frame_count_must_be_consecutive(self):
        self.set_traffic(True, "red")
        self.logic.step(1.0)
        self.set_traffic(False, "none")
        self.logic.step(2.0)
        self.set_traffic(False, "none")
        self.logic.step(3.0)

        self.set_traffic(True, "red")
        self.assertEqual(self.logic.traffic_missed_frames, 0)
        self.set_traffic(False, "none")
        output = self.logic.step(4.0)

        self.assertEqual(output.mission_state, MISSION_TRAFFIC)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)
        self.assertEqual(self.logic.traffic_missed_frames, 1)

    def test_red_hold_releases_on_green(self):
        self.set_traffic(True, "red")
        self.assertEqual(
            self.logic.step(1.0).traffic_substate,
            TRAFFIC_RED_HOLD,
        )

        self.set_traffic(True, "green")
        output = self.logic.step(2.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.assertEqual(output.selected_steer_source, "lane")
        self.assertEqual((output.steer_deg, output.throttle), (0.0, 0.6))

    def test_yellow_deceleration_and_hold(self):
        self.set_traffic(True, "yellow")
        output = self.logic.step(1.0)
        self.assertEqual(
            output.traffic_substate, TRAFFIC_YELLOW_DECELERATING
        )
        self.assertAlmostEqual(output.throttle, 0.6)

        self.set_traffic(False, "none")
        self.assertAlmostEqual(self.logic.step(2.5).throttle, 0.3)
        output = self.logic.step(4.0)
        self.assertEqual(output.traffic_substate, TRAFFIC_YELLOW_HOLD)
        self.assertEqual(output.throttle, 0.0)

    def test_yellow_red_and_green_rules(self):
        self.set_traffic(True, "yellow")
        self.logic.step(1.0)
        self.set_traffic(True, "green")
        output = self.logic.step(2.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.set_traffic(True, "red")
        output = self.logic.step(2.1)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)
        self.assertEqual(output.throttle, 0.0)

    def test_green_remains_normal_lane_driving_without_latch(self):
        self.set_traffic(True, "green")
        output = self.logic.step(1.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.assertEqual(output.selected_steer_source, "lane")

        output = self.logic.step(10.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.assertFalse(self.logic.traffic_green_completed)

        self.set_traffic(False, "none")
        self.logic.step(11.0)
        self.assertFalse(self.logic.traffic_green_completed)

    def test_green_does_not_disable_obstacle_avoidance(self):
        self.logic.obstacle_active = True
        self.logic.obstacle.update_steer(-12.0)
        self.logic.obstacle.update_valid(True)
        self.set_traffic(True, "green")

        output = self.logic.step(1.0)

        self.assertEqual(output.mission_state, MISSION_OBSTACLE)
        self.assertEqual(output.traffic_substate, TRAFFIC_IDLE)
        self.assertEqual(output.selected_steer_source, "obstacle")


if __name__ == "__main__":
    unittest.main()
