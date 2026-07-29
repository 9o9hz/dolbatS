import math
import unittest

from mission_manager import (
    MISSION_LANE,
    MISSION_OBSTACLE,
    MISSION_TRAFFIC,
    TRAFFIC_GREEN_FORWARD,
    TRAFFIC_GREEN_WAIT_CLEAR,
    TRAFFIC_RED_HOLD,
    TRAFFIC_YELLOW_DECELERATING,
    TRAFFIC_YELLOW_HOLD,
    MissionLogic,
    map_throttle_by_steer,
)


class ThrottleMappingTest(unittest.TestCase):
    def test_linear_mapping_and_clamp(self):
        values = [
            (0.0, 0.8),
            (23.0, 0.4),
            (-23.0, 0.4),
            (30.0, 0.4),
        ]
        for steer, expected in values:
            with self.subTest(steer=steer):
                actual = map_throttle_by_steer(
                    steer, 23.0, 0.4, 0.8, 0.0
                )
                self.assertAlmostEqual(actual, expected)


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

    def test_obstacle_requires_its_own_candidate(self):
        logic = MissionLogic()
        logic.lane.update_steer(5.0)
        logic.lane.update_valid(True)
        logic.obstacle_detected = True
        output = logic.step(0.0)
        self.assertEqual(output.mission_state, MISSION_OBSTACLE)
        self.assertEqual((output.steer_deg, output.throttle), (0.0, 0.0))

        logic.obstacle.update_steer(-12.0)
        logic.obstacle.update_valid(True)
        output = logic.step(1.0)
        self.assertEqual(output.selected_steer_source, "obstacle")
        self.assertEqual(output.steer_deg, -12.0)
        self.assertGreater(output.throttle, 0.4)
        self.assertLess(output.throttle, 0.8)

        logic.obstacle_detected = False
        self.assertEqual(logic.step(2.0).mission_state, MISSION_LANE)


class TrafficStateTest(unittest.TestCase):
    def setUp(self):
        self.logic = MissionLogic()
        self.logic.lane.update_steer(0.0)
        self.logic.lane.update_valid(True)
        self.assertAlmostEqual(self.logic.step(0.0).throttle, 0.8)

    def set_traffic(self, detected, color):
        self.logic.traffic_detected = detected
        self.logic.traffic_color = color

    def test_red_hold_priority_and_green_release(self):
        self.logic.obstacle_detected = True
        self.set_traffic(True, "red")
        output = self.logic.step(1.0)
        self.assertEqual(output.mission_state, MISSION_TRAFFIC)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)
        self.assertEqual(output.throttle, 0.0)

        self.set_traffic(False, "none")
        output = self.logic.step(2.0)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)

        self.set_traffic(True, "green")
        output = self.logic.step(3.0)
        self.assertEqual(output.traffic_substate, TRAFFIC_GREEN_FORWARD)
        self.assertEqual((output.steer_deg, output.throttle), (0.0, 0.4))

    def test_yellow_deceleration_and_hold(self):
        self.set_traffic(True, "yellow")
        output = self.logic.step(1.0)
        self.assertEqual(
            output.traffic_substate, TRAFFIC_YELLOW_DECELERATING
        )
        self.assertAlmostEqual(output.throttle, 0.8)

        self.set_traffic(False, "none")
        self.assertAlmostEqual(self.logic.step(2.5).throttle, 0.4)
        output = self.logic.step(4.0)
        self.assertEqual(output.traffic_substate, TRAFFIC_YELLOW_HOLD)
        self.assertEqual(output.throttle, 0.0)

    def test_yellow_red_and_green_rules(self):
        self.set_traffic(True, "yellow")
        self.logic.step(1.0)
        self.set_traffic(True, "green")
        output = self.logic.step(2.0)
        self.assertEqual(
            output.traffic_substate, TRAFFIC_YELLOW_DECELERATING
        )
        self.set_traffic(True, "red")
        output = self.logic.step(2.1)
        self.assertEqual(output.traffic_substate, TRAFFIC_RED_HOLD)
        self.assertEqual(output.throttle, 0.0)

    def test_green_completion_latch(self):
        self.set_traffic(True, "green")
        output = self.logic.step(1.0)
        self.assertEqual(output.traffic_substate, TRAFFIC_GREEN_FORWARD)
        self.assertEqual(self.logic.step(3.9).mission_state, MISSION_TRAFFIC)

        output = self.logic.step(4.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(
            output.traffic_substate, TRAFFIC_GREEN_WAIT_CLEAR
        )
        self.assertTrue(self.logic.traffic_green_completed)

        output = self.logic.step(10.0)
        self.assertEqual(output.mission_state, MISSION_LANE)
        self.assertEqual(
            output.traffic_substate, TRAFFIC_GREEN_WAIT_CLEAR
        )

        self.set_traffic(False, "none")
        self.logic.step(11.0)
        self.assertFalse(self.logic.traffic_green_completed)


if __name__ == "__main__":
    unittest.main()
