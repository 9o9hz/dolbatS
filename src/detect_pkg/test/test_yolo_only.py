import unittest

from yolo_obstacle_yolo_only import (
    TimedTurnState,
    centered_bbox_is_large_enough,
    next_timed_turn_state,
    opposite_turn_state,
)


class CenteredBboxTriggerTest(unittest.TestCase):
    def test_each_timed_turn_returns_to_lane(self):
        state = TimedTurnState.TURN_LEFT
        self.assertEqual(
            next_timed_turn_state(state, 0.9, 1.0),
            TimedTurnState.TURN_LEFT,
        )
        state = next_timed_turn_state(state, 1.0, 1.0)
        self.assertEqual(state, TimedTurnState.WAIT_CLEAR)

        self.assertEqual(
            next_timed_turn_state(TimedTurnState.TURN_RIGHT, 1.0, 1.0),
            TimedTurnState.WAIT_CLEAR,
        )

    def test_each_timed_turn_can_countersteer_for_equal_duration(self):
        state = next_timed_turn_state(
            TimedTurnState.TURN_LEFT, 1.0, 1.0, True
        )
        self.assertEqual(state, TimedTurnState.COUNTERSTEER_RIGHT)
        self.assertEqual(
            next_timed_turn_state(state, 0.9, 1.0, True),
            TimedTurnState.COUNTERSTEER_RIGHT,
        )
        self.assertEqual(
            next_timed_turn_state(state, 1.0, 1.0, True),
            TimedTurnState.WAIT_CLEAR,
        )

        state = next_timed_turn_state(
            TimedTurnState.TURN_RIGHT, 1.0, 1.0, True
        )
        self.assertEqual(state, TimedTurnState.COUNTERSTEER_LEFT)

    def test_avoidance_direction_alternates_left_and_right(self):
        state = TimedTurnState.TURN_LEFT
        state = opposite_turn_state(state)
        self.assertEqual(state, TimedTurnState.TURN_RIGHT)
        state = opposite_turn_state(state)
        self.assertEqual(state, TimedTurnState.TURN_LEFT)

    def test_centered_bbox_at_minimum_area_triggers(self):
        self.assertTrue(
            centered_bbox_is_large_enough(
                320.0,
                160.0,
                153.6,
                640,
                480,
                1.0 / 3.0,
                2.0 / 3.0,
                0.08,
            )
        )

    def test_bbox_outside_middle_third_does_not_trigger(self):
        self.assertFalse(
            centered_bbox_is_large_enough(
                100.0, 300.0, 200.0, 640, 480, 1.0 / 3.0, 2.0 / 3.0, 0.08
            )
        )

    def test_small_centered_bbox_does_not_trigger(self):
        self.assertFalse(
            centered_bbox_is_large_enough(
                320.0, 100.0, 100.0, 640, 480, 1.0 / 3.0, 2.0 / 3.0, 0.08
            )
        )


if __name__ == "__main__":
    unittest.main()
