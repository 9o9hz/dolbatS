import unittest
from types import SimpleNamespace

from std_msgs.msg import Float32MultiArray

from yolo_bbox_utils import bbox_crossed_exit_boundary
from yolo_obstacle_yolo_only import (
    YoloObstacleYoloOnly,
    YoloOnlyState,
    centered_bbox_is_large_enough,
    turn_state_from_direction,
)


class CenteredBboxTriggerTest(unittest.TestCase):
    def test_l_argument_selects_left_turn(self):
        self.assertEqual(
            turn_state_from_direction("L"), YoloOnlyState.TURN_LEFT
        )

    def test_r_argument_selects_right_turn(self):
        self.assertEqual(
            turn_state_from_direction("R"), YoloOnlyState.TURN_RIGHT
        )

    def test_direction_argument_is_case_insensitive(self):
        self.assertEqual(
            turn_state_from_direction(" r "), YoloOnlyState.TURN_RIGHT
        )

    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "L.*R"):
            turn_state_from_direction("forward")

    def test_left_turn_ends_at_same_right_boundary_as_yolo_end(self):
        self.assertTrue(
            bbox_crossed_exit_boundary(
                451.2, 640, "left", 0.29, 0.705
            )
        )
        self.assertFalse(
            bbox_crossed_exit_boundary(
                451.1, 640, "left", 0.29, 0.705
            )
        )

    def test_right_turn_ends_at_same_left_boundary_as_yolo_end(self):
        self.assertTrue(
            bbox_crossed_exit_boundary(
                185.6, 640, "right", 0.29, 0.705
            )
        )
        self.assertFalse(
            bbox_crossed_exit_boundary(
                185.7, 640, "right", 0.29, 0.705
            )
        )

    def test_turn_ends_after_configured_consecutive_exit_frames(self):
        finished = []
        fake_node = SimpleNamespace(
            state=YoloOnlyState.TURN_LEFT,
            image_width=640,
            bbox_left_boundary_ratio=0.29,
            bbox_right_boundary_ratio=0.705,
            bbox_exit_consecutive_frames=8,
            bbox_exit_frames=0,
            last_bbox_center_x=None,
            finish_turn_from_bbox=lambda center_x, direction: finished.append(
                (center_x, direction)
            ),
        )
        crossed = Float32MultiArray(data=[500.0, 240.0, 100.0, 100.0])

        for _ in range(7):
            YoloObstacleYoloOnly.on_bbox(fake_node, crossed)
        self.assertEqual(finished, [])

        YoloObstacleYoloOnly.on_bbox(fake_node, crossed)
        self.assertEqual(finished, [(500.0, "left")])

    def test_exit_frame_count_resets_before_consecutive_limit(self):
        fake_node = SimpleNamespace(
            state=YoloOnlyState.TURN_RIGHT,
            image_width=640,
            bbox_left_boundary_ratio=0.29,
            bbox_right_boundary_ratio=0.705,
            bbox_exit_consecutive_frames=3,
            bbox_exit_frames=2,
            last_bbox_center_x=None,
            finish_turn_from_bbox=lambda center_x, direction: self.fail(
                "non-consecutive bbox exits must not finish the turn"
            ),
        )
        not_crossed = Float32MultiArray(
            data=[300.0, 240.0, 100.0, 100.0]
        )

        YoloObstacleYoloOnly.on_bbox(fake_node, not_crossed)
        self.assertEqual(fake_node.bbox_exit_frames, 0)

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
