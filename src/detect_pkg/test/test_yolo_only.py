import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from std_msgs.msg import Bool, Float32MultiArray

from obstacle_detector_publisher import (
    ObstacleDetectorPublisher,
    normalized_overlay_geometry,
)
from yolo_bbox_utils import bbox_crossed_exit_boundary
from yolo_obstacle_yolo_only import (
    YoloObstacleYoloOnly,
    YoloOnlyState,
    centered_bbox_is_large_enough,
    opposite_turn_state,
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

    def test_left_turn_is_followed_by_right_turn(self):
        self.assertEqual(
            opposite_turn_state(YoloOnlyState.TURN_LEFT),
            YoloOnlyState.TURN_RIGHT,
        )

    def test_right_turn_is_followed_by_left_turn(self):
        self.assertEqual(
            opposite_turn_state(YoloOnlyState.TURN_RIGHT),
            YoloOnlyState.TURN_LEFT,
        )

    def test_non_turn_state_has_no_opposite_direction(self):
        with self.assertRaisesRegex(ValueError, "TURN_LEFT.*TURN_RIGHT"):
            opposite_turn_state(YoloOnlyState.WAIT_TRIGGER)

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

    def test_completed_left_turn_sets_next_turn_to_right(self):
        published_reasons = []
        fake_node = SimpleNamespace(
            state=YoloOnlyState.TURN_LEFT,
            image_width=640,
            bbox_left_boundary_ratio=0.20,
            bbox_right_boundary_ratio=0.80,
            next_turn_state=YoloOnlyState.TURN_LEFT,
            clear_frames=7,
            publish_candidate=lambda: None,
            publish_status=published_reasons.append,
            get_logger=lambda: SimpleNamespace(warning=lambda message: None),
        )

        YoloObstacleYoloOnly.finish_turn_from_bbox(
            fake_node, 520.0, "left"
        )

        self.assertEqual(fake_node.state, YoloOnlyState.WAIT_CLEAR)
        self.assertEqual(
            fake_node.next_turn_state, YoloOnlyState.TURN_RIGHT
        )
        self.assertEqual(fake_node.clear_frames, 0)
        self.assertEqual(
            published_reasons, ["bbox_crossed_right_boundary"]
        )

    def test_successive_obstacles_use_alternating_turn_directions(self):
        published_states = []
        fake_node = SimpleNamespace(
            state=YoloOnlyState.WAIT_TRIGGER,
            next_turn_state=YoloOnlyState.TURN_LEFT,
            image_width=640,
            image_height=480,
            middle_left_ratio=0.20,
            middle_right_ratio=0.80,
            roi_top_ratio=0.0,
            roi_bottom_ratio=1.0,
            min_bbox_area_ratio=0.015,
            bbox_left_boundary_ratio=0.20,
            bbox_right_boundary_ratio=0.80,
            bbox_exit_frames=0,
            clear_frames=0,
            rearm_clear_frames=1,
            yolo_detected=True,
            last_bbox_center_x=None,
            publish_status=lambda reason: None,
            get_logger=lambda: SimpleNamespace(warning=lambda message: None),
        )
        fake_node.publish_candidate = lambda: published_states.append(
            fake_node.state
        )
        centered = Float32MultiArray(
            data=[320.0, 240.0, 100.0, 100.0]
        )

        YoloObstacleYoloOnly.on_bbox(fake_node, centered)
        self.assertEqual(fake_node.state, YoloOnlyState.TURN_LEFT)

        YoloObstacleYoloOnly.finish_turn_from_bbox(
            fake_node, 520.0, "left"
        )
        YoloObstacleYoloOnly.on_yolo_detected(
            fake_node, Bool(data=False)
        )
        YoloObstacleYoloOnly.on_bbox(fake_node, centered)

        self.assertEqual(fake_node.state, YoloOnlyState.TURN_RIGHT)
        self.assertEqual(
            published_states,
            [
                YoloOnlyState.TURN_LEFT,
                YoloOnlyState.WAIT_CLEAR,
                YoloOnlyState.TURN_RIGHT,
            ],
        )

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

    def test_bbox_center_above_bottom_limit_can_trigger(self):
        self.assertTrue(
            centered_bbox_is_large_enough(
                320.0,
                100.0,
                100.0,
                640,
                480,
                0.20,
                0.80,
                0.02,
                center_y=420.0,
                roi_top_ratio=0.0,
                roi_bottom_ratio=0.90,
            )
        )

    def test_bbox_center_below_configured_roi_does_not_trigger(self):
        self.assertFalse(
            centered_bbox_is_large_enough(
                320.0,
                100.0,
                100.0,
                640,
                480,
                0.20,
                0.80,
                0.02,
                center_y=450.0,
                roi_top_ratio=0.0,
                roi_bottom_ratio=0.90,
            )
        )

    def test_overlay_draws_configured_bottom_limit(self):
        geometry = normalized_overlay_geometry(
            640,
            480,
            0.20,
            0.80,
            0.0,
            0.90,
            0.20,
            0.80,
        )

        self.assertEqual(geometry["roi_top"], 0)
        self.assertEqual(geometry["roi_bottom"], 432)
        self.assertEqual(geometry["exit_left"], 128)
        self.assertEqual(geometry["exit_right"], 512)

    def test_overlay_labels_every_bbox_with_its_confidence(self):
        fake_node = SimpleNamespace(
            draw_yolo_only_guides=lambda frame: None,
            draw_status_panel=lambda frame: None,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = (
            ((120.0, 200.0, 80.0, 100.0), 0.91),
            ((420.0, 220.0, 70.0, 90.0), 0.73),
        )

        with patch("obstacle_detector_publisher.cv2.putText") as put_text:
            ObstacleDetectorPublisher.draw_detection_overlay(
                fake_node, frame, detections
            )

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(
            labels, ["confidence 0.91", "confidence 0.73"]
        )


if __name__ == "__main__":
    unittest.main()
