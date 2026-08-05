import unittest

from yolo_obstacle_turn import (
    LeftHalfDisappearanceTrigger,
    should_monitor_ultrasonic,
    turn_end_threshold_reached,
)


class LeftHalfDisappearanceTriggerTest(unittest.TestCase):
    def test_disabled_yolo_gate_monitors_ultrasonic_immediately(self):
        self.assertTrue(should_monitor_ultrasonic(False, False))

    def test_enabled_yolo_gate_waits_until_open(self):
        self.assertFalse(should_monitor_ultrasonic(True, False))
        self.assertTrue(should_monitor_ultrasonic(True, True))

    def test_turn_ends_only_above_threshold(self):
        self.assertFalse(turn_end_threshold_reached(69.9, 70.0))
        self.assertFalse(turn_end_threshold_reached(70.0, 70.0))
        self.assertTrue(turn_end_threshold_reached(70.1, 70.0))

    def test_invalid_distance_does_not_end_turn(self):
        self.assertFalse(turn_end_threshold_reached(-1.0, 70.0))
        self.assertFalse(turn_end_threshold_reached(float("nan"), 70.0))

    def test_right_half_target_does_not_enable_ultrasonic(self):
        trigger = LeftHalfDisappearanceTrigger(missing_frames_required=3)

        trigger.observe_detection(True)
        trigger.observe_bbox(500.0, 640)

        self.assertFalse(trigger.observe_detection(False))
        self.assertFalse(trigger.observe_detection(False))
        self.assertFalse(trigger.observe_detection(False))
        self.assertFalse(trigger.enabled)

    def test_left_half_target_then_three_missing_frames_enables(self):
        trigger = LeftHalfDisappearanceTrigger(missing_frames_required=3)

        trigger.observe_detection(True)
        self.assertTrue(trigger.observe_bbox(200.0, 640))
        self.assertFalse(trigger.observe_detection(False))
        self.assertFalse(trigger.observe_detection(False))
        self.assertTrue(trigger.observe_detection(False))
        self.assertTrue(trigger.enabled)

    def test_detection_reappearing_resets_missing_count(self):
        trigger = LeftHalfDisappearanceTrigger(missing_frames_required=2)

        trigger.observe_bbox(100.0, 640)
        self.assertFalse(trigger.observe_detection(False))
        trigger.observe_detection(True)
        self.assertFalse(trigger.observe_detection(False))
        self.assertTrue(trigger.observe_detection(False))

    def test_reset_requires_a_new_left_half_detection(self):
        trigger = LeftHalfDisappearanceTrigger(missing_frames_required=1)
        trigger.observe_bbox(100.0, 640)
        self.assertTrue(trigger.observe_detection(False))

        trigger.reset()

        self.assertFalse(trigger.observe_detection(False))
        self.assertFalse(trigger.enabled)


if __name__ == "__main__":
    unittest.main()
