import unittest
from types import SimpleNamespace

from yolo_obstacle_bbox_turn import bbox_crossed_exit_boundary
from yolo_obstacle_fusion_end import YoloObstacleFusionEnd


class CombinedEndConditionTest(unittest.TestCase):
    def test_sonic_threshold_can_finish_turn(self):
        finished = []
        fake_node = SimpleNamespace(
            turn_end_threshold_cm=70.0,
            finish_turn=finished.append,
        )

        YoloObstacleFusionEnd.update_turn_end(fake_node, 70.0)
        self.assertEqual(finished, [])
        YoloObstacleFusionEnd.update_turn_end(fake_node, 70.1)
        self.assertEqual(finished, [70.1])

    def test_bbox_boundary_can_finish_turn(self):
        self.assertTrue(
            bbox_crossed_exit_boundary(
                center_x=500.0,
                image_width=640,
                turn_side="left",
                left_boundary_ratio=0.29,
                right_boundary_ratio=0.705,
            )
        )


if __name__ == "__main__":
    unittest.main()
