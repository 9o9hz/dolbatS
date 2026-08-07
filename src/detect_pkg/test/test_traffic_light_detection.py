import unittest

from traffic_light_detection import red_bbox_is_tall_enough


class RedBboxHeightTest(unittest.TestCase):
    def test_height_must_be_at_least_threshold(self):
        self.assertFalse(
            red_bbox_is_tall_enough((10.0, 20.0, 500.0, 109.9), 110.0)
        )
        self.assertTrue(
            red_bbox_is_tall_enough((10.0, 20.0, 1.0, 110.0), 110.0)
        )


if __name__ == "__main__":
    unittest.main()
