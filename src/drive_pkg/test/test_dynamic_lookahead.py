import unittest

from pure_pursuit import PurePursuitController


def make_controller():
    return PurePursuitController(
        wheelbase_m=0.545,
        ld_throttle_min=0.4,
        ld_throttle_max=0.8,
        lookahead_min_m=2.3,
        lookahead_max_m=2.8,
        fixed_lookahead_m=-1.0,
        max_steer_deg=18.0,
        steering_ema_alpha=0.35,
        max_steering_change_deg=3.0,
    )


class DynamicLookaheadTest(unittest.TestCase):
    def test_throttle_mapping_and_clamp(self):
        controller = make_controller()
        cases = [
            (0.4, 2.3),
            (0.8, 2.8),
            (0.6, 2.55),
            (0.1, 2.3),
            (1.0, 2.8),
        ]
        for throttle, expected in cases:
            with self.subTest(throttle=throttle):
                controller.set_current_throttle(throttle)
                self.assertAlmostEqual(
                    controller._dynamic_lookahead(),
                    expected,
                    places=5,
                )

    def test_fixed_lookahead_is_preserved(self):
        controller = make_controller()
        controller.lookahead_min_m = 2.5
        controller.lookahead_max_m = 2.5
        controller.set_current_throttle(0.8)
        self.assertEqual(controller._dynamic_lookahead(), 2.5)


if __name__ == "__main__":
    unittest.main()
