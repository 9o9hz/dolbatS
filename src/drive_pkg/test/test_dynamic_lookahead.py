import unittest

from pure_pursuit import PurePursuitController


def make_controller(dynamic=True, fixed=1.5):
    return PurePursuitController(
        wheelbase_m=0.545,
        ld_throttle_min=0.4,
        ld_throttle_max=0.8,
        lookahead_min_m=1.1,
        lookahead_max_m=2.0,
        fixed_lookahead_m=fixed,
        dynamic_lookahead_enabled=dynamic,
        max_steer_deg=18.0,
        steering_gain=1.8,
        steering_ema_alpha=0.35,
        steering_deadband_deg=0.8,
        max_steering_change_deg=3.0,
    )


class DynamicLookaheadTest(unittest.TestCase):
    def test_steering_mapping_and_clamp(self):
        controller = make_controller()
        cases = [
            (0.0, 2.0),
            (9.0, 1.55),
            (18.0, 1.1),
            (30.0, 1.1),
        ]
        for steering, expected in cases:
            with self.subTest(steering=steering):
                controller.last_steering_deg = steering
                self.assertAlmostEqual(
                    controller._dynamic_lookahead(),
                    expected,
                    places=5,
                )

    def test_fixed_lookahead_is_preserved(self):
        controller = make_controller(dynamic=False, fixed=1.65)
        controller.last_steering_deg = 18.0
        self.assertEqual(controller._dynamic_lookahead(), 1.65)


if __name__ == "__main__":
    unittest.main()
