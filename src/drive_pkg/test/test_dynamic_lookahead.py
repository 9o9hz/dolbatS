import math
import unittest

import numpy as np

from pure_pursuit import PurePursuitController


def make_controller(dynamic=True, fixed=1.5, **overrides):
    parameters = {
        "wheelbase_m": 0.545,
        "ld_throttle_min": 0.4,
        "ld_throttle_max": 0.8,
        "lookahead_min_m": 1.1,
        "lookahead_max_m": 2.0,
        "fixed_lookahead_m": fixed,
        "dynamic_lookahead_enabled": dynamic,
        "max_steer_deg": 18.0,
        "steering_gain": 1.8,
        "steering_ema_alpha": 0.35,
        "steering_deadband_deg": 0.8,
        "max_steering_change_deg": 3.0,
        "lookahead_search_mode": "continuous_arc_length",
        "curvature_lookahead_enabled": True,
        "curvature_full_scale_1pm": 0.5,
        "curvature_reduction_max_m": 0.15,
        "curvature_sample_gap_m": 0.15,
        "lookahead_filter_tau_sec": 0.25,
        "time_based_steering_limit_enabled": True,
        "nominal_control_rate_hz": 30.0,
        "min_control_dt_sec": 0.005,
        "max_control_dt_sec": 0.2,
        "max_steering_rate_deg_s": 75.0,
        "max_steering_accel_deg_s2": 300.0,
    }
    parameters.update(overrides)
    return PurePursuitController(**parameters)


class DynamicLookaheadTest(unittest.TestCase):
    def test_throttle_mapping_and_clamp(self):
        controller = make_controller()
        cases = [
            (0.0, 1.1),
            (0.4, 1.1),
            (0.6, 1.55),
            (0.8, 2.0),
            (1.0, 2.0),
            (-0.8, 2.0),
        ]
        for throttle, expected in cases:
            with self.subTest(throttle=throttle):
                controller.set_current_throttle(throttle)
                self.assertAlmostEqual(
                    controller._dynamic_lookahead(), expected, places=5
                )

    def test_steering_does_not_feed_back_into_lookahead(self):
        controller = make_controller()
        controller.set_current_throttle(0.6)
        expected = controller._dynamic_lookahead()
        controller.last_steering_deg = controller.max_steer_deg
        self.assertEqual(controller._dynamic_lookahead(), expected)

    def test_fixed_lookahead_is_preserved(self):
        controller = make_controller(dynamic=False, fixed=1.65)
        controller.set_current_throttle(0.8)
        self.assertEqual(controller._dynamic_lookahead(), 1.65)
        self.assertEqual(
            controller._curvature_adjusted_lookahead(1.65, 1.0), 1.65
        )

    def test_continuous_target_interpolates_inside_segment(self):
        points = np.asarray([[0.2, 0.0], [1.2, 0.0], [2.2, 0.0]])
        target = PurePursuitController.interpolate_lookahead_point(
            points, 1.55
        )
        np.testing.assert_allclose(target.point, [1.55, 0.0])
        self.assertEqual((target.lower_index, target.upper_index), (1, 2))
        self.assertAlmostEqual(target.segment_ratio, 0.35)
        self.assertFalse(target.clamped_to_endpoint)

    def test_reversed_path_produces_same_target_point(self):
        points = np.asarray(
            [[0.2, 0.0], [0.8, 0.1], [1.4, 0.3], [2.0, 0.6]]
        )
        forward = PurePursuitController.interpolate_lookahead_point(
            points, 1.25
        )
        reverse = PurePursuitController.interpolate_lookahead_point(
            points[::-1], 1.25
        )
        np.testing.assert_allclose(forward.point, reverse.point)

    def test_duplicate_points_do_not_create_zero_length_segment(self):
        points = np.asarray(
            [[0.2, 0.0], [0.8, 0.0], [0.8, 0.0], [1.8, 0.0]]
        )
        target = PurePursuitController.interpolate_lookahead_point(
            points, 1.3
        )
        np.testing.assert_allclose(target.point, [1.3, 0.0])
        self.assertTrue(math.isfinite(target.segment_ratio))

    def test_target_is_independent_of_path_sample_density(self):
        sparse = np.column_stack((np.linspace(0.2, 2.2, 9), np.zeros(9)))
        dense = np.column_stack((np.linspace(0.2, 2.2, 81), np.zeros(81)))
        sparse_target = PurePursuitController.interpolate_lookahead_point(
            sparse, 1.47
        )
        dense_target = PurePursuitController.interpolate_lookahead_point(
            dense, 1.47
        )
        np.testing.assert_allclose(sparse_target.point, dense_target.point)

    def test_short_path_clamps_to_endpoint(self):
        points = np.asarray([[0.2, 0.0], [0.7, 0.1]])
        target = PurePursuitController.interpolate_lookahead_point(
            points, 1.5
        )
        np.testing.assert_allclose(target.point, points[-1])
        self.assertTrue(target.clamped_to_endpoint)

    def test_three_point_curvature_for_two_meter_radius(self):
        radius = 2.0
        angles = np.asarray([0.0, 0.2, 0.4])
        points = np.column_stack(
            (radius * np.sin(angles), radius * (1.0 - np.cos(angles)))
        )
        curvature = PurePursuitController._three_point_curvature(*points)
        self.assertAlmostEqual(curvature, 1.0 / radius, places=5)

    def test_estimated_path_curvature_for_two_meter_arc(self):
        controller = make_controller()
        radius = 2.0
        angles = np.linspace(0.0, 0.9, 181)
        points = np.column_stack(
            (radius * np.sin(angles), radius * (1.0 - np.cos(angles)))
        )
        curvature = controller.estimate_path_curvature(points, 1.5)
        self.assertAlmostEqual(curvature, 0.5, delta=0.02)

    def test_curvature_reduces_lookahead_and_respects_minimum(self):
        controller = make_controller()
        self.assertAlmostEqual(
            controller._curvature_adjusted_lookahead(2.0, 0.25), 1.925
        )
        self.assertEqual(
            controller._curvature_adjusted_lookahead(1.1, 10.0), 1.1
        )

    def test_lookahead_filter_uses_elapsed_time(self):
        controller = make_controller(lookahead_filter_tau_sec=0.25)
        controller.filtered_lookahead_m = 2.0
        filtered = controller._filter_lookahead(1.1, 0.25)
        expected = 1.1 + 0.9 * math.exp(-1.0)
        self.assertAlmostEqual(filtered, expected)

    def test_steering_limiter_respects_rate_and_acceleration(self):
        controller = make_controller(
            max_steering_rate_deg_s=60.0,
            max_steering_accel_deg_s2=120.0,
        )
        dt = 0.1
        previous_angle = 0.0
        previous_rate = 0.0
        for _ in range(8):
            angle = controller._limit_steering_dynamics(18.0, dt)
            rate = controller.last_steering_rate_deg_s
            self.assertLessEqual(abs(rate), 60.0 + 1e-9)
            self.assertLessEqual(abs(rate - previous_rate), 12.0 + 1e-9)
            self.assertLessEqual(abs(angle - previous_angle), 6.0 + 1e-9)
            previous_angle = angle
            previous_rate = rate

    def test_steering_result_is_similar_at_30_and_60_hz(self):
        def simulate(rate_hz):
            controller = make_controller()
            dt = 1.0 / rate_hz
            for _ in range(int(0.6 * rate_hz)):
                controller._limit_steering_dynamics(18.0, dt)
            return controller.last_steering_deg

        self.assertAlmostEqual(simulate(30), simulate(60), delta=0.5)

    def test_invalid_or_long_dt_falls_back_to_nominal(self):
        controller = make_controller(nominal_control_rate_hz=30.0)
        nominal = 1.0 / 30.0
        for value in (None, 0.0, float("nan"), 0.5):
            controller.last_steering_rate_deg_s = 10.0
            with self.subTest(dt=value):
                self.assertAlmostEqual(
                    controller._normalize_control_dt(value), nominal
                )
                self.assertEqual(controller.last_steering_rate_deg_s, 0.0)
        self.assertEqual(controller._normalize_control_dt(0.001), 0.005)


if __name__ == "__main__":
    unittest.main()
