import unittest

import numpy as np

from lane_detect import LaneDetectorCore
from lane_processing import LaneConfig, SegmentationLaneProcessor


def group(x_ref: float, dashed: bool) -> dict:
    return {
        "x_ref": x_ref,
        "pieces": [{}, {}] if dashed else [{}],
        "curve": (0.0, 0.0, x_ref),
        "points": np.asarray(
            [[x_ref, 180.0], [x_ref, 600.0]], dtype=np.float32
        ),
        "y_min": 180.0,
        "y_max": 600.0,
        "span": 420.0,
        "area": 1000,
    }


class LaneTopologyTest(unittest.TestCase):
    def test_component_points_use_zhang_suen_skeleton(self):
        component = np.zeros((20, 20), dtype=np.uint8)
        component[2:18, 6:14] = 1
        skeleton = SegmentationLaneProcessor._zhang_suen_thinning(
            component
        )
        points = SegmentationLaneProcessor._component_center_points(
            component, 1
        )

        self.assertGreater(np.count_nonzero(skeleton), 1)
        self.assertTrue(np.all(skeleton[:, :8] == 0))
        self.assertTrue(np.all(skeleton[:, 12:] == 0))
        self.assertIsNotNone(points)
        self.assertGreater(np.unique(points[:, 1]).size, 1)
        self.assertLessEqual(float(np.ptp(points[:, 0])), 1.0)

    def test_yolo_semantics_override_piece_count(self):
        processor = SegmentationLaneProcessor(None, LaneConfig())
        solid_split = group(120.0, True)
        for piece in solid_split["pieces"]:
            piece.update(
                semantic_type="SOLID",
                semantic_confidence=0.91,
            )
        dashed_single = group(390.0, False)
        dashed_single["pieces"][0].update(
            semantic_type="DASHED",
            semantic_confidence=0.87,
        )

        solid_type, solid_confidence, solid_source = (
            processor._group_type(solid_split)
        )
        dashed_type, dashed_confidence, dashed_source = (
            processor._group_type(dashed_single)
        )

        self.assertEqual(solid_type, "SOLID")
        self.assertAlmostEqual(solid_confidence, 0.91)
        self.assertEqual(solid_source, "yolo")
        self.assertEqual(dashed_type, "DASHED")
        self.assertAlmostEqual(dashed_confidence, 0.87)
        self.assertEqual(dashed_source, "yolo")

    def test_too_close_pair_keeps_higher_yolo_confidence(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                pixels_per_meter_x=100.0,
                lane_width_m=0.85,
                min_boundary_spacing_m=0.80,
            ),
        )
        left = group(120.0, False)
        right = group(190.0, False)
        left["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.90,
        )
        right["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.70,
        )

        selected_left, selected_right = (
            processor._filter_boundary_spacing(left, right)
        )

        self.assertIs(selected_left, left)
        self.assertIsNone(selected_right)

    def test_too_far_pair_keeps_higher_yolo_confidence(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                pixels_per_meter_x=100.0,
                lane_width_m=0.85,
                min_boundary_spacing_m=0.80,
                max_boundary_spacing_m=0.90,
            ),
        )
        left = group(100.0, False)
        right = group(200.0, False)
        left["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.60,
        )
        right["pieces"][0].update(
            semantic_type="DASHED",
            semantic_confidence=0.95,
        )

        selected_left, selected_right = (
            processor._filter_boundary_spacing(left, right)
        )

        self.assertIsNone(selected_left)
        self.assertIs(selected_right, right)

    def test_invalid_topology_uses_stronger_yolo_line(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                pixels_per_meter_x=100.0,
                lane_width_m=0.85,
                min_boundary_spacing_m=0.80,
                max_boundary_spacing_m=0.90,
                initial_lane="lane_2",
            ),
        )
        dashed = group(200.0, True)
        solid = group(270.0, False)
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.72,
            )
        solid["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.94,
        )

        selected_left, selected_right, mode, _ = (
            processor._choose_boundaries([dashed, solid])
        )

        self.assertIsNone(selected_left)
        self.assertIs(selected_right, solid)
        self.assertEqual(mode, "invalid_spacing_yolo_confidence")

    def test_single_boundary_does_not_require_spacing(self):
        processor = SegmentationLaneProcessor(None, LaneConfig())
        left = group(120.0, False)

        selected_left, selected_right = (
            processor._filter_boundary_spacing(left, None)
        )

        self.assertIs(selected_left, left)
        self.assertIsNone(selected_right)

    def test_solid_dashed_solid_respects_latched_lane(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                warp_height=640,
                pixels_per_meter_x=200.0,
                initial_lane="lane_2",
            ),
        )
        left = group(20.0, False)
        dashed = group(200.0, True)
        right = group(380.0, False)

        selected_left, selected_right, mode, _ = (
            processor._choose_boundaries([left, dashed, right])
        )

        self.assertIs(selected_left, dashed)
        self.assertIs(selected_right, right)
        self.assertEqual(mode, "lane2_dashed_solid_hold")

    def test_current_boundary_is_not_mixed_with_stale_boundary(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                warp_height=640,
                initial_lane="auto",
                prefer_solid_when_dashed=False,
            ),
        )
        left = group(120.0, False)
        right = group(390.0, False)
        processor._choose_boundaries([left, right])

        selected_left, selected_right, _, _ = (
            processor._choose_boundaries([right])
        )

        self.assertIsNone(selected_left)
        self.assertIs(selected_right, right)

    def test_no_current_boundary_defers_to_path_fallback(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                warp_height=640,
                initial_lane="auto",
                prefer_solid_when_dashed=False,
            ),
        )
        processor._choose_boundaries(
            [group(120.0, False), group(390.0, False)]
        )

        selected_left, selected_right, _, _ = (
            processor._choose_boundaries([])
        )

        self.assertIsNone(selected_left)
        self.assertIsNone(selected_right)

    def test_local_path_is_bspline_and_anchored_to_vehicle(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                warp_height=640,
                path_spline_smooth_factor=10.0,
                path_spline_points=100,
            ),
        )
        raw_path, _ = processor._build_path(
            group(120.0, False),
            group(390.0, False),
        )
        path = processor._anchor_path_to_vehicle_center(
            processor._smooth_spatial(raw_path)
        )
        nearest = path[int(np.argmax(path[:, 1]))]

        self.assertAlmostEqual(nearest[0], 250.0)
        self.assertAlmostEqual(nearest[1], 639.0)


class LineWidthFilterTest(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetectorCore.__new__(LaneDetectorCore)
        self.detector.pixels_per_meter_x = 254.0
        self.detector.line_width_target_m = 0.050
        self.detector.line_width_tolerance_m = 0.010
        self.detector.line_width_measurement_scale = 0.80

    @staticmethod
    def component(width_px: int) -> np.ndarray:
        component = np.zeros((100, 100), dtype=np.uint8)
        component[:, 40 : 40 + width_px] = 1
        return component

    def test_accepts_fifty_mm_line(self):
        width = self.detector._component_width_m(self.component(16))
        self.assertTrue(self.detector._width_is_lane(width))

    def test_rejects_thick_marking(self):
        width = self.detector._component_width_m(self.component(25))
        self.assertFalse(self.detector._width_is_lane(width))


class DuplicateInstanceTest(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetectorCore.__new__(LaneDetectorCore)
        self.detector.pixels_per_meter_x = 254.0
        self.detector.line_width_target_m = 0.050
        self.detector.line_width_recovery_tolerance_m = 0.015
        self.detector.line_width_measurement_scale = 0.80

    @staticmethod
    def instance(
        x_start: int,
        confidence: float,
        class_name: str,
    ) -> dict:
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[10:110, x_start : x_start + 14] = 1
        return {
            "class_name": class_name,
            "confidence": confidence,
            "x_min": x_start,
            "y_min": 10,
            "x_max": x_start + 13,
            "y_max": 109,
            "mask": mask,
        }

    def test_same_line_class_conflict_keeps_higher_confidence(self):
        solid = self.instance(60, 0.91, "SOLID")
        dashed = self.instance(62, 0.73, "DASHED")

        kept = self.detector._deduplicate_instances([dashed, solid])

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["class_name"], "SOLID")
        self.assertEqual(kept[0]["confidence"], 0.91)

    def test_parallel_lines_are_not_suppressed(self):
        left = self.instance(25, 0.90, "SOLID")
        right = self.instance(115, 0.88, "DASHED")

        kept = self.detector._deduplicate_instances([left, right])

        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
