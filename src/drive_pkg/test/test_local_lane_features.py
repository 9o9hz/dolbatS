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
    @staticmethod
    def topology(left_x: float, dashed_x: float, right_x: float):
        left = group(left_x, False)
        dashed = group(dashed_x, True)
        right = group(right_x, False)
        left["pieces"][0].update(
            semantic_type="SOLID", semantic_confidence=0.95
        )
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED", semantic_confidence=0.95
            )
        right["pieces"][0].update(
            semantic_type="SOLID", semantic_confidence=0.95
        )
        return [left, dashed, right]

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

    def test_lane2_dashed_fallback_uses_strongest_not_nearest(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(warp_width=500, initial_lane="lane_2"),
        )
        stronger_far = group(80.0, True)
        weaker_near = group(220.0, True)
        for piece in stronger_far["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.99,
            )
        for piece in weaker_near["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.70,
            )

        selected_left, selected_right, mode, _ = (
            processor._choose_boundaries([stronger_far, weaker_near])
        )

        self.assertIs(selected_left, stronger_far)
        self.assertIsNone(selected_right)
        self.assertEqual(mode, "dashed_detected_no_solid")

    def test_boundary_reliability_uses_best_original_confidence(self):
        processor = SegmentationLaneProcessor(None, LaneConfig())
        contains_full_confidence = group(80.0, True)
        contains_full_confidence["pieces"][0].update(
            semantic_type="DASHED",
            semantic_confidence=1.0,
        )
        contains_full_confidence["pieces"][1].update(
            semantic_type="DASHED",
            semantic_confidence=0.40,
        )
        consistent_but_weaker = group(220.0, True)
        for piece in consistent_but_weaker["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.80,
            )

        self.assertGreater(
            processor._group_reliability(contains_full_confidence),
            processor._group_reliability(consistent_but_weaker),
        )

    def test_new_track_uses_strongest_not_nearest(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(warp_width=500),
        )
        stronger_far = group(80.0, False)
        weaker_near = group(220.0, False)
        stronger_far["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.99,
        )
        weaker_near["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.70,
        )

        selected_left, selected_right = processor._update_lane_tracks(
            [stronger_far, weaker_near]
        )

        self.assertIs(selected_left, stronger_far)
        self.assertIsNone(selected_right)

    def test_horizontal_crosswalk_candidate_is_rejected(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(crosswalk_max_aspect_ratio=1.5),
        )
        crosswalk = group(120.0, False)
        crosswalk["points"] = np.asarray(
            [[20.0, 400.0], [260.0, 410.0]], dtype=np.float32
        )

        self.assertEqual(
            processor._filter_crosswalk_candidates([crosswalk]), []
        )

    def test_longitudinal_lane_candidate_is_preserved(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(crosswalk_max_aspect_ratio=1.5),
        )
        lane = group(120.0, False)

        self.assertEqual(
            processor._filter_crosswalk_candidates([lane]), [lane]
        )

    def test_three_regular_lane_groups_keep_rightmost_edge(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                crosswalk_regular_min_groups=3,
                crosswalk_spacing_tolerance_ratio=0.25,
            ),
        )
        groups = [group(100.0, False), group(180.0, False), group(260.0, False)]

        self.assertEqual(
            processor._filter_crosswalk_candidates(groups), [groups[-1]]
        )

    def test_irregular_lane_groups_are_preserved(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                crosswalk_regular_min_groups=3,
                crosswalk_spacing_tolerance_ratio=0.25,
            ),
        )
        groups = [
            group(100.0, False),
            group(180.0, False),
            group(340.0, False),
        ]

        self.assertEqual(
            processor._filter_crosswalk_candidates(groups), groups
        )

    def test_regular_mixed_topology_keeps_rightmost_edge(self):
        processor = SegmentationLaneProcessor(None, LaneConfig())
        left = group(20.0, False)
        dashed = group(200.0, True)
        right = group(380.0, False)
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.9,
            )

        groups = [left, dashed, right]
        self.assertEqual(
            processor._filter_crosswalk_candidates(groups), [right]
        )

    def test_regular_dashed_cluster_is_filtered_beside_solid(self):
        processor = SegmentationLaneProcessor(None, LaneConfig())
        dashed_groups = [
            group(80.0, True),
            group(140.0, True),
            group(200.0, True),
        ]
        for dashed in dashed_groups:
            for piece in dashed["pieces"]:
                piece.update(
                    semantic_type="DASHED",
                    semantic_confidence=0.9,
                )
        solid = group(400.0, False)
        solid["pieces"][0].update(
            semantic_type="SOLID",
            semantic_confidence=0.95,
        )

        self.assertEqual(
            processor._filter_crosswalk_candidates(
                dashed_groups + [solid]
            ),
            [dashed_groups[-1], solid],
        )

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
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED",
                semantic_confidence=0.9,
            )

        selected_left, selected_right, mode, _ = (
            processor._choose_boundaries([left, dashed, right])
        )

        self.assertIs(selected_left, dashed)
        self.assertIs(selected_right, right)
        self.assertEqual(mode, "lane2_dashed_solid_hold")

    def test_confirmed_lane_can_change_with_stable_target_topology(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                pixels_per_meter_x=200.0,
                initial_lane="lane_2",
                lane_change_confirm_frames=2,
                lane_change_center_tolerance_m=0.15,
            ),
        )
        target_lane_1 = self.topology(160.0, 340.0, 520.0)

        self.assertIsNone(processor._classify_which_lane(target_lane_1))
        self.assertEqual(
            processor._lane_transition_state, "lane_2_to_lane_1"
        )
        self.assertEqual(
            processor._classify_which_lane(target_lane_1), "lane_1"
        )
        self.assertEqual(processor._current_lane, "lane_1")
        self.assertIsNone(processor._lane_transition_state)

    def test_lane_state_does_not_require_outer_solid_boundary(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                pixels_per_meter_x=200.0,
                lane_width_m=0.85,
                initial_lane="auto",
                lane_state_confirm_frames=1,
                lane_change_center_tolerance_m=0.15,
            ),
        )
        dashed = group(335.0, True)
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED", semantic_confidence=0.95
            )

        self.assertEqual(
            processor._classify_which_lane([dashed]), "lane_1"
        )
        self.assertEqual(processor._current_lane, "lane_1")

    def test_lane_state_uses_configured_vehicle_reference_x(self):
        processor = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=640,
                vehicle_reference_x_px=340.0,
                pixels_per_meter_x=200.0,
                initial_lane="auto",
                lane_state_confirm_frames=1,
                lane_change_center_tolerance_m=1.0,
            ),
        )
        dashed = group(330.0, True)
        for piece in dashed["pieces"]:
            piece.update(
                semantic_type="DASHED", semantic_confidence=0.95
            )

        self.assertEqual(
            processor._classify_which_lane([dashed]), "lane_2"
        )

    def test_center_line_dead_zone_can_be_enabled_or_disabled(self):
        topology = self.topology(80.0, 260.0, 440.0)
        enabled = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                pixels_per_meter_x=200.0,
                initial_lane="lane_2",
                lane_dead_zone_enabled=True,
                lane_dead_zone_m=0.12,
                lane_change_confirm_frames=1,
                lane_change_center_tolerance_m=0.5,
            ),
        )
        disabled = SegmentationLaneProcessor(
            None,
            LaneConfig(
                warp_width=500,
                pixels_per_meter_x=200.0,
                initial_lane="lane_2",
                lane_dead_zone_enabled=False,
                lane_dead_zone_m=0.12,
                lane_change_confirm_frames=1,
                lane_change_center_tolerance_m=0.5,
            ),
        )

        self.assertIsNone(enabled._classify_which_lane(topology))
        self.assertEqual(enabled._current_lane, "lane_2")
        self.assertEqual(
            disabled._classify_which_lane(topology), "lane_1"
        )

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
