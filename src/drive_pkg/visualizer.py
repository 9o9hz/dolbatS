#!/usr/bin/env python3
"""Combined cv2 visualizer for segmentation, lane lines, and path.

Extracted from what used to be a standalone ``path_visualizer`` ROS node so
``pure_pursuit.py`` can own it directly (main5.py-style local cv2 window,
no extra process) -- see howtorun.md. ``DrivingVisualizer`` itself has no
ROS Node dependency; ``pure_pursuit.PurePursuitNode`` wires ROS
subscriptions directly to its ``on_*`` methods and forwards its own
in-process control status.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


LOW_LATENCY_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


@dataclass
class LineTrack:
    track_id: int
    box: np.ndarray
    line_type: str
    history: list[str] = field(default_factory=list)
    candidate_type: str | None = None
    candidate_streak: int = 0
    hits: int = 1
    missed: int = 0


class StableLineTracker:
    """Smooth boxes and apply hysteresis to DASHED/SOLID classifications."""

    HISTORY_SIZE = 20

    def __init__(
        self,
        box_ema_alpha: float,
        type_switch_frames: int,
        max_missed_frames: int,
        match_distance_px: float,
        confidence_full_hits: int,
    ) -> None:
        self.box_ema_alpha = float(np.clip(box_ema_alpha, 0.01, 1.0))
        self.type_switch_frames = max(1, int(type_switch_frames))
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.match_distance_px = max(1.0, float(match_distance_px))
        self.confidence_full_hits = max(1, int(confidence_full_hits))
        self.tracks: list[LineTrack] = []
        self.next_track_id = 1

    @staticmethod
    def observation_box(observation: dict) -> np.ndarray:
        return np.asarray(
            [
                observation["x_min"],
                observation["y_min"],
                observation["x_max"],
                observation["y_max"],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def box_center(box: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            ],
            dtype=np.float32,
        )

    def create_track(self, observation: dict) -> LineTrack:
        line_type = str(observation["type"])
        track = LineTrack(
            track_id=self.next_track_id,
            box=self.observation_box(observation),
            line_type=line_type,
            history=[line_type],
        )
        self.next_track_id += 1
        return track

    def update_track(self, track: LineTrack, observation: dict) -> None:
        observed_box = self.observation_box(observation)
        alpha = self.box_ema_alpha
        track.box = (1.0 - alpha) * track.box + alpha * observed_box
        track.hits += 1
        track.missed = 0

        observed_type = str(observation["type"])
        track.history.append(observed_type)
        track.history = track.history[-self.HISTORY_SIZE :]
        if observed_type == track.line_type:
            track.candidate_type = None
            track.candidate_streak = 0
            return

        if observed_type == track.candidate_type:
            track.candidate_streak += 1
        else:
            track.candidate_type = observed_type
            track.candidate_streak = 1

        if track.candidate_streak >= self.type_switch_frames:
            track.line_type = observed_type
            track.candidate_type = None
            track.candidate_streak = 0

    def confidence(self, track: LineTrack) -> int:
        matching = sum(
            observed_type == track.line_type
            for observed_type in track.history
        )
        agreement = matching / max(1, len(track.history))
        warmup = min(1.0, track.hits / self.confidence_full_hits)
        visibility = max(
            0.0,
            1.0 - track.missed / (self.max_missed_frames + 1.0),
        )
        return int(round(100.0 * agreement * warmup * visibility))

    def display_tracks(self) -> list[dict]:
        return [
            {
                "track_id": track.track_id,
                "type": track.line_type,
                "confidence": self.confidence(track),
                "x_min": float(track.box[0]),
                "y_min": float(track.box[1]),
                "x_max": float(track.box[2]),
                "y_max": float(track.box[3]),
            }
            for track in self.tracks
        ]

    def update(self, observations: list[dict]) -> list[dict]:
        pairs = []
        for track_index, track in enumerate(self.tracks):
            track_center = self.box_center(track.box)
            for observation_index, observation in enumerate(observations):
                observation_center = self.box_center(
                    self.observation_box(observation)
                )
                distance = float(
                    np.linalg.norm(track_center - observation_center)
                )
                if distance <= self.match_distance_px:
                    pairs.append(
                        (distance, track_index, observation_index)
                    )

        matched_tracks = set()
        matched_observations = set()
        for _, track_index, observation_index in sorted(pairs):
            if (
                track_index in matched_tracks
                or observation_index in matched_observations
            ):
                continue
            self.update_track(
                self.tracks[track_index],
                observations[observation_index],
            )
            matched_tracks.add(track_index)
            matched_observations.add(observation_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed += 1

        for observation_index, observation in enumerate(observations):
            if observation_index not in matched_observations:
                self.tracks.append(self.create_track(observation))

        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.max_missed_frames
        ]
        return self.display_tracks()


class DrivingVisualizer:
    """Owns the combined cv2 window; fed by ROS subscription callbacks
    (segmentation/BEV/debug/instances/path status) and by a direct
    in-process call for control status (see ``pure_pursuit.py``)."""

    def __init__(
        self,
        window_name: str,
        window_x: int,
        window_y: int,
        display_scale: float,
        box_ema_alpha: float,
        type_switch_frames: int,
        track_max_missed_frames: int,
        track_match_distance_px: float,
        confidence_full_hits: int,
        yolo_confidence_aggregation: str,
        lookahead_target_hold_sec: float,
        reference_path_hold_sec: float,
        logger: Optional[object] = None,
    ) -> None:
        self.window_name = str(window_name)
        self.window_x = int(window_x)
        self.window_y = int(window_y)
        self.display_scale = max(0.1, float(display_scale))
        self.lookahead_target_hold_sec = max(
            0.0, float(lookahead_target_hold_sec)
        )
        self.reference_path_hold_sec = max(
            0.0, float(reference_path_hold_sec)
        )
        self.yolo_confidence_aggregation = str(
            yolo_confidence_aggregation
        ).strip().lower()
        if self.yolo_confidence_aggregation not in {"mean", "max"}:
            raise ValueError(
                "yolo_confidence_aggregation must be 'mean' or 'max'"
            )
        self.logger = logger
        self.line_tracker = StableLineTracker(
            box_ema_alpha=box_ema_alpha,
            type_switch_frames=type_switch_frames,
            max_missed_frames=track_max_missed_frames,
            match_distance_px=track_match_distance_px,
            confidence_full_hits=confidence_full_hits,
        )
        self.latest_segmentation: np.ndarray | None = None
        self.latest_bev: np.ndarray | None = None
        self.latest_path_debug: np.ndarray | None = None
        self.detected_lines: list[dict] = []
        self.path_status: dict = {}
        self.control_status: dict = {}
        self.yolo_instances: list[dict] = []
        self.last_lookahead_target: tuple[int, int] | None = None
        self.last_lookahead_target_at: float | None = None
        self.latest_control_has_target = False
        self.last_reference_path: np.ndarray | None = None
        self.last_reference_path_at: float | None = None
        self.latest_path_has_points = False
        # If the user closes the window from the window manager (or the
        # X/Qt backend otherwise drops it), OpenCV's Python-side handle is
        # left dangling and the next resizeWindow/imshow/destroyWindow call
        # on it raises cv2.error ("NULL guiReceiver"). That must never take
        # down this whole process -- it also publishes steering/throttle
        # commands. Once the window is gone, stop touching it instead of
        # crashing.
        self._window_broken = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.window_x, self.window_y)

    def decode_image(
        self,
        message: CompressedImage,
        label: str,
    ) -> np.ndarray | None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            if self.logger is not None:
                self.logger.warning(
                    f"Could not decode {label} image",
                    throttle_duration_sec=2.0,
                )
            return None
        return image

    def on_segmentation_image(self, message: CompressedImage) -> None:
        image = self.decode_image(message, "segmentation")
        if image is None:
            return
        self.latest_segmentation = image
        self.show_combined()

    def on_bev_image(self, message: CompressedImage) -> None:
        image = self.decode_image(message, "BEV")
        if image is None:
            return
        self.latest_bev = image
        self.show_combined()

    def on_debug_image(self, message: CompressedImage) -> None:
        image = self.decode_image(message, "path debug")
        if image is None:
            return
        self.latest_path_debug = image
        self.show_combined()

    def on_path_status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
            lines = status.get("detected_lines", [])
            if not isinstance(lines, list):
                raise TypeError("detected_lines must be a list")
            self.path_status = status
            path_points = self.status_points(status, "path_pixels")
            self.latest_path_has_points = path_points is not None
            if status.get("path_valid") is False:
                self.clear_reference_path()
                self.clear_lookahead_target()
            elif path_points is not None:
                self.last_reference_path = path_points.copy()
                self.last_reference_path_at = time.monotonic()
            observations = [
                line
                for line in lines
                if (
                    isinstance(line, dict)
                    and line.get("type") in {"DASHED", "SOLID"}
                    and all(
                        isinstance(line.get(key), (int, float))
                        for key in (
                            "x_min",
                            "y_min",
                            "x_max",
                            "y_max",
                        )
                    )
                )
            ]
            self.detected_lines = self.line_tracker.update(observations)
            self.show_combined()
        except (json.JSONDecodeError, TypeError) as error:
            if self.logger is not None:
                self.logger.warning(
                    f"Could not parse path status labels: {error}",
                    throttle_duration_sec=2.0,
                )

    def on_control_status(self, status: dict) -> None:
        """Fed directly from ``PurePursuitNode._publish_command`` -- no
        topic round trip, since control status is this process's own
        state now."""
        self.control_status = status
        if status.get("path_valid") is False:
            self.clear_lookahead_target()
            self.clear_reference_path()
        else:
            self.update_lookahead_target(status)
        self.show_combined()

    def on_yolo_instances(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            instances = payload.get("instances", [])
            if not isinstance(instances, list):
                raise TypeError("instances must be a list")
            self.yolo_instances = [
                instance
                for instance in instances
                if (
                    isinstance(instance, dict)
                    and isinstance(
                        instance.get("confidence"),
                        (int, float),
                    )
                    and all(
                        isinstance(instance.get(key), (int, float))
                        for key in (
                            "x_min",
                            "y_min",
                            "x_max",
                            "y_max",
                        )
                    )
                )
            ]
            self.show_combined()
        except (json.JSONDecodeError, TypeError) as error:
            if self.logger is not None:
                self.logger.warning(
                    f"Could not parse YOLO instances: {error}",
                    throttle_duration_sec=2.0,
                )

    @staticmethod
    def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
        if image.shape[0] == height:
            return image
        scale = height / float(image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def add_header(image: np.ndarray, title: str) -> np.ndarray:
        panel = cv2.copyMakeBorder(
            image,
            44,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )
        cv2.putText(
            panel,
            title,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return panel

    @staticmethod
    def draw_line_type_labels(
        image: np.ndarray,
        detected_lines: list[dict],
    ) -> np.ndarray:
        labeled = image.copy()
        height, width = labeled.shape[:2]
        for line in detected_lines:
            line_type = str(line["type"])
            confidence = int(line.get("confidence", 0))
            label = f"{line_type} {confidence}%"
            x_min = int(np.clip(
                round(float(line["x_min"])),
                0,
                max(0, width - 2),
            ))
            y_min = int(np.clip(
                round(float(line["y_min"])),
                0,
                max(0, height - 2),
            ))
            x_max = int(np.clip(
                round(float(line["x_max"])),
                x_min + 1,
                max(x_min + 1, width - 1),
            ))
            y_max = int(np.clip(
                round(float(line["y_max"])),
                y_min + 1,
                max(y_min + 1, height - 1),
            ))
            color = (
                (0, 255, 255)
                if line_type == "DASHED"
                else (255, 255, 0)
            )
            cv2.rectangle(
                labeled,
                (x_min, y_min),
                (x_max, y_max),
                color,
                3,
                cv2.LINE_AA,
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2,
            )
            label_top = max(0, y_min - text_height - baseline - 8)
            label_right = min(
                width - 1,
                x_min + text_width + 12,
            )
            cv2.rectangle(
                labeled,
                (x_min, label_top),
                (label_right, y_min),
                color,
                -1,
            )
            cv2.putText(
                labeled,
                label,
                (x_min + 6, max(text_height + 2, y_min - baseline - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return labeled

    @classmethod
    def compose_panels(
        cls,
        segmentation: np.ndarray,
        path_debug: np.ndarray,
        detected_lines: list[dict] | None = None,
        driving_detail: np.ndarray | None = None,
    ) -> np.ndarray:
        if driving_detail is None:
            driving_detail = np.zeros_like(segmentation)
        panel_height = min(
            segmentation.shape[0],
            driving_detail.shape[0],
            path_debug.shape[0],
        )
        segmentation = cls.draw_line_type_labels(
            segmentation,
            detected_lines or [],
        )
        left = cls.resize_to_height(segmentation, panel_height)
        middle = cls.resize_to_height(driving_detail, panel_height)
        right = cls.resize_to_height(path_debug, panel_height)
        left = cls.add_header(left, "SEGMENTATION")
        middle = cls.add_header(middle, "BEV PATH + CONTROL")
        right = cls.add_header(
            right,
            "DETECTED LINES + REFERENCE PATH",
        )
        separator = np.full(
            (left.shape[0], 4, 3),
            (210, 210, 210),
            dtype=np.uint8,
        )
        return cv2.hconcat(
            [left, separator, middle, separator.copy(), right]
        )

    def show_combined(self) -> None:
        if self._window_broken:
            return
        if (
            self.latest_segmentation is None
            or self.latest_bev is None
            or self.latest_path_debug is None
        ):
            return

        reference_path, path_is_held = self.resolve_reference_path()
        lookahead_target, target_is_held = (
            self.resolve_lookahead_target()
        )
        driving_detail = self.render_driving_detail(
            self.latest_bev,
            self.path_status,
            self.control_status,
            self.detected_lines,
            self.yolo_instances,
            self.yolo_confidence_aggregation,
            lookahead_target,
            target_is_held,
            reference_path,
            path_is_held,
        )
        combined = self.compose_panels(
            self.latest_segmentation,
            self.latest_path_debug,
            self.detected_lines,
            driving_detail,
        )
        preview = cv2.resize(
            combined,
            None,
            fx=self.display_scale,
            fy=self.display_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        try:
            cv2.resizeWindow(
                self.window_name,
                preview.shape[1],
                preview.shape[0],
            )
            cv2.imshow(self.window_name, preview)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            self._window_broken = True
            if self.logger is not None:
                self.logger.warning(
                    "Visualization window is gone (closed by the user or "
                    f"the window manager); disabling display: {error}"
                )
            return
        if key in (27, ord("q")):
            rclpy.shutdown()

    def clear_lookahead_target(self) -> None:
        self.last_lookahead_target = None
        self.last_lookahead_target_at = None
        self.latest_control_has_target = False

    def clear_reference_path(self) -> None:
        self.last_reference_path = None
        self.last_reference_path_at = None
        self.latest_path_has_points = False

    def update_lookahead_target(self, status: dict) -> None:
        path_points = self.last_reference_path
        try:
            target_index = int(
                status.get("lookahead_target_index", -1)
            )
        except (TypeError, ValueError):
            target_index = -1

        self.latest_control_has_target = bool(
            path_points is not None
            and 0 <= target_index < len(path_points)
        )
        if not self.latest_control_has_target:
            return

        target = path_points[target_index]
        self.last_lookahead_target = (
            int(target[0]),
            int(target[1]),
        )
        self.last_lookahead_target_at = time.monotonic()

    def resolve_lookahead_target(
        self,
    ) -> tuple[tuple[int, int] | None, bool]:
        """Hold brief dropouts, but never retain an invalid stale target."""
        if (
            self.path_status.get("path_valid") is False
            or self.control_status.get("path_valid") is False
        ):
            self.clear_lookahead_target()
            return None, False

        if (
            self.last_lookahead_target is None
            or self.last_lookahead_target_at is None
        ):
            return None, False

        age = time.monotonic() - self.last_lookahead_target_at
        if age > self.lookahead_target_hold_sec:
            self.clear_lookahead_target()
            return None, False
        return (
            self.last_lookahead_target,
            not self.latest_control_has_target,
        )

    def resolve_reference_path(
        self,
    ) -> tuple[np.ndarray | None, bool]:
        """Hold a valid red path only across a brief missing-frame gap."""
        if (
            self.path_status.get("path_valid") is False
            or self.control_status.get("path_valid") is False
        ):
            self.clear_reference_path()
            return None, False

        if (
            self.last_reference_path is None
            or self.last_reference_path_at is None
        ):
            return None, False

        age = time.monotonic() - self.last_reference_path_at
        if age > self.reference_path_hold_sec:
            self.clear_reference_path()
            return None, False
        return (
            self.last_reference_path.copy(),
            not self.latest_path_has_points,
        )

    @staticmethod
    def status_points(status: dict, key: str) -> np.ndarray | None:
        raw_points = status.get(key, [])
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            return None
        points = np.asarray(raw_points, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            return None
        return np.rint(points).astype(np.int32)

    @staticmethod
    def box_overlap_ratio(first: dict, second: dict) -> float:
        x_min = max(float(first["x_min"]), float(second["x_min"]))
        y_min = max(float(first["y_min"]), float(second["y_min"]))
        x_max = min(float(first["x_max"]), float(second["x_max"]))
        y_max = min(float(first["y_max"]), float(second["y_max"]))
        intersection = max(0.0, x_max - x_min) * max(
            0.0,
            y_max - y_min,
        )
        first_area = max(
            1.0,
            (float(first["x_max"]) - float(first["x_min"]))
            * (float(first["y_max"]) - float(first["y_min"])),
        )
        second_area = max(
            1.0,
            (float(second["x_max"]) - float(second["x_min"]))
            * (float(second["y_max"]) - float(second["y_min"])),
        )
        return intersection / min(first_area, second_area)

    @classmethod
    def aggregate_yolo_confidences(
        cls,
        detected_lines: list[dict],
        instances: list[dict],
        aggregation: str,
    ) -> list[dict]:
        results = []
        for line in detected_lines:
            matched = [
                float(instance["confidence"])
                for instance in instances
                if cls.box_overlap_ratio(line, instance) >= 0.05
            ]
            if not matched:
                continue
            confidence = (
                max(matched)
                if aggregation == "max"
                else float(np.mean(matched))
            )
            results.append(
                {
                    **line,
                    "yolo_confidence": float(
                        np.clip(confidence, 0.0, 1.0)
                    ),
                    "matched_masks": len(matched),
                }
            )
        return results

    @staticmethod
    def draw_yolo_confidences(
        image: np.ndarray,
        aggregated_lines: list[dict],
        aggregation: str,
    ) -> None:
        height, width = image.shape[:2]
        for line in aggregated_lines:
            x = int(np.clip(
                round(float(line["x_min"])),
                0,
                max(0, width - 230),
            ))
            y = int(np.clip(
                round(
                    (
                        float(line["y_min"])
                        + float(line["y_max"])
                    )
                    * 0.5
                ),
                175,
                max(175, height - 10),
            ))
            confidence_percent = 100.0 * float(
                line["yolo_confidence"]
            )
            label = (
                f"YOLO {aggregation.upper()} "
                f"{confidence_percent:.1f}% "
                f"(n={line['matched_masks']})"
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )
            cv2.rectangle(
                image,
                (x, y - text_height - baseline - 8),
                (
                    min(width - 1, x + text_width + 12),
                    y + 4,
                ),
                (255, 255, 255),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x + 6, y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    @classmethod
    def render_driving_detail(
        cls,
        bev: np.ndarray,
        path_status: dict,
        control_status: dict,
        detected_lines: list[dict],
        yolo_instances: list[dict],
        yolo_aggregation: str,
        lookahead_target: tuple[int, int] | None = None,
        target_is_held: bool = False,
        reference_path: np.ndarray | None = None,
        path_is_held: bool = False,
    ) -> np.ndarray:
        detail = bev.copy()
        overlays = (
            ("left_boundary_pixels", (255, 0, 0), 3),
            ("right_boundary_pixels", (0, 255, 255), 3),
        )
        for key, color, thickness in overlays:
            points = cls.status_points(path_status, key)
            if points is not None:
                cv2.polylines(
                    detail,
                    [points],
                    False,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        if reference_path is not None:
            cv2.polylines(
                detail,
                [reference_path],
                False,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

        if lookahead_target is not None:
            target = lookahead_target
            cv2.circle(detail, target, 11, (255, 255, 255), -1)
            cv2.circle(detail, target, 11, (0, 0, 0), 2)
            cv2.putText(
                detail,
                (
                    "LOOK-AHEAD TARGET (HOLD)"
                    if target_is_held
                    else "LOOK-AHEAD TARGET"
                ),
                (target[0] + 14, target[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        info_overlay = detail.copy()
        cv2.rectangle(
            info_overlay,
            (0, 0),
            (detail.shape[1] - 1, 152),
            (0, 0, 0),
            -1,
        )
        detail = cv2.addWeighted(
            info_overlay,
            0.72,
            detail,
            0.28,
            0.0,
        )
        steering_deg = float(control_status.get("steering_deg", 0.0))
        lookahead_m = float(control_status.get("lookahead_m", 0.0))
        target_m = float(
            control_status.get("lookahead_target_m", 0.0)
        )
        info_lines = (
            f"STEERING: {steering_deg:+.2f} deg",
            (
                f"DYNAMIC LOOK-AHEAD: {lookahead_m:.2f} m  "
                f"TARGET: {target_m:.2f} m"
            ),
            (
                "BLUE: LEFT  YELLOW: RIGHT  "
                f"RED: PATH{' (HOLD)' if path_is_held else ''}  "
                "WHITE: TARGET"
            ),
            (
                "YOLO ORIGINAL CONFIDENCE: "
                f"{yolo_aggregation.upper()} per matched line"
            ),
        )
        colors = (
            (255, 255, 255),
            (255, 255, 255),
            (0, 255, 255),
            (0, 255, 0),
        )
        for index, (text, color) in enumerate(zip(info_lines, colors)):
            cv2.putText(
                detail,
                text,
                (14, 30 + index * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        aggregated_lines = cls.aggregate_yolo_confidences(
            detected_lines,
            yolo_instances,
            yolo_aggregation,
        )
        cls.draw_yolo_confidences(
            detail,
            aggregated_lines,
            yolo_aggregation,
        )
        return detail

    def destroy(self) -> None:
        if self._window_broken:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
