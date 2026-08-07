#!/usr/bin/env python3
"""ROS 2 node: nav_msgs/Path -> lane control candidate topics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Optional, Sequence

from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String

from visualizer import LOW_LATENCY_QOS, DrivingVisualizer


PARAMETER_DEFAULTS = {
    "path_topic": "/lane/path",
    "path_status_topic": "/lane/path/status",
    "steer_angle_topic": "/control/candidate/lane/steer_angle",
    "candidate_valid_topic": "/control/candidate/lane/valid",
    "throttle_feedback_topic": "/auto_throttle",
    "status_topic": "/lane/control/status",
    "wheelbase_m": 0.545,
    "ld_throttle_min": 0.4,
    "ld_throttle_max": 0.8,
    "dynamic_lookahead_enabled": True,
    "lookahead_min_m": 1.10,
    "lookahead_max_m": 2.00,
    "lookahead_m": 1.50,
    "minimum_path_preview_m": 0.30,
    "max_steer_deg": 26.5,
    "steering_gain": 1.80,
    "steering_ema_alpha": 0.30,
    "steering_deadband_deg": 0.8,
    "max_steering_change_deg": 5.0,
    "curvature_lookahead_enabled": True,
    "curvature_full_scale_1pm": 0.50,
    "curvature_reduction_max_m": 0.15,
    "curvature_sample_gap_m": 0.15,
    "curvature_tracking_enabled": True,
    "curvature_tracking_gain": 0.20,
    "curvature_tracking_sample_gap_m": 0.15,
    "curvature_tracking_preview_m": 0.45,
    "curvature_tracking_min_samples": 3,
    "curvature_tracking_max_mad_1pm": 0.25,
    "curvature_tracking_max_correction_1pm": 0.20,
    "curvature_tracking_sign_guard_1pm": 0.05,
    "curvature_tracking_min_deficit_1pm": 0.01,
    "lookahead_filter_tau_sec": 0.25,
    "time_based_steering_limit_enabled": True,
    "nominal_control_rate_hz": 30.0,
    "min_control_dt_sec": 0.005,
    "max_control_dt_sec": 0.20,
    "max_steering_rate_deg_s": 75.0,
    "max_steering_accel_deg_s2": 300.0,
    # Legacy nearest-index modes remain available for rollback. The default
    # interpolates inside the segment where cumulative arc length reaches Ld.
    "lookahead_search_mode": "continuous_arc_length",
    # main5.py-style local cv2 window (segmentation + BEV control + lines/
    # path), merged in-process from what used to be the separate
    # path_visualizer node. Set to false for headless runs.
    "local_display": True,
    "segmentation_topic": "/lane/detection/segmentation/compressed",
    "bev_topic": "/lane/detection/bev/compressed",
    "debug_topic": "/lane/path/debug/compressed",
    "instances_topic": "/lane/detection/instances",
    "window_name": (
        "drive visualizer: segmentation | BEV control | lines + path"
    ),
    "window_x": 20,
    "window_y": 60,
    "display_scale": 0.90,
    "box_ema_alpha": 0.15,
    "type_switch_frames": 5,
    "track_max_missed_frames": 12,
    "track_match_distance_px": 140.0,
    "confidence_full_hits": 5,
    "yolo_confidence_aggregation": "mean",
    "lookahead_target_hold_sec": 0.45,
    "reference_path_hold_sec": 0.45,
}


@dataclass
class SteeringCommand:
    """Result of one Pure Pursuit control step."""

    path_valid: bool
    reason: str
    steering_deg: float
    raw_steering_deg: float
    steering_rate_deg_s: float
    lookahead_m: float
    lookahead_reference_m: float
    target_search_lookahead_m: float
    target_path_preview_m: float
    path_curvature_1pm: float
    pure_pursuit_curvature_1pm: float
    target_path_curvature_1pm: float
    target_path_curvature_samples: int
    target_path_curvature_mad_1pm: float
    curvature_tracking_error_1pm: float
    curvature_tracking_correction_1pm: float
    curvature_tracking_applied: bool
    curvature_tracking_reason: str
    target_distance_m: float
    target_index: int
    target_upper_index: int
    target_segment_ratio: float
    target_forward_m: float
    target_left_m: float
    target_clamped_to_endpoint: bool
    control_dt_sec: float


@dataclass
class LookaheadTarget:
    """Continuous target interpolated on one reference-path segment."""

    point: np.ndarray
    lower_index: int
    upper_index: int
    segment_ratio: float
    clamped_to_endpoint: bool


class PurePursuitController:
    """Pure Pursuit steering math, independent of ROS.

    Extracted out of ``PurePursuitNode`` so the same steering logic can be
    called directly (no topic round trip) from an integrated node such as
    ``drive_main.LaneDriveNode``, while ``PurePursuitNode`` keeps working
    unchanged as a thin ROS wrapper around it.
    """

    def __init__(
        self,
        wheelbase_m: float,
        ld_throttle_min: float,
        ld_throttle_max: float,
        lookahead_min_m: float,
        lookahead_max_m: float,
        fixed_lookahead_m: float,
        dynamic_lookahead_enabled: bool,
        max_steer_deg: float,
        steering_gain: float,
        steering_ema_alpha: float,
        steering_deadband_deg: float,
        max_steering_change_deg: float,
        lookahead_search_mode: str = "simple",
        curvature_lookahead_enabled: bool = True,
        curvature_full_scale_1pm: float = 0.50,
        curvature_reduction_max_m: float = 0.15,
        curvature_sample_gap_m: float = 0.15,
        lookahead_filter_tau_sec: float = 0.25,
        time_based_steering_limit_enabled: bool = True,
        nominal_control_rate_hz: float = 30.0,
        min_control_dt_sec: float = 0.005,
        max_control_dt_sec: float = 0.20,
        max_steering_rate_deg_s: float = 75.0,
        max_steering_accel_deg_s2: float = 300.0,
        minimum_path_preview_m: float = 0.30,
        curvature_tracking_enabled: bool = True,
        curvature_tracking_gain: float = 0.20,
        curvature_tracking_sample_gap_m: float = 0.15,
        curvature_tracking_preview_m: float = 0.45,
        curvature_tracking_min_samples: int = 3,
        curvature_tracking_max_mad_1pm: float = 0.25,
        curvature_tracking_max_correction_1pm: float = 0.20,
        curvature_tracking_sign_guard_1pm: float = 0.05,
        curvature_tracking_min_deficit_1pm: float = 0.01,
    ) -> None:
        self.wheelbase_m = float(wheelbase_m)
        self.ld_throttle_min = float(ld_throttle_min)
        self.ld_throttle_max = float(ld_throttle_max)
        if self.ld_throttle_max < self.ld_throttle_min:
            self.ld_throttle_min, self.ld_throttle_max = (
                self.ld_throttle_max,
                self.ld_throttle_min,
            )
        self.current_throttle = self.ld_throttle_min
        self.lookahead_min_m = float(lookahead_min_m)
        self.lookahead_max_m = float(lookahead_max_m)
        self.dynamic_lookahead_enabled = bool(
            dynamic_lookahead_enabled
        )
        if not self.dynamic_lookahead_enabled:
            if float(fixed_lookahead_m) <= 0.0:
                raise ValueError(
                    "lookahead_m must be positive when dynamic "
                    "lookahead is disabled"
                )
            self.lookahead_min_m = float(fixed_lookahead_m)
            self.lookahead_max_m = float(fixed_lookahead_m)
        self.max_steer_deg = float(max_steer_deg)
        self.steering_gain = float(steering_gain)
        self.steering_ema_alpha = float(steering_ema_alpha)
        self.steering_deadband_deg = float(steering_deadband_deg)
        self.max_steering_change_deg = float(max_steering_change_deg)
        self.lookahead_search_mode = str(lookahead_search_mode)
        self.curvature_lookahead_enabled = bool(
            curvature_lookahead_enabled
        )
        self.curvature_full_scale_1pm = float(
            curvature_full_scale_1pm
        )
        self.curvature_reduction_max_m = float(
            curvature_reduction_max_m
        )
        self.curvature_sample_gap_m = float(curvature_sample_gap_m)
        self.lookahead_filter_tau_sec = float(
            lookahead_filter_tau_sec
        )
        self.time_based_steering_limit_enabled = bool(
            time_based_steering_limit_enabled
        )
        self.nominal_control_rate_hz = float(nominal_control_rate_hz)
        self.min_control_dt_sec = float(min_control_dt_sec)
        self.max_control_dt_sec = float(max_control_dt_sec)
        self.max_steering_rate_deg_s = float(
            max_steering_rate_deg_s
        )
        self.max_steering_accel_deg_s2 = float(
            max_steering_accel_deg_s2
        )
        self.minimum_path_preview_m = float(minimum_path_preview_m)
        self.curvature_tracking_enabled = bool(
            curvature_tracking_enabled
        )
        self.curvature_tracking_gain = float(curvature_tracking_gain)
        self.curvature_tracking_sample_gap_m = float(
            curvature_tracking_sample_gap_m
        )
        self.curvature_tracking_preview_m = float(
            curvature_tracking_preview_m
        )
        self.curvature_tracking_min_samples = int(
            curvature_tracking_min_samples
        )
        self.curvature_tracking_max_mad_1pm = float(
            curvature_tracking_max_mad_1pm
        )
        self.curvature_tracking_max_correction_1pm = float(
            curvature_tracking_max_correction_1pm
        )
        self.curvature_tracking_sign_guard_1pm = float(
            curvature_tracking_sign_guard_1pm
        )
        self.curvature_tracking_min_deficit_1pm = float(
            curvature_tracking_min_deficit_1pm
        )
        self._validate_parameters()

        self.last_steering_deg = 0.0
        self.last_steering_rate_deg_s = 0.0
        self.filtered_lookahead_m: Optional[float] = None
        self.path_fallback = False

    def _validate_parameters(self) -> None:
        if self.wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be positive")
        if (
            self.lookahead_min_m <= 0.0
            or self.lookahead_max_m < self.lookahead_min_m
        ):
            raise ValueError(
                "lookahead must satisfy 0 < min <= max"
            )
        if self.ld_throttle_max <= self.ld_throttle_min:
            raise ValueError(
                "ld throttle range must satisfy min < max"
            )
        if self.max_steer_deg <= 0.0:
            raise ValueError("max_steer_deg must be positive")
        if self.steering_gain <= 0.0:
            raise ValueError("steering_gain must be positive")
        if not 0.0 <= self.steering_ema_alpha <= 1.0:
            raise ValueError(
                "steering_ema_alpha must be between 0 and 1"
            )
        if self.steering_deadband_deg < 0.0:
            raise ValueError("steering_deadband_deg cannot be negative")
        if self.lookahead_search_mode not in (
            "simple",
            "arc_length",
            "continuous_arc_length",
        ):
            raise ValueError(
                "lookahead_search_mode must be simple, arc_length, or "
                "continuous_arc_length"
            )
        if self.max_steering_change_deg <= 0.0:
            raise ValueError(
                "max_steering_change_deg must be positive"
            )
        if self.curvature_full_scale_1pm <= 0.0:
            raise ValueError("curvature_full_scale_1pm must be positive")
        if self.curvature_reduction_max_m < 0.0:
            raise ValueError("curvature_reduction_max_m cannot be negative")
        if self.curvature_sample_gap_m <= 0.0:
            raise ValueError("curvature_sample_gap_m must be positive")
        if self.lookahead_filter_tau_sec < 0.0:
            raise ValueError("lookahead_filter_tau_sec cannot be negative")
        if self.nominal_control_rate_hz <= 0.0:
            raise ValueError("nominal_control_rate_hz must be positive")
        if (
            self.min_control_dt_sec <= 0.0
            or self.max_control_dt_sec < self.min_control_dt_sec
        ):
            raise ValueError("control dt must satisfy 0 < min <= max")
        if self.max_steering_rate_deg_s <= 0.0:
            raise ValueError("max_steering_rate_deg_s must be positive")
        if self.max_steering_accel_deg_s2 <= 0.0:
            raise ValueError("max_steering_accel_deg_s2 must be positive")
        if self.minimum_path_preview_m < 0.0:
            raise ValueError("minimum_path_preview_m cannot be negative")
        if not 0.0 <= self.curvature_tracking_gain <= 1.0:
            raise ValueError(
                "curvature_tracking_gain must be between 0 and 1"
            )
        if self.curvature_tracking_sample_gap_m <= 0.0:
            raise ValueError(
                "curvature_tracking_sample_gap_m must be positive"
            )
        if (
            self.curvature_tracking_preview_m
            < 2.0 * self.curvature_tracking_sample_gap_m
        ):
            raise ValueError(
                "curvature_tracking_preview_m must be at least twice "
                "curvature_tracking_sample_gap_m"
            )
        if self.curvature_tracking_min_samples < 1:
            raise ValueError(
                "curvature_tracking_min_samples must be positive"
            )
        if self.curvature_tracking_max_mad_1pm < 0.0:
            raise ValueError(
                "curvature_tracking_max_mad_1pm cannot be negative"
            )
        if self.curvature_tracking_max_correction_1pm < 0.0:
            raise ValueError(
                "curvature_tracking_max_correction_1pm cannot be negative"
            )
        if self.curvature_tracking_sign_guard_1pm < 0.0:
            raise ValueError(
                "curvature_tracking_sign_guard_1pm cannot be negative"
            )
        if self.curvature_tracking_min_deficit_1pm < 0.0:
            raise ValueError(
                "curvature_tracking_min_deficit_1pm cannot be negative"
            )

    def set_path_fallback(self, fallback: bool) -> None:
        self.path_fallback = bool(fallback)

    def set_current_throttle(self, throttle: float) -> None:
        if not math.isfinite(throttle):
            return
        # This is a normalized command proxy because the vehicle currently
        # has no wheel-speed/odometry topic.  Use magnitude so reverse and
        # forward commands with the same requested speed get the same Ld.
        self.current_throttle = float(
            np.clip(
                abs(throttle),
                self.ld_throttle_min,
                self.ld_throttle_max,
            )
        )

    def compute(
        self,
        points: np.ndarray,
        dt_sec: Optional[float] = None,
    ) -> SteeringCommand:
        if (
            points is None
            or len(points) == 0
            or points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            return self.stop("empty_or_invalid_path")

        control_dt = self._normalize_control_dt(dt_sec)
        lookahead_reference_m = self._dynamic_lookahead()
        path_curvature = self.estimate_path_curvature(
            points,
            lookahead_reference_m,
        )
        lookahead_target_m = self._curvature_adjusted_lookahead(
            lookahead_reference_m,
            path_curvature,
        )
        lookahead_m = self._filter_lookahead(
            lookahead_target_m,
            control_dt,
        )
        target_search_lookahead_m = self._target_search_lookahead(
            points,
            lookahead_m,
        )

        if self.lookahead_search_mode == "continuous_arc_length":
            target = self.interpolate_lookahead_point(
                points,
                target_search_lookahead_m,
            )
        else:
            distances = np.linalg.norm(points, axis=1)
            target_index = self._find_lookahead_index(
                points,
                distances,
                target_search_lookahead_m,
            )
            target_point = np.asarray(
                points[target_index], dtype=np.float64
            )
            target = LookaheadTarget(
                point=target_point,
                lower_index=target_index,
                upper_index=target_index,
                segment_ratio=0.0,
                clamped_to_endpoint=False,
            )

        forward, left = target.point
        target_distance = max(
            float(np.linalg.norm(target.point)), 1e-3
        )
        target_path_preview_m = self._path_arc_to_point(
            points,
            target.point,
        )
        heading_error = math.atan2(
            float(left),
            max(float(forward), 1e-3),
        )
        pure_pursuit_curvature = float(
            2.0 * math.sin(heading_error) / target_distance
        )
        (
            target_path_curvature,
            target_curvature_valid,
            target_curvature_reason,
            target_curvature_samples,
            target_curvature_mad,
        ) = self.estimate_target_path_curvature(
            points,
            target.point,
        )
        (
            commanded_curvature,
            curvature_tracking_error,
            curvature_tracking_correction,
            curvature_tracking_applied,
            curvature_tracking_reason,
        ) = self._apply_curvature_tracking(
            pure_pursuit_curvature,
            target_path_curvature,
            target_curvature_valid,
            target_curvature_reason,
        )
        raw_steering_deg = math.degrees(
            math.atan2(
                self.wheelbase_m * commanded_curvature,
                1.0,
            )
        )
        raw_steering_deg = float(
            np.clip(
                raw_steering_deg * self.steering_gain,
                -self.max_steer_deg,
                self.max_steer_deg,
            )
        )
        if abs(raw_steering_deg) < self.steering_deadband_deg:
            raw_steering_deg = 0.0

        if self.time_based_steering_limit_enabled:
            steering_deg = self._limit_steering_dynamics(
                raw_steering_deg,
                control_dt,
            )
        else:
            steering_deg = self._limit_steering_legacy(
                raw_steering_deg
            )

        return SteeringCommand(
            path_valid=True,
            reason="ok",
            steering_deg=steering_deg,
            raw_steering_deg=raw_steering_deg,
            steering_rate_deg_s=self.last_steering_rate_deg_s,
            lookahead_m=lookahead_m,
            lookahead_reference_m=lookahead_reference_m,
            target_search_lookahead_m=target_search_lookahead_m,
            target_path_preview_m=target_path_preview_m,
            path_curvature_1pm=path_curvature,
            pure_pursuit_curvature_1pm=pure_pursuit_curvature,
            target_path_curvature_1pm=target_path_curvature,
            target_path_curvature_samples=target_curvature_samples,
            target_path_curvature_mad_1pm=target_curvature_mad,
            curvature_tracking_error_1pm=curvature_tracking_error,
            curvature_tracking_correction_1pm=(
                curvature_tracking_correction
            ),
            curvature_tracking_applied=curvature_tracking_applied,
            curvature_tracking_reason=curvature_tracking_reason,
            target_distance_m=target_distance,
            target_index=target.lower_index,
            target_upper_index=target.upper_index,
            target_segment_ratio=target.segment_ratio,
            target_forward_m=float(forward),
            target_left_m=float(left),
            target_clamped_to_endpoint=target.clamped_to_endpoint,
            control_dt_sec=control_dt,
        )

    def _limit_steering_legacy(self, target_deg: float) -> float:
        filtered = (
            self.steering_ema_alpha * target_deg
            + (1.0 - self.steering_ema_alpha) * self.last_steering_deg
        )
        steering_step = float(
            np.clip(
                filtered - self.last_steering_deg,
                -self.max_steering_change_deg,
                self.max_steering_change_deg,
            )
        )
        self.last_steering_deg = float(
            np.clip(
                self.last_steering_deg + steering_step,
                -self.max_steer_deg,
                self.max_steer_deg,
            )
        )
        self.last_steering_rate_deg_s = 0.0
        return self.last_steering_deg

    def _limit_steering_dynamics(
        self,
        target_deg: float,
        dt_sec: float,
    ) -> float:
        error = float(target_deg) - self.last_steering_deg
        if abs(error) <= 1e-9:
            desired_rate = 0.0
        else:
            accel_step = self.max_steering_accel_deg_s2 * dt_sec
            # Discrete stopping-speed bound. The additional accel_step
            # term starts braking one sample earlier than sqrt(2*a*x),
            # preventing a sampled controller from crossing the target.
            stopping_rate = max(
                0.0,
                -accel_step
                + math.sqrt(
                    accel_step * accel_step
                    + 2.0
                    * self.max_steering_accel_deg_s2
                    * abs(error)
                ),
            )
            desired_rate = math.copysign(
                min(self.max_steering_rate_deg_s, stopping_rate),
                error,
            )

        max_rate_change = self.max_steering_accel_deg_s2 * dt_sec
        rate_delta = float(
            np.clip(
                desired_rate - self.last_steering_rate_deg_s,
                -max_rate_change,
                max_rate_change,
            )
        )
        new_rate = self.last_steering_rate_deg_s + rate_delta
        new_angle = self.last_steering_deg + new_rate * dt_sec

        # Do not cross the requested angle when the discrete integration
        # step reaches it. This also settles the stored rate at zero.
        if error * (float(target_deg) - new_angle) <= 0.0:
            new_angle = float(target_deg)
            new_rate = 0.0

        clipped_angle = float(
            np.clip(new_angle, -self.max_steer_deg, self.max_steer_deg)
        )
        if clipped_angle != new_angle:
            new_rate = 0.0

        self.last_steering_deg = clipped_angle
        self.last_steering_rate_deg_s = float(new_rate)
        return self.last_steering_deg

    def _normalize_control_dt(
        self, dt_sec: Optional[float]
    ) -> float:
        nominal_dt = 1.0 / self.nominal_control_rate_hz
        if dt_sec is None or not math.isfinite(dt_sec) or dt_sec <= 0.0:
            self.last_steering_rate_deg_s = 0.0
            return nominal_dt
        if dt_sec > self.max_control_dt_sec:
            # A long path gap must not become one large integration step.
            self.last_steering_rate_deg_s = 0.0
            return nominal_dt
        return float(
            np.clip(
                dt_sec,
                self.min_control_dt_sec,
                self.max_control_dt_sec,
            )
        )

    def stop(self, reason: str) -> SteeringCommand:
        self.last_steering_rate_deg_s = 0.0
        lookahead_m = (
            self.filtered_lookahead_m
            if self.filtered_lookahead_m is not None
            else self._dynamic_lookahead()
        )
        return SteeringCommand(
            path_valid=False,
            reason=reason,
            steering_deg=self.last_steering_deg,
            raw_steering_deg=self.last_steering_deg,
            steering_rate_deg_s=self.last_steering_rate_deg_s,
            lookahead_m=lookahead_m,
            lookahead_reference_m=self._dynamic_lookahead(),
            target_search_lookahead_m=lookahead_m,
            target_path_preview_m=0.0,
            path_curvature_1pm=0.0,
            pure_pursuit_curvature_1pm=0.0,
            target_path_curvature_1pm=0.0,
            target_path_curvature_samples=0,
            target_path_curvature_mad_1pm=0.0,
            curvature_tracking_error_1pm=0.0,
            curvature_tracking_correction_1pm=0.0,
            curvature_tracking_applied=False,
            curvature_tracking_reason="invalid_path",
            target_distance_m=0.0,
            target_index=-1,
            target_upper_index=-1,
            target_segment_ratio=0.0,
            target_forward_m=0.0,
            target_left_m=0.0,
            target_clamped_to_endpoint=False,
            control_dt_sec=0.0,
        )

    @staticmethod
    def _prepare_path(
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a topology-preserving near-to-far path and index map."""

        path = np.asarray(points, dtype=np.float64)
        if (
            path.ndim != 2
            or path.shape[1] != 2
            or len(path) == 0
            or not np.all(np.isfinite(path))
        ):
            raise ValueError("invalid path")

        original_indices = np.arange(len(path), dtype=np.int64)
        # Do not sort individual points by radius: that can scramble a
        # curved path. Only reverse the complete array when its endpoint
        # order is far-to-near. drive_pkg currently already publishes
        # near-to-far, so this is a compatibility guard.
        if len(path) >= 2 and path[0, 0] > path[-1, 0]:
            path = path[::-1]
            original_indices = original_indices[::-1]

        if len(path) >= 2:
            segment_lengths = np.linalg.norm(
                np.diff(path, axis=0), axis=1
            )
            keep = np.concatenate(([True], segment_lengths > 1e-6))
            path = path[keep]
            original_indices = original_indices[keep]
        return path, original_indices

    @classmethod
    def interpolate_lookahead_point(
        cls,
        points: np.ndarray,
        lookahead_m: float,
    ) -> LookaheadTarget:
        """Interpolate continuously where cumulative path length hits Ld."""

        path, original_indices = cls._prepare_path(points)
        first_distance = float(np.linalg.norm(path[0]))
        if len(path) == 1 or lookahead_m <= first_distance:
            index = int(original_indices[0])
            return LookaheadTarget(
                point=path[0].copy(),
                lower_index=index,
                upper_index=index,
                segment_ratio=0.0,
                clamped_to_endpoint=True,
            )

        cumulative = np.empty(len(path), dtype=np.float64)
        cumulative[0] = first_distance
        cumulative[1:] = first_distance + np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)
        )
        if lookahead_m >= cumulative[-1]:
            last = len(path) - 1
            index = int(original_indices[last])
            return LookaheadTarget(
                point=path[last].copy(),
                lower_index=index,
                upper_index=index,
                segment_ratio=1.0,
                clamped_to_endpoint=True,
            )

        upper = int(
            np.searchsorted(cumulative, lookahead_m, side="left")
        )
        lower = upper - 1
        segment_length = cumulative[upper] - cumulative[lower]
        ratio = float(
            np.clip(
                (lookahead_m - cumulative[lower])
                / max(float(segment_length), 1e-6),
                0.0,
                1.0,
            )
        )
        target = path[lower] + ratio * (path[upper] - path[lower])
        return LookaheadTarget(
            point=target,
            lower_index=int(original_indices[lower]),
            upper_index=int(original_indices[upper]),
            segment_ratio=ratio,
            clamped_to_endpoint=False,
        )

    def _target_search_lookahead(
        self,
        points: np.ndarray,
        filtered_lookahead_m: float,
    ) -> float:
        """Keep the target a usable arc distance into the visible path.

        The BEV path can begin more than one metre in front of the rear
        axle.  In that case a nominal Ld only slightly larger than the first
        point would make PP react to roughly the first 20 cm of observed
        path.  This guard preserves the vehicle-relative Ld whenever it is
        already long enough and only extends the target search when the
        visible-path preview would otherwise be too short.
        """

        path, _ = self._prepare_path(points)
        first_distance = float(np.linalg.norm(path[0]))
        preview_guard = min(
            first_distance + self.minimum_path_preview_m,
            self.lookahead_max_m,
        )
        return float(
            max(
                filtered_lookahead_m,
                preview_guard,
            )
        )

    @staticmethod
    def _point_at_arc_length(
        path: np.ndarray,
        cumulative: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        if distance_m <= cumulative[0]:
            return path[0]
        if distance_m >= cumulative[-1]:
            return path[-1]
        upper = int(np.searchsorted(cumulative, distance_m, side="left"))
        lower = upper - 1
        length = cumulative[upper] - cumulative[lower]
        ratio = (distance_m - cumulative[lower]) / max(length, 1e-6)
        return path[lower] + ratio * (path[upper] - path[lower])

    @staticmethod
    def _three_point_curvature(
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray,
    ) -> float:
        ab = point_b - point_a
        ac = point_c - point_a
        bc = point_c - point_b
        denominator = (
            np.linalg.norm(ab)
            * np.linalg.norm(ac)
            * np.linalg.norm(bc)
        )
        if denominator <= 1e-9:
            return 0.0
        cross = ab[0] * ac[1] - ab[1] * ac[0]
        return float(2.0 * cross / denominator)

    @staticmethod
    def _closest_path_arc_length(
        path: np.ndarray,
        cumulative: np.ndarray,
        point: np.ndarray,
    ) -> float:
        """Project a point onto the path and return its path arc length."""

        segments = np.diff(path, axis=0)
        length_squared = np.sum(segments * segments, axis=1)
        valid = length_squared > 1e-12
        if not np.any(valid):
            return 0.0

        ratios = np.zeros(len(segments), dtype=np.float64)
        offsets = np.asarray(point, dtype=np.float64) - path[:-1]
        ratios[valid] = np.sum(
            offsets[valid] * segments[valid], axis=1
        ) / length_squared[valid]
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = path[:-1] + ratios[:, None] * segments
        distances_squared = np.sum(
            (projections - point) ** 2,
            axis=1,
        )
        distances_squared[~valid] = np.inf
        index = int(np.argmin(distances_squared))
        segment_length = math.sqrt(float(length_squared[index]))
        return float(
            cumulative[index] + ratios[index] * segment_length
        )

    def _path_arc_to_point(
        self,
        points: np.ndarray,
        point: np.ndarray,
    ) -> float:
        """Return visible-path arc length from its first point to point."""

        path, _ = self._prepare_path(points)
        if len(path) < 2:
            return 0.0
        cumulative = np.zeros(len(path), dtype=np.float64)
        cumulative[1:] = np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)
        )
        return self._closest_path_arc_length(
            path,
            cumulative,
            np.asarray(point, dtype=np.float64),
        )

    def estimate_target_path_curvature(
        self,
        points: np.ndarray,
        target_point: np.ndarray,
    ) -> tuple[float, bool, str, int, float]:
        """Robustly measure curvature around and ahead of the PP target.

        Multiple overlapping three-point curvatures are sampled over a
        forward preview window.  Their median rejects a single local kink;
        MAD rejects a window whose curvature is too inconsistent to be a
        trustworthy feed-forward reference.  This is deliberately stricter
        than the curvature used only to schedule Ld.
        """

        path, _ = self._prepare_path(points)
        if len(path) < 3:
            return 0.0, False, "insufficient_path_points", 0, 0.0

        cumulative = np.zeros(len(path), dtype=np.float64)
        cumulative[1:] = np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)
        )
        target_arc = self._closest_path_arc_length(
            path,
            cumulative,
            np.asarray(target_point, dtype=np.float64),
        )
        gap = self.curvature_tracking_sample_gap_m
        preview_end = min(
            float(cumulative[-1]),
            target_arc + self.curvature_tracking_preview_m,
        )
        first_start = max(0.0, target_arc - gap)
        last_start = preview_end - 2.0 * gap
        if last_start + 1e-6 < first_start:
            return 0.0, False, "insufficient_target_coverage", 0, 0.0

        sample_step = 0.5 * gap
        starts = np.arange(
            first_start,
            last_start + 0.25 * sample_step,
            sample_step,
            dtype=np.float64,
        )
        if starts.size == 0 or last_start - starts[-1] > 1e-6:
            starts = np.append(starts, last_start)

        curvatures = []
        for start in starts:
            point_a = self._point_at_arc_length(
                path, cumulative, float(start)
            )
            point_b = self._point_at_arc_length(
                path, cumulative, float(start + gap)
            )
            point_c = self._point_at_arc_length(
                path, cumulative, float(start + 2.0 * gap)
            )
            curvature = self._three_point_curvature(
                point_a,
                point_b,
                point_c,
            )
            if math.isfinite(curvature):
                curvatures.append(curvature)

        sample_count = len(curvatures)
        if sample_count < self.curvature_tracking_min_samples:
            return (
                0.0,
                False,
                "insufficient_curvature_samples",
                sample_count,
                0.0,
            )

        values = np.asarray(curvatures, dtype=np.float64)
        curvature = float(np.median(values))
        mad = float(np.median(np.abs(values - curvature)))
        if mad > self.curvature_tracking_max_mad_1pm:
            return (
                curvature,
                False,
                "unstable_preview_curvature",
                sample_count,
                mad,
            )
        return curvature, True, "ok", sample_count, mad

    def _apply_curvature_tracking(
        self,
        pure_pursuit_curvature_1pm: float,
        target_path_curvature_1pm: float,
        target_curvature_valid: bool,
        invalid_reason: str,
    ) -> tuple[float, float, float, bool, str]:
        """Add only a bounded, quality-gated PP curvature deficit.

        Reference curvature may fill missing PP curvature, but it never
        reduces a stronger same-direction PP command because that command
        can contain the feedback needed to return the vehicle to the path.
        """

        if not self.curvature_tracking_enabled:
            return (
                pure_pursuit_curvature_1pm,
                0.0,
                0.0,
                False,
                "disabled",
            )
        if self.path_fallback:
            return (
                pure_pursuit_curvature_1pm,
                0.0,
                0.0,
                False,
                "fallback_path",
            )
        if not target_curvature_valid:
            return (
                pure_pursuit_curvature_1pm,
                0.0,
                0.0,
                False,
                invalid_reason,
            )

        tracking_error = (
            target_path_curvature_1pm
            - pure_pursuit_curvature_1pm
        )
        sign_guard = self.curvature_tracking_sign_guard_1pm
        if abs(target_path_curvature_1pm) < sign_guard:
            return (
                pure_pursuit_curvature_1pm,
                tracking_error,
                0.0,
                False,
                "reference_curvature_below_guard",
            )
        if (
            abs(pure_pursuit_curvature_1pm) >= sign_guard
            and pure_pursuit_curvature_1pm
            * target_path_curvature_1pm
            < 0.0
        ):
            return (
                pure_pursuit_curvature_1pm,
                tracking_error,
                0.0,
                False,
                "opposite_direction_guard",
            )

        same_direction = (
            pure_pursuit_curvature_1pm
            * target_path_curvature_1pm
            > 0.0
        )
        if (
            same_direction
            and abs(pure_pursuit_curvature_1pm)
            >= abs(target_path_curvature_1pm)
        ):
            return (
                pure_pursuit_curvature_1pm,
                tracking_error,
                0.0,
                False,
                "pp_already_sufficient",
            )

        curvature_deficit = max(
            abs(target_path_curvature_1pm)
            - abs(pure_pursuit_curvature_1pm),
            0.0,
        )
        if (
            curvature_deficit
            <= self.curvature_tracking_min_deficit_1pm
        ):
            return (
                pure_pursuit_curvature_1pm,
                tracking_error,
                0.0,
                False,
                "curvature_deficit_below_guard",
            )

        bounded_deficit = min(
            curvature_deficit,
            self.curvature_tracking_max_correction_1pm,
        )
        correction = math.copysign(
            self.curvature_tracking_gain * bounded_deficit,
            target_path_curvature_1pm,
        )
        return (
            pure_pursuit_curvature_1pm + correction,
            tracking_error,
            correction,
            abs(correction) > 1e-9,
            "applied" if abs(correction) > 1e-9 else "no_error",
        )

    def estimate_path_curvature(
        self,
        points: np.ndarray,
        reference_lookahead_m: float,
    ) -> float:
        if not self.curvature_lookahead_enabled:
            return 0.0
        path, _ = self._prepare_path(points)
        if len(path) < 3:
            return 0.0

        cumulative = np.zeros(len(path), dtype=np.float64)
        cumulative[1:] = np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)
        )
        first_distance = float(np.linalg.norm(path[0]))
        preview_length = max(reference_lookahead_m - first_distance, 0.0)
        preview_length = min(
            float(cumulative[-1]),
            preview_length + self.curvature_sample_gap_m,
        )
        if preview_length <= 1e-3:
            return 0.0

        gap = min(
            self.curvature_sample_gap_m,
            0.5 * preview_length,
        )
        if gap <= 1e-3:
            return 0.0

        last_start = preview_length - 2.0 * gap
        if last_start <= 1e-6:
            starts = np.asarray([0.0], dtype=np.float64)
        else:
            starts = np.arange(
                0.0,
                last_start + 0.25 * gap,
                0.5 * gap,
                dtype=np.float64,
            )

        curvatures = []
        for start in starts:
            point_a = self._point_at_arc_length(
                path, cumulative, float(start)
            )
            point_b = self._point_at_arc_length(
                path, cumulative, float(start + gap)
            )
            point_c = self._point_at_arc_length(
                path, cumulative, float(start + 2.0 * gap)
            )
            curvature = self._three_point_curvature(
                point_a, point_b, point_c
            )
            if math.isfinite(curvature):
                curvatures.append(curvature)
        if not curvatures:
            return 0.0
        return float(np.median(curvatures))

    def _curvature_adjusted_lookahead(
        self,
        reference_lookahead_m: float,
        path_curvature_1pm: float,
    ) -> float:
        if (
            not self.dynamic_lookahead_enabled
            or not self.curvature_lookahead_enabled
        ):
            return reference_lookahead_m
        curvature_ratio = float(
            np.clip(
                abs(path_curvature_1pm)
                / self.curvature_full_scale_1pm,
                0.0,
                1.0,
            )
        )
        target = reference_lookahead_m - (
            self.curvature_reduction_max_m * curvature_ratio
        )
        return float(
            np.clip(target, self.lookahead_min_m, self.lookahead_max_m)
        )

    def _filter_lookahead(
        self,
        target_lookahead_m: float,
        dt_sec: float,
    ) -> float:
        if self.filtered_lookahead_m is None:
            self.filtered_lookahead_m = float(target_lookahead_m)
        elif self.lookahead_filter_tau_sec <= 0.0:
            self.filtered_lookahead_m = float(target_lookahead_m)
        else:
            alpha = 1.0 - math.exp(
                -dt_sec / self.lookahead_filter_tau_sec
            )
            self.filtered_lookahead_m += alpha * (
                target_lookahead_m - self.filtered_lookahead_m
            )
        self.filtered_lookahead_m = float(
            np.clip(
                self.filtered_lookahead_m,
                self.lookahead_min_m,
                self.lookahead_max_m,
            )
        )
        return self.filtered_lookahead_m

    def _find_lookahead_index(
        self,
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        if self.lookahead_search_mode == "arc_length":
            return self._arc_length_lookahead_index(
                points, distances, lookahead_m
            )
        return self._simple_lookahead_index(points, distances, lookahead_m)

    @staticmethod
    def _simple_lookahead_index(
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        """Original local closest-to-Ld point search."""

        del points
        return int(np.argmin(np.abs(distances - lookahead_m)))

    @classmethod
    def _arc_length_lookahead_index(
        cls,
        points: np.ndarray,
        distances: np.ndarray,
        lookahead_m: float,
    ) -> int:
        """Legacy arc-length mode, corrected for either path direction."""

        del distances
        path, original_indices = cls._prepare_path(points)
        cumulative = np.empty(len(path), dtype=np.float64)
        cumulative[0] = float(np.linalg.norm(path[0]))
        if len(path) > 1:
            cumulative[1:] = cumulative[0] + np.cumsum(
                np.linalg.norm(np.diff(path, axis=0), axis=1)
            )
        position = int(
            np.searchsorted(cumulative, lookahead_m, side="left")
        )
        position = min(position, len(path) - 1)
        return int(original_indices[position])

    def _dynamic_lookahead(self) -> float:
        if self.lookahead_min_m == self.lookahead_max_m:
            return self.lookahead_min_m

        # Schedule Ld from the requested speed proxy, not from the previous
        # steering output.  Steering-based scheduling formed a positive
        # feedback loop: more steering -> shorter Ld -> more steering.
        throttle_ratio = float(
            np.clip(
                (
                    self.current_throttle - self.ld_throttle_min
                )
                / (self.ld_throttle_max - self.ld_throttle_min),
                0.0,
                1.0,
            )
        )
        lookahead = self.lookahead_min_m + (
            self.lookahead_max_m - self.lookahead_min_m
        ) * throttle_ratio
        return float(
            np.clip(
                lookahead,
                self.lookahead_min_m,
                self.lookahead_max_m,
            )
        )


class PurePursuitNode(Node):
    """Publish lane steering candidates from a metric path."""

    def __init__(self) -> None:
        super().__init__("pure_pursuit")
        for name, default in PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value

        self.controller = PurePursuitController(
            wheelbase_m=float(parameter("wheelbase_m")),
            ld_throttle_min=float(parameter("ld_throttle_min")),
            ld_throttle_max=float(parameter("ld_throttle_max")),
            lookahead_min_m=float(parameter("lookahead_min_m")),
            lookahead_max_m=float(parameter("lookahead_max_m")),
            fixed_lookahead_m=float(parameter("lookahead_m")),
            dynamic_lookahead_enabled=bool(
                parameter("dynamic_lookahead_enabled")
            ),
            max_steer_deg=float(parameter("max_steer_deg")),
            steering_gain=float(parameter("steering_gain")),
            steering_ema_alpha=float(
                parameter("steering_ema_alpha")
            ),
            steering_deadband_deg=float(
                parameter("steering_deadband_deg")
            ),
            max_steering_change_deg=float(
                parameter("max_steering_change_deg")
            ),
            lookahead_search_mode=str(
                parameter("lookahead_search_mode")
            ),
            curvature_lookahead_enabled=bool(
                parameter("curvature_lookahead_enabled")
            ),
            curvature_full_scale_1pm=float(
                parameter("curvature_full_scale_1pm")
            ),
            curvature_reduction_max_m=float(
                parameter("curvature_reduction_max_m")
            ),
            curvature_sample_gap_m=float(
                parameter("curvature_sample_gap_m")
            ),
            curvature_tracking_enabled=bool(
                parameter("curvature_tracking_enabled")
            ),
            curvature_tracking_gain=float(
                parameter("curvature_tracking_gain")
            ),
            curvature_tracking_sample_gap_m=float(
                parameter("curvature_tracking_sample_gap_m")
            ),
            curvature_tracking_max_correction_1pm=float(
                parameter("curvature_tracking_max_correction_1pm")
            ),
            curvature_tracking_sign_guard_1pm=float(
                parameter("curvature_tracking_sign_guard_1pm")
            ),
            curvature_tracking_min_deficit_1pm=float(
                parameter("curvature_tracking_min_deficit_1pm")
            ),
            lookahead_filter_tau_sec=float(
                parameter("lookahead_filter_tau_sec")
            ),
            time_based_steering_limit_enabled=bool(
                parameter("time_based_steering_limit_enabled")
            ),
            nominal_control_rate_hz=float(
                parameter("nominal_control_rate_hz")
            ),
            min_control_dt_sec=float(
                parameter("min_control_dt_sec")
            ),
            max_control_dt_sec=float(
                parameter("max_control_dt_sec")
            ),
            max_steering_rate_deg_s=float(
                parameter("max_steering_rate_deg_s")
            ),
            max_steering_accel_deg_s2=float(
                parameter("max_steering_accel_deg_s2")
            ),
            minimum_path_preview_m=float(
                parameter("minimum_path_preview_m")
            ),
            curvature_tracking_preview_m=float(
                parameter("curvature_tracking_preview_m")
            ),
            curvature_tracking_min_samples=int(
                parameter("curvature_tracking_min_samples")
            ),
            curvature_tracking_max_mad_1pm=float(
                parameter("curvature_tracking_max_mad_1pm")
            ),
        )
        self.last_control_time_ns: Optional[int] = None

        path_topic = str(parameter("path_topic"))
        path_status_topic = str(parameter("path_status_topic"))
        steer_angle_topic = str(parameter("steer_angle_topic"))
        throttle_feedback_topic = str(
            parameter("throttle_feedback_topic")
        )
        candidate_valid_topic = str(parameter("candidate_valid_topic"))
        status_topic = str(parameter("status_topic"))
        self.steer_publisher = self.create_publisher(
            Float32, steer_angle_topic, 10
        )
        self.candidate_valid_publisher = self.create_publisher(
            Bool, candidate_valid_topic, 10
        )
        self.status_publisher = self.create_publisher(
            String,
            status_topic,
            10,
        )
        self.path_subscription = self.create_subscription(
            Path,
            path_topic,
            self.on_path,
            10,
        )
        self.path_status_subscription = self.create_subscription(
            String,
            path_status_topic,
            self.on_path_status,
            10,
        )
        self.throttle_feedback_subscription = self.create_subscription(
            Float32,
            throttle_feedback_topic,
            self.on_throttle_feedback,
            10,
        )

        self.visualizer: Optional[DrivingVisualizer] = None
        if bool(parameter("local_display")):
            self.visualizer = DrivingVisualizer(
                window_name=str(parameter("window_name")),
                window_x=int(parameter("window_x")),
                window_y=int(parameter("window_y")),
                display_scale=float(parameter("display_scale")),
                box_ema_alpha=float(parameter("box_ema_alpha")),
                type_switch_frames=int(
                    parameter("type_switch_frames")
                ),
                track_max_missed_frames=int(
                    parameter("track_max_missed_frames")
                ),
                track_match_distance_px=float(
                    parameter("track_match_distance_px")
                ),
                confidence_full_hits=int(
                    parameter("confidence_full_hits")
                ),
                yolo_confidence_aggregation=str(
                    parameter("yolo_confidence_aggregation")
                ),
                lookahead_target_hold_sec=float(
                    parameter("lookahead_target_hold_sec")
                ),
                reference_path_hold_sec=float(
                    parameter("reference_path_hold_sec")
                ),
                logger=self.get_logger(),
            )
            self.segmentation_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("segmentation_topic")),
                self.visualizer.on_segmentation_image,
                LOW_LATENCY_QOS,
            )
            self.bev_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("bev_topic")),
                self.visualizer.on_bev_image,
                LOW_LATENCY_QOS,
            )
            self.debug_subscription = self.create_subscription(
                CompressedImage,
                str(parameter("debug_topic")),
                self.visualizer.on_debug_image,
                LOW_LATENCY_QOS,
            )
            self.instances_subscription = self.create_subscription(
                String,
                str(parameter("instances_topic")),
                self.visualizer.on_yolo_instances,
                10,
            )

        self.get_logger().info(
            f"{path_topic} -> lane candidates: {steer_angle_topic}, "
            f"{candidate_valid_topic}; throttle feedback: "
            f"{throttle_feedback_topic}"
        )

    def on_throttle_feedback(self, message: Float32) -> None:
        self.controller.set_current_throttle(float(message.data))

    def on_path_status(self, message: String) -> None:
        if self.visualizer is not None:
            self.visualizer.on_path_status(message)
        try:
            status = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        self.controller.set_path_fallback(status.get("fallback", False))

    def on_path(self, message: Path) -> None:
        now_ns = self.get_clock().now().nanoseconds
        dt_sec: Optional[float] = None
        if self.last_control_time_ns is not None:
            dt_sec = (now_ns - self.last_control_time_ns) * 1e-9
        self.last_control_time_ns = now_ns

        points = np.asarray(
            [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in message.poses
            ],
            dtype=np.float64,
        )
        command = self.controller.compute(points, dt_sec=dt_sec)
        self._publish_command(command)

    def _publish_command(self, command: SteeringCommand) -> None:
        steering_deg = command.steering_deg  # path_valid=False여도 last_steering_deg 유지
        self.steer_publisher.publish(Float32(data=float(steering_deg)))
        self.candidate_valid_publisher.publish(Bool(data=command.path_valid))
        status = {
            "path_valid": command.path_valid,
            "reason": command.reason,
            "fallback": self.controller.path_fallback,
            "steering_deg": round(command.steering_deg, 2),
            "raw_steering_deg": round(command.raw_steering_deg, 2),
            "steering_rate_deg_s": round(
                command.steering_rate_deg_s, 2
            ),
            "control_dt_sec": round(command.control_dt_sec, 4),
            "dynamic_lookahead_enabled": (
                self.controller.dynamic_lookahead_enabled
            ),
            "lookahead_m": round(command.lookahead_m, 3),
            "lookahead_reference_m": round(
                command.lookahead_reference_m, 3
            ),
            "target_search_lookahead_m": round(
                command.target_search_lookahead_m, 3
            ),
            "target_path_preview_m": round(
                command.target_path_preview_m, 3
            ),
            "path_curvature_1pm": round(
                command.path_curvature_1pm, 4
            ),
            "pure_pursuit_curvature_1pm": round(
                command.pure_pursuit_curvature_1pm, 4
            ),
            "target_path_curvature_1pm": round(
                command.target_path_curvature_1pm, 4
            ),
            "target_path_curvature_samples": int(
                command.target_path_curvature_samples
            ),
            "target_path_curvature_mad_1pm": round(
                command.target_path_curvature_mad_1pm, 4
            ),
            "curvature_tracking_error_1pm": round(
                command.curvature_tracking_error_1pm, 4
            ),
            "curvature_tracking_deficit_1pm": round(
                max(
                    abs(command.target_path_curvature_1pm)
                    - abs(command.pure_pursuit_curvature_1pm),
                    0.0,
                ),
                4,
            ),
            "curvature_tracking_correction_1pm": round(
                command.curvature_tracking_correction_1pm, 4
            ),
            "curvature_tracking_applied": (
                command.curvature_tracking_applied
            ),
            "curvature_tracking_reason": (
                command.curvature_tracking_reason
            ),
            "lookahead_target_m": round(
                command.target_distance_m, 3
            ),
            "lookahead_target_index": int(command.target_index),
            "lookahead_target_upper_index": int(
                command.target_upper_index
            ),
            "lookahead_target_segment_ratio": round(
                command.target_segment_ratio, 4
            ),
            "lookahead_target_forward_m": round(
                command.target_forward_m, 4
            ),
            "lookahead_target_left_m": round(
                command.target_left_m, 4
            ),
            "lookahead_target_clamped": (
                command.target_clamped_to_endpoint
            ),
            "current_final_throttle": round(
                self.controller.current_throttle, 3
            ),
            "candidate_steer_deg": round(steering_deg, 2),
            "candidate_valid": command.path_valid,
        }
        self.status_publisher.publish(
            String(data=json.dumps(status, ensure_ascii=False))
        )
        if self.visualizer is not None:
            self.visualizer.on_control_status(status)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.steer_publisher.publish(Float32(data=0.0))
            self.candidate_valid_publisher.publish(Bool(data=False))
        if self.visualizer is not None:
            self.visualizer.destroy()
        return super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PurePursuitNode] = None
    try:
        node = PurePursuitNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
