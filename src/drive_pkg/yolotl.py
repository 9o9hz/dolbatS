#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import pickle
from pathlib import Path as FilePath

import numpy as np
import cv2
import torch
import argparse
from math import atan2, degrees, sqrt, hypot

try:
    from scipy.interpolate import splprep, splev
except ImportError:
    splprep, splev = None, None

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


def _default_resource_path(filename: str) -> str:
    """Resolve a drive_pkg resource file from the source tree when run
    
    un-installed, else from the installed share directory."""
    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resource", filename
    )
    if os.path.isfile(source_path):
        return source_path
    try:
        return os.path.join(
            get_package_share_directory("drive_pkg"), "resource", filename
        )
    except Exception:
        return source_path


def _default_calibration_path() -> str:
    """Use the workspace calibration when available, else legacy PKL."""
    workspace_calibration = (
        FilePath(__file__).resolve().parents[2] / "camera_calibration.npz"
    )
    if workspace_calibration.is_file():
        return str(workspace_calibration)
    return _default_resource_path("camera_calibration.pkl")


def _load_camera_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load current NPZ or legacy PKL camera calibration data."""
    if FilePath(path).suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as calibration:
            camera_matrix = np.asarray(
                calibration["camera_matrix"], dtype=np.float64
            )
            distortion_key = (
                "distortion_coefficients"
                if "distortion_coefficients" in calibration.files
                else "dist_coeffs"
            )
            dist_coeffs = np.asarray(
                calibration[distortion_key], dtype=np.float64
            )
        return camera_matrix, dist_coeffs

    with open(path, "rb") as calibration_file:
        calibration = pickle.load(calibration_file)
    camera_matrix = np.asarray(
        calibration["camera_matrix"], dtype=np.float64
    )
    dist_coeffs = np.asarray(
        calibration["dist_coeffs"], dtype=np.float64
    )
    return camera_matrix, dist_coeffs


# ==============================================================================
# 유틸리티 함수
# ==============================================================================
def morph_close(binary_mask, ksize=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


def remove_small_components(binary_mask, min_size=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    cleaned = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 255
    return cleaned


def keep_top2_components(binary_mask, min_area=300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(binary_mask)

    comps = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            comps.append((i, stats[i, cv2.CC_STAT_AREA]))

    comps.sort(key=lambda x: x[1], reverse=True)

    cleaned = np.zeros_like(binary_mask)
    for i in range(min(len(comps), 2)):
        idx = comps[i][0]
        cleaned[labels == idx] = 255
    return cleaned


def final_filter(bev_mask):
    f2 = morph_close(bev_mask, ksize=5)
    f3 = remove_small_components(f2, min_size=10000)
    f4 = keep_top2_components(f3, min_area=300)
    return f4


def overlay_points(image, xs, ys, color=(0, 255, 255), radius=2):
    if xs is None or ys is None:
        return image

    h, w = image.shape[:2]
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(image, (xi, yi), radius, color, -1)

    return image


def overlay_point_path(image, xs, ys, color=(0, 0, 255), thickness=2):
    if xs is None or ys is None or len(xs) < 2:
        return image

    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(image, [pts], False, color, thickness)
    return image


def extract_centerline_points_from_component(component_mask, min_points_per_y=1, y_step=1):
    """
    하나의 connected component(binary mask)에서
    각 y마다 x의 중심점(mean)을 구해 centerline points를 만든다.
    """
    h, _ = component_mask.shape
    ys_center = []
    xs_center = []

    for y in range(0, h, y_step):
        xs = np.where(component_mask[y] > 0)[0]
        if len(xs) >= min_points_per_y:
            x_center = float(np.mean(xs))
            ys_center.append(float(y))
            xs_center.append(x_center)

    if len(ys_center) == 0:
        return None, None

    ys_center = np.array(ys_center, dtype=np.float32)
    xs_center = np.array(xs_center, dtype=np.float32)

    sort_idx = np.argsort(ys_center)
    return ys_center[sort_idx], xs_center[sort_idx]


def resample_polyline_by_y(xs, ys, y_min, y_max, y_step=1):
    """
    주어진 polyline(xs, ys)을 y축 기준으로 보간해서
    일정 y 간격의 점열로 만든다.
    """
    if xs is None or ys is None or len(xs) < 2:
        return None, None

    ys = np.asarray(ys, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)

    sort_idx = np.argsort(ys)
    ys = ys[sort_idx]
    xs = xs[sort_idx]

    y_query = np.arange(y_min, y_max + 1, y_step, dtype=np.float32)

    valid_mask = (y_query >= ys[0]) & (y_query <= ys[-1])
    y_query = y_query[valid_mask]
    if len(y_query) < 2:
        return None, None

    x_query = np.interp(y_query, ys, xs)
    return x_query.astype(np.float32), y_query.astype(np.float32)


def compute_fusion_centerline(left_xs, left_ys, right_xs, right_ys, offset_px, y_step=1):
    """
    Y축(픽셀) 단위로 차선 존재 여부를 판별하여 융합(Fusion)합니다.
    - 양차선이 존재하는 Y 구간: 두 차선의 평균(정중앙)
    - 한쪽 차선만 존재하는 Y 구간: 해당 차선의 법선 오프셋 경로
    """
    if left_xs is None or right_xs is None:
        return None, None

    # 1. 원본 차선을 y_step 단위로 촘촘하게 보간
    y_min_L, y_max_L = int(np.min(left_ys)), int(np.max(left_ys))
    lx_dense, ly_dense = resample_polyline_by_y(left_xs, left_ys, y_min_L, y_max_L, y_step)

    y_min_R, y_max_R = int(np.min(right_ys)), int(np.max(right_ys))
    rx_dense, ry_dense = resample_polyline_by_y(right_xs, right_ys, y_min_R, y_max_R, y_step)

    # 2. 각 차선의 법선 오프셋 경로를 구한 후 보간
    lx_off, ly_off = offset_polyline_points(left_xs, left_ys, offset_px, direction=+1.0)
    lx_off_dense, ly_off_dense = None, None
    if lx_off is not None and len(lx_off) > 1:
        lx_off_dense, ly_off_dense = resample_polyline_by_y(lx_off, ly_off, int(np.min(ly_off)), int(np.max(ly_off)), y_step)

    rx_off, ry_off = offset_polyline_points(right_xs, right_ys, offset_px, direction=-1.0)
    rx_off_dense, ry_off_dense = None, None
    if rx_off is not None and len(rx_off) > 1:
        rx_off_dense, ry_off_dense = resample_polyline_by_y(rx_off, ry_off, int(np.min(ry_off)), int(np.max(ry_off)), y_step)

    # 3. Y좌표를 키(Key)로 하는 딕셔너리로 매핑하여 빠른 탐색 지원
    dict_lx = dict(zip(ly_dense, lx_dense)) if ly_dense is not None else {}
    dict_rx = dict(zip(ry_dense, rx_dense)) if ry_dense is not None else {}
    dict_lx_off = dict(zip(ly_off_dense, lx_off_dense)) if ly_off_dense is not None else {}
    dict_rx_off = dict(zip(ry_off_dense, rx_off_dense)) if ry_off_dense is not None else {}

    # 4. 존재하는 전체 Y 범위 확인
    all_y = set(dict_lx.keys()).union(set(dict_rx.keys()))
    if not all_y:
        return None, None

    min_y, max_y = int(min(all_y)), int(max(all_y))

    merged_xs, merged_ys = [], []
    # [핵심] 화면 전체가 아닌, Y축 높이별로 차선 개수를 판별
    for y_val in range(min_y, max_y + 1, y_step):
        y_f = float(y_val)
        has_L = y_f in dict_lx
        has_R = y_f in dict_rx

        cx = None
        if has_L and has_R:
            # 양쪽 다 존재 -> 두 차선의 평균 (정중앙)
            cx = (dict_lx[y_f] + dict_rx[y_f]) / 2.0
        elif has_L and y_f in dict_lx_off:
            # 왼쪽만 존재 -> 왼쪽 차선을 기반으로 한 법선 오프셋
            cx = dict_lx_off[y_f]
        elif has_R and y_f in dict_rx_off:
            # 오른쪽만 존재 -> 오른쪽 차선을 기반으로 한 법선 오프셋
            cx = dict_rx_off[y_f]

        if cx is not None:
            merged_xs.append(cx)
            merged_ys.append(y_f)

    if len(merged_xs) < 2:
        return None, None

    center_xs = np.array(merged_xs, dtype=np.float32)
    center_ys = np.array(merged_ys, dtype=np.float32)

    # 5. 연결 부위(단차선 <-> 양차선)의 꺾임을 스무딩 (이동 평균)
    k_size = 15
    if len(center_xs) >= k_size:
        padded = np.pad(center_xs, (k_size//2, k_size//2), mode='edge')
        center_xs = np.convolve(padded, np.ones(k_size)/k_size, mode='valid')

    return center_xs, center_ys


def offset_polyline_points(xs, ys, offset_px, direction=1.0, sample_step=20):
    """
    점열을 일정 간격(sample_step)으로 샘플링하여 국소 접선을 구하고 법선 방향으로 offset한 뒤,
    나머지 점들은 선형 보간(Interpolation)하여 채웁니다.
    """
    if xs is None or ys is None or len(xs) < 2:
        return None, None

    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    n = len(xs)

    if n <= sample_step:
        sample_step = max(1, n // 2)

    # 1. 샘플링할 인덱스 생성
    indices = np.arange(0, n, sample_step)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    sparse_xo = []
    sparse_yo = []

    window = 30  # 노이즈를 줄이기 위해 기울기를 구할 참조 간격 설정
    
    # 2. 선택된 점들에서만 법선 벡터 및 오프셋 계산
    for i in indices:
        idx_prev = max(0, i - window)
        idx_next = min(n - 1, i + window)
        
        if idx_prev == idx_next:
            dx, dy = 0, 1
        else:
            dx = xs[idx_next] - xs[idx_prev]
            dy = ys[idx_next] - ys[idx_prev]

        norm = np.sqrt(dx * dx + dy * dy) + 1e-6

        nx = dy / norm
        ny = -dx / norm

        xo = xs[i] + direction * offset_px * nx
        yo = ys[i] + direction * offset_px * ny

        sparse_xo.append(float(xo))
        sparse_yo.append(float(yo))

    # 3. 인덱스를 기준으로 선형 보간(np.interp)하여 원래 배열의 밀도로 복원
    all_indices = np.arange(n)
    out_xs = np.interp(all_indices, indices, sparse_xo).astype(np.float32)
    out_ys = np.interp(all_indices, indices, sparse_yo).astype(np.float32)

    # 4. 결과물(X좌표)에 이동 평균을 적용하여 한 번 더 부드럽게 평활화
    if len(out_xs) >= 3:
        k_size = 5
        padded_xs = np.pad(out_xs, (k_size//2, k_size//2), mode='edge')
        out_xs = np.convolve(padded_xs, np.ones(k_size)/k_size, mode='valid')

    return out_xs, out_ys


def smooth_and_interpolate_path(xs, ys, num_points=100, smooth_factor=10.0):
    """
    중심 경로를 B-Spline으로 부드럽게 스무딩하고 일정한 간격의 점으로 보간합니다.
    """
    if splprep is None or xs is None or ys is None or len(xs) < 4:
        return xs, ys
        
    # splprep는 중복점이 있으면 에러가 발생할 수 있으므로 중복점 제거
    pts = np.column_stack((xs, ys))
    _, idx = np.unique(pts, axis=0, return_index=True)
    idx = np.sort(idx)
    xs_u, ys_u = xs[idx], ys[idx]
    
    if len(xs_u) < 4:
        return xs, ys
        
    try:
        tck, u = splprep([xs_u, ys_u], s=smooth_factor, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return x_new.astype(np.float32), y_new.astype(np.float32)
    except Exception:
        return xs, ys


# ==============================================================================
# ROS 2 노드 클래스
# ==============================================================================
class LaneFollowerNode(Node):
    def __init__(self, opt):
        super().__init__('lane_follower_node')
        self.opt = opt
        self.get_logger().info("[Lane Follower] Initializing...")

        # True: arc length 기반 (새로운 방식), False: 전방 거리 기반 (기존 방식)
        self.use_arc_length_lookahead = False

        # Lazily imported/constructed only if a raw (uncompressed) Image
        # topic is selected -- the compressed path below never touches it,
        # so a broken cv_bridge install doesn't block the common case.
        self.bridge = None

        # 1. 모델 로드
        device_str = self.opt.device
        if device_str.isdigit() and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device_str}')
        else:
            self.device = torch.device('cpu')

        self.get_logger().info(f"Loading weights from: {opt.weights}")
        self.model = YOLO(opt.weights).to(self.device)

        # 2. BEV 파라미터 로드
        self.get_logger().info(f"Loading params from: {opt.param_file}")
        try:
            params = np.load(opt.param_file)
            self.bev_params = {
                'src_points': params['src_points'],
                'dst_points': params['dst_points']
            }
            self.bev_w, self.bev_h = int(params['warp_w']), int(params['warp_h'])
            self.get_logger().info(
                f"[SUCCESS] Loaded Params: BEV Size={self.bev_w}x{self.bev_h}"
            )
        except Exception as e:
            self.get_logger().error(f"[FATAL] Failed to load BEV params: {e}")
            self.get_logger().error(f"Check file path: {opt.param_file}")
            sys.exit(1)

        # 2-1. 카메라 캘리브레이션(NPZ/legacy PKL) 로드
        self.get_logger().info(f"Loading camera calibration from: {opt.calib_file}")
        try:
            self.camera_matrix, self.dist_coeffs = (
                _load_camera_calibration(opt.calib_file)
            )
            self.use_undistort = True
            self.get_logger().info("[SUCCESS] Loaded camera calibration data")
        except Exception as e:
            self.get_logger().warn(f"[WARN] Failed to load camera calibration file: {e}")
            self.get_logger().warn("[WARN] Proceeding without undistortion")
            self.camera_matrix = None
            self.dist_coeffs = None
            self.use_undistort = False

        # 3. 주행 파라미터
        # BEV 640x640px 기준: x/y 방향 각각 1.20m (0803).
        self.m_per_pixel_y, self.y_offset_m, self.m_per_pixel_x = (
            0.001875,
            0.94,
            0.001875,
        )

        self.tracked_lanes = {
            'left': {'xs': None, 'ys': None, 'age': 0, 'class_name': None},
            'right': {'xs': None, 'ys': None, 'age': 0, 'class_name': None}
        }
        self.tracked_center_path = {'xs': None, 'ys': None}

        self.MAX_LANE_AGE = 7
        self.L = 0.54  # wheelbase 54.5cm 실측

        self.THROTTLE_MIN_FOR_LD, self.THROTTLE_MAX_FOR_LD = 0.95,0.99
        self.current_throttle = self.THROTTLE_MIN_FOR_LD

        self.MIN_LOOKAHEAD_DISTANCE = 1.3
        self.MAX_LOOKAHEAD_DISTANCE = 1.6
        self.MAX_STEER_DEG = 26.5
        self.prev_steer_deg = 0.0
        self.MAX_STEER_RATE = 12.0
        # pure_pursuit.py(drive_pipeline.yaml steering_gain=5.00)와 동일한
        # 캘리브레이션. 순수 기하학적 pure-pursuit 각도는 작아서 이 게인
        # 없이는 아두이노 폐루프가 정지마찰을 못 넘기고 거의 안 움직인다.
        self.STEERING_GAIN = 2.5

        self.last_valid_ld = float(self.MAX_LOOKAHEAD_DISTANCE)

        self.LANE_WIDTH_M = 0.9
        # 좌/우 두 선이 이보다 가까우면 같은 물리적 선의 중복 검출로
        # 보고 페어링하지 않는다 (차선폭의 70%).
        self.MIN_LANE_SPACING_M = self.LANE_WIDTH_M * 0.7

        # 4. ROS Setup
        # steer/valid 토픽명은 mission_manager.py의 lane_steer_topic/
        # lane_valid_topic 기본값과 맞춤 -- pure_pursuit 대신 이 노드를
        # 쓸 때 mission_manager가 그대로 받아쓸 수 있도록.
        self.pub_steering = self.create_publisher(
            Float32, '/control/candidate/lane/steer_angle', 1
        )
        self.pub_lane_valid = self.create_publisher(
            Bool, '/control/candidate/lane/valid', 1
        )
        self.pub_lane_status = self.create_publisher(Bool, 'lane_detect', 1)
        self.pub_path = self.create_publisher(Path, 'lane_path', 10)
        self.pub_drivable_area = self.create_publisher(
            CompressedImage, 'drivable_area/compressed', 10
        )
        self.pub_lookahead = self.create_publisher(Float32, 'lookahead_distance', 1)

        if 'compressed' in self.opt.topic:
            self.msg_type = CompressedImage
        else:
            self.msg_type = Image

        self.sub_image = self.create_subscription(
            self.msg_type,
            self.opt.topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        # drive_main.py의 image_topic 기본값(/camera/lane/raw/compressed)과
        # 동일한 카메라를 구독. throttle 피드백은 mission_manager.py가
        # 발행하는 최종 /auto_throttle을 그대로 구독.
        self.sub_throttle = self.create_subscription(
            Float32,
            '/auto_throttle',
            self.throttle_callback,
            1
        )

        cv2.namedWindow("Original Camera View", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Final Path & Logs (on BEV)", cv2.WINDOW_AUTOSIZE)

        self.frame_count = 0

    def throttle_callback(self, msg):
        self.current_throttle = np.clip(
            msg.data, self.THROTTLE_MIN_FOR_LD, self.THROTTLE_MAX_FOR_LD
        )

    def undistort_image(self, image):
        if not self.use_undistort or self.camera_matrix is None or self.dist_coeffs is None:
            return image

        h, w = image.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,
            (w, h)
        )

        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix
        )

        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y + rh, x:x + rw]
            undistorted = cv2.resize(undistorted, (w, h))

        return undistorted

    def do_bev_transform(self, image):
        M = cv2.getPerspectiveTransform(
            self.bev_params['src_points'],
            self.bev_params['dst_points']
        )
        return cv2.warpPerspective(
            image, M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR
        )

    def draw_drivable_area_on_original(self, image, left_xs, left_ys, right_xs, right_ys, center_xs, center_ys):
        M_inv = cv2.getPerspectiveTransform(
            self.bev_params['dst_points'],
            self.bev_params['src_points']
        )
        draw_img = image.copy()

        lane_width_pixels = self.LANE_WIDTH_M / self.m_per_pixel_x

        viz_left_xs, viz_left_ys = left_xs, left_ys
        viz_right_xs, viz_right_ys = right_xs, right_ys

        if viz_left_xs is not None and viz_right_xs is None:
            r_xs, r_ys = offset_polyline_points(viz_left_xs, viz_left_ys, lane_width_pixels, direction=+1.0)
            if r_xs is not None:
                viz_right_xs, viz_right_ys = r_xs, r_ys
        elif viz_right_xs is not None and viz_left_xs is None:
            l_xs, l_ys = offset_polyline_points(viz_right_xs, viz_right_ys, lane_width_pixels, direction=-1.0)
            if l_xs is not None:
                viz_left_xs, viz_left_ys = l_xs, l_ys

        if viz_left_xs is not None and viz_right_xs is not None and len(viz_left_xs) > 1 and len(viz_right_xs) > 1:
            pts_left = np.stack([viz_left_xs, viz_left_ys], axis=1)
            pts_right = np.stack([viz_right_xs, viz_right_ys], axis=1)
            
            pts_bev = np.vstack([pts_left, pts_right[::-1]])
            pts_bev = np.array([pts_bev], dtype=np.float32)
            
            pts_orig = cv2.perspectiveTransform(pts_bev, M_inv)
            
            overlay = draw_img.copy()
            cv2.fillPoly(overlay, [np.int32(pts_orig)], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, draw_img, 0.7, 0, draw_img)

        def transform_and_draw(xs, ys, color, thickness=3):
            if xs is None or ys is None or len(xs) < 2:
                return
            pts_bev = np.array([np.stack([xs, ys], axis=1)], dtype=np.float32)
            pts_orig = cv2.perspectiveTransform(pts_bev, M_inv)
            cv2.polylines(draw_img, [np.int32(pts_orig)[0]], False, color, thickness)

        transform_and_draw(left_xs, left_ys, (255, 0, 0), thickness=2)
        transform_and_draw(right_xs, right_ys, (0, 0, 255), thickness=2)
        transform_and_draw(center_xs, center_ys, (0, 255, 0), thickness=3)

        return draw_img

    def image_to_vehicle(self, pt_bev):
        u, v = pt_bev
        x_vehicle = (self.bev_h - v) * self.m_per_pixel_y + self.y_offset_m
        y_vehicle = (self.bev_w / 2 - u) * self.m_per_pixel_x
        return x_vehicle, y_vehicle

    def _lane_pair_is_plausible(self, xs_l, ys_l, xs_r, ys_r):
        """Two lines can only be trusted as a left/right pair where their
        y-ranges actually overlap -- comparing x at each side's own
        (possibly very different) y otherwise mistakes two pieces of the
        same physical line for separate boundaries."""
        y_min_l, y_max_l = float(ys_l[0]), float(ys_l[-1])
        y_min_r, y_max_r = float(ys_r[0]), float(ys_r[-1])
        overlap_min = max(y_min_l, y_min_r)
        overlap_max = min(y_max_l, y_max_r)
        if overlap_max <= overlap_min:
            return False
        y_ref = overlap_max
        x_l = float(np.interp(y_ref, ys_l, xs_l))
        x_r = float(np.interp(y_ref, ys_r, xs_r))
        spacing_px = x_r - x_l
        min_spacing_px = self.MIN_LANE_SPACING_M / self.m_per_pixel_x
        return spacing_px >= min_spacing_px

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            if self.msg_type == Image:
                self.get_logger().info(
                    f"[Alive] Image Size: {msg.width}x{msg.height}, enc={msg.encoding}"
                )
            else:
                self.get_logger().info(
                    f"[Alive] Compressed Image Received, format={msg.format}"
                )

        try:
            if self.msg_type == Image:
                if self.bridge is None:
                    from cv_bridge import CvBridge

                    self.bridge = CvBridge()
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                np_arr = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return

            self.process_image(np.ascontiguousarray(img))

        except Exception as e:
            self.get_logger().error(f"Decode Error: {e}")

    def process_image(self, im0s):
        im0s = self.undistort_image(im0s)

        steer_deg = None
        goal_point_bev = None
        lane_detected_bool = False
        dynamic_lookahead_distance = self.last_valid_ld

        annotated_frame = im0s.copy()

        left_fit_xs, left_fit_ys = None, None
        right_fit_xs, right_fit_ys = None, None
        center_fit_xs, center_fit_ys = None, None

        # 1. BEV Transform
        bev_image_input = self.do_bev_transform(im0s)

        # 2. Inference
        results = self.model(
            bev_image_input,
            imgsz=self.opt.img_size,
            conf=self.opt.conf_thres,
            iou=self.opt.iou_thres,
            device=self.device,
            verbose=False
        )
        result = results[0]

        # 3. Mask Processing
        # combined_mask_bev feeds final_filter/connected-components as
        # before; class_masks (one per model class, e.g. dashed/solid) is
        # kept alongside purely so each final component's dominant class
        # can be looked up afterwards without changing the existing
        # combine/filter/top-2 selection behavior.
        combined_mask_bev = np.zeros(result.orig_shape[:2], dtype=np.uint8)
        class_masks = {
            class_id: np.zeros(result.orig_shape[:2], dtype=np.uint8)
            for class_id in self.model.names
        }
        if result.masks is not None:
            confidences = result.boxes.conf
            classes = result.boxes.cls
            for i, mask_tensor in enumerate(result.masks.data):
                if confidences[i] >= self.opt.conf_thres:
                    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
                    if mask_np.shape != result.orig_shape[:2]:
                        mask_np = cv2.resize(
                            mask_np,
                            (result.orig_shape[1], result.orig_shape[0])
                        )
                    combined_mask_bev = np.maximum(combined_mask_bev, mask_np)
                    class_id = int(classes[i])
                    class_masks[class_id] = np.maximum(
                        class_masks[class_id], mask_np
                    )

        # solid는 기존 파이프라인(면적 10000px 이상 요구) 그대로 둔다 --
        # 잘 되고 있으니 안 건드림. dashed는 조각 하나가 그 기준보다
        # 훨씬 작아서 전부 걸러졌었다 -- dashed 클래스만 큰 면적 게이트를
        # 건너뛰고 작은 노이즈 제거(min_size=100)만 적용해 경로 생성에
        # 쓸 수 있게 한다.
        dashed_class_ids = {
            class_id for class_id, name in self.model.names.items()
            if 'dashed' in str(name).strip().lower()
        }
        dashed_mask = np.zeros_like(combined_mask_bev)
        solid_like_mask = np.zeros_like(combined_mask_bev)
        for class_id, mask in class_masks.items():
            if class_id in dashed_class_ids:
                dashed_mask = np.maximum(dashed_mask, mask)
            else:
                solid_like_mask = np.maximum(solid_like_mask, mask)

        solid_filtered = final_filter(solid_like_mask)
        dashed_filtered = morph_close(dashed_mask, ksize=5)
        dashed_filtered = remove_small_components(dashed_filtered, min_size=100)

        final_mask = keep_top2_components(
            np.maximum(solid_filtered, dashed_filtered), min_area=300
        )
        bev_im_for_drawing = bev_image_input.copy()

        # 4. Lane Extraction: component -> y별 중심점 추출
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8
        )
        current_detections = []

        if num_labels > 1:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 100:
                    component_mask = np.zeros_like(final_mask, dtype=np.uint8)
                    component_mask[labels == i] = 255

                    ys_center, xs_center = extract_centerline_points_from_component(
                        component_mask,
                        min_points_per_y=1,
                        y_step=1
                    )

                    if ys_center is None or len(ys_center) < 5:
                        continue

                    x_bottom = xs_center[-1]

                    # 이 성분과 가장 많이 겹치는 클래스 마스크를 그 성분의
                    # 클래스로 판정 (dashed/solid 구분, 추적/표시용).
                    class_id, class_name = None, None
                    best_overlap = 0
                    for candidate_id, class_mask in class_masks.items():
                        overlap = int(
                            np.count_nonzero(
                                (component_mask > 0) & (class_mask > 0)
                            )
                        )
                        if overlap > best_overlap:
                            best_overlap = overlap
                            class_id = candidate_id
                            class_name = self.model.names[candidate_id]

                    current_detections.append({
                        'xs': xs_center,
                        'ys': ys_center,
                        'x_bottom': x_bottom,
                        'class_id': class_id,
                        'class_name': class_name,
                    })

        current_detections.sort(key=lambda c: c['x_bottom'])

        # 5. Tracking
        left_lane_tracked = self.tracked_lanes['left']
        right_lane_tracked = self.tracked_lanes['right']
        current_left, current_right = None, None

        if len(current_detections) == 2:
            left_candidate, right_candidate = current_detections[0], current_detections[1]

            if self._lane_pair_is_plausible(
                left_candidate['xs'], left_candidate['ys'],
                right_candidate['xs'], right_candidate['ys'],
            ):
                current_left, current_right = left_candidate, right_candidate
            else:
                # y구간이 안 겹치거나(같은 선의 다른 조각일 가능성) 간격이
                # 너무 좁다 -- 둘 다 신뢰하지 않고, 차량에 더 가까운(더
                # 아래까지 내려온) 조각 하나만 채택한다.
                nearer = (
                    left_candidate
                    if float(left_candidate['ys'][-1]) >= float(right_candidate['ys'][-1])
                    else right_candidate
                )
                if nearer['x_bottom'] < self.bev_w / 2:
                    current_left = nearer
                else:
                    current_right = nearer

        elif len(current_detections) > 2:
            # 횡단보도 등으로 같은 프레임에 선이 3개 이상 잡히면
            # 좌우 페어링 대신 가장 오른쪽 선 하나만 신뢰하고 나머지는 버린다.
            current_right = current_detections[-1]

        elif len(current_detections) == 1:
            detected_lane = current_detections[0]

            dist_to_left = abs(detected_lane['x_bottom'] - left_lane_tracked['xs'][-1]) \
                if left_lane_tracked['xs'] is not None else float('inf')
            dist_to_right = abs(detected_lane['x_bottom'] - right_lane_tracked['xs'][-1]) \
                if right_lane_tracked['xs'] is not None else float('inf')

            if dist_to_left < dist_to_right and left_lane_tracked['xs'] is not None:
                current_left = detected_lane
            elif dist_to_right < dist_to_left and right_lane_tracked['xs'] is not None:
                current_right = detected_lane
            else:
                if detected_lane['x_bottom'] < self.bev_w / 2:
                    current_left = detected_lane
                else:
                    current_right = detected_lane

        if current_left is not None:
            left_fit_xs = current_left['xs']
            left_fit_ys = current_left['ys']
            left_lane_tracked['xs'] = current_left['xs']
            left_lane_tracked['ys'] = current_left['ys']
            left_lane_tracked['class_name'] = current_left['class_name']
            left_lane_tracked['age'] = 0
        else:
            left_lane_tracked['age'] += 1
            if left_lane_tracked['age'] > self.MAX_LANE_AGE:
                left_lane_tracked['xs'] = None
                left_lane_tracked['ys'] = None
                left_lane_tracked['class_name'] = None

        if current_right is not None:
            right_fit_xs = current_right['xs']
            right_fit_ys = current_right['ys']
            right_lane_tracked['xs'] = current_right['xs']
            right_lane_tracked['ys'] = current_right['ys']
            right_lane_tracked['class_name'] = current_right['class_name']
            right_lane_tracked['age'] = 0
        else:
            right_lane_tracked['age'] += 1
            if right_lane_tracked['age'] > self.MAX_LANE_AGE:
                right_lane_tracked['xs'] = None
                right_lane_tracked['ys'] = None
                right_lane_tracked['class_name'] = None

        final_left_xs, final_left_ys = left_lane_tracked['xs'], left_lane_tracked['ys']
        final_right_xs, final_right_ys = right_lane_tracked['xs'], right_lane_tracked['ys']

        if (
            final_left_xs is not None and final_right_xs is not None
            and not self._lane_pair_is_plausible(
                final_left_xs, final_left_ys, final_right_xs, final_right_ys
            )
        ):
            # 이번 프레임엔 한쪽만 새로 검출되고 반대쪽은 age 유지 로직으로
            # 남아있는 예전 값일 수 있다 -- 그 둘이 좌우로 말이 안 되면
            # (같은 선 조각이거나 y가 안 겹침) 더 최근에 갱신된 쪽만 쓴다.
            if left_lane_tracked['age'] <= right_lane_tracked['age']:
                final_right_xs, final_right_ys = None, None
            else:
                final_left_xs, final_left_ys = None, None

        lane_detected_bool = (final_left_xs is not None) or (final_right_xs is not None)
        self.pub_lane_status.publish(Bool(data=lane_detected_bool))

        # 6. Steering: 중앙 경로도 점 기반
        if lane_detected_bool:
            lane_width_pixels = self.LANE_WIDTH_M / self.m_per_pixel_x

            center_xs, center_ys = None, None

            if final_left_xs is not None and final_right_xs is not None:
                center_xs, center_ys = compute_fusion_centerline(
                    final_left_xs, final_left_ys,
                    final_right_xs, final_right_ys,
                    offset_px=lane_width_pixels / 2.0,
                    y_step=1
                )

            elif final_left_xs is not None:
                center_xs, center_ys = offset_polyline_points(
                    final_left_xs, final_left_ys,
                    offset_px=lane_width_pixels / 2.0, direction=+1.0
                )

            elif final_right_xs is not None:
                center_xs, center_ys = offset_polyline_points(
                    final_right_xs, final_right_ys,
                    offset_px=lane_width_pixels / 2.0, direction=-1.0
                )

            if center_xs is not None and len(center_xs) > 1:
                # 스무딩 및 B-Spline 보간 적용 (점의 개수를 100개로 촘촘하게 맞춤)
                center_xs, center_ys = smooth_and_interpolate_path(center_xs, center_ys, num_points=100, smooth_factor=10.0)
                
                self.tracked_center_path['xs'] = center_xs
                self.tracked_center_path['ys'] = center_ys
                center_fit_xs = center_xs
                center_fit_ys = center_ys

            if self.tracked_center_path['xs'] is not None and len(self.tracked_center_path['xs']) > 1:
                final_center_xs = self.tracked_center_path['xs']
                final_center_ys = self.tracked_center_path['ys']

                path_msg = Path()
                path_msg.header.frame_id = "base_link"
                path_msg.header.stamp = self.get_clock().now().to_msg()

                step = 20
                for idx in range(len(final_center_xs) - 1, -1, -step):
                    x_bev = final_center_xs[idx]
                    y_bev = final_center_ys[idx]
                    x_veh, y_veh = self.image_to_vehicle((x_bev, y_bev))

                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = float(x_veh)
                    pose.pose.position.y = float(y_veh)
                    pose.pose.position.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path_msg.poses.append(pose)

                self.pub_path.publish(path_msg)

                v_norm = (self.current_throttle - self.THROTTLE_MIN_FOR_LD) / (
                    self.THROTTLE_MAX_FOR_LD - self.THROTTLE_MIN_FOR_LD + 1e-6
                )
                v_norm = np.clip(v_norm, 0.0, 1.0)

                LD_target = self.MIN_LOOKAHEAD_DISTANCE + (
                    self.MAX_LOOKAHEAD_DISTANCE - self.MIN_LOOKAHEAD_DISTANCE
                ) * v_norm

                dynamic_lookahead_distance = float(np.clip(
                    LD_target,
                    self.MIN_LOOKAHEAD_DISTANCE,
                    self.MAX_LOOKAHEAD_DISTANCE
                ))

                self.last_valid_ld = dynamic_lookahead_distance
                self.pub_lookahead.publish(Float32(data=dynamic_lookahead_distance))

                transformed_points = []
                for idx in range(len(final_center_xs) - 1, -1, -1):
                    x_bev = final_center_xs[idx]
                    y_bev = final_center_ys[idx]
                    x_veh, y_veh_right = self.image_to_vehicle((x_bev, y_bev))
                    transformed_points.append((x_veh, y_veh_right, x_bev, y_bev))

                lookahead_point = None
                Ld = dynamic_lookahead_distance

                if self.use_arc_length_lookahead:
                    # 현재 방식: 최근접 제어점부터 경로 arc length를 누적해 Ld 지점을 선택
                    nearest_idx = 0
                    nearest_dist = float('inf')
                    for i, (x_rel, y_rel, _, _) in enumerate(transformed_points):
                        dist = hypot(x_rel, y_rel)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = i

                    lookahead_point = transformed_points[nearest_idx]
                    # 프레임 하단(경로 시작)이 아닌 후륜축으로부터의 거리를 누적 길이에 포함
                    accum = nearest_dist
                    
                    if accum >= Ld:
                        lookahead_point = transformed_points[nearest_idx]
                    else:
                        for i in range(nearest_idx, len(transformed_points) - 1):
                            x0, y0, xb0, yb0 = transformed_points[i]
                            x1, y1, xb1, yb1 = transformed_points[i + 1]
                            seg_len = hypot(x1 - x0, y1 - y0)
                            if seg_len <= 1e-6:
                                continue

                            if accum + seg_len >= Ld:
                                ratio = (Ld - accum) / seg_len
                                lx = x0 + ratio * (x1 - x0)
                                ly = y0 + ratio * (y1 - y0)
                                lbx = xb0 + ratio * (xb1 - xb0)
                                lby = yb0 + ratio * (yb1 - yb0)
                                lookahead_point = (lx, ly, lbx, lby)
                                break

                            accum += seg_len
                            lookahead_point = (x1, y1, xb1, yb1)
                else:
                    # 기존 방식: 전방점(x_rel > 0) 중 |dist - Ld| 최소점을 선택
                    best_diff = float('inf')
                    for x_rel, y_rel, x_b, y_b in transformed_points:
                        if x_rel < 0:
                            continue
                        dist = hypot(x_rel, y_rel)
                        diff = abs(dist - Ld)
                        if diff < best_diff:
                            best_diff = diff
                            lookahead_point = (x_rel, y_rel, x_b, y_b)

                if lookahead_point is not None:
                    x_veh, y_veh_right, x_bev, y_bev = lookahead_point
                    goal_point_bev = (int(x_bev), int(y_bev))
                    steer_rad = atan2(
                        2.0 * self.L * y_veh_right,
                        x_veh ** 2 + y_veh_right ** 2
                    )

                    # pure_pursuit.py와 동일 부호(목표가 왼쪽이면 양수) +
                    # 동일 게인 적용 후 클램프.
                    raw_steer_deg = float(np.clip(
                        degrees(steer_rad) * self.STEERING_GAIN,
                        -self.MAX_STEER_DEG,
                        self.MAX_STEER_DEG
                    ))

                    delta = np.clip(
                        raw_steer_deg - self.prev_steer_deg,
                        -self.MAX_STEER_RATE,
                        self.MAX_STEER_RATE
                    )
                    steer_deg = float(self.prev_steer_deg + delta)

                    self.prev_steer_deg = steer_deg
                    self.pub_steering.publish(Float32(data=steer_deg))
                    self.pub_lane_valid.publish(Bool(data=True))
                else:
                    self.pub_lane_valid.publish(Bool(data=False))
            else:
                dynamic_lookahead_distance = float(self.last_valid_ld)
                self.pub_lookahead.publish(Float32(data=float(self.last_valid_ld)))
                self.pub_lane_valid.publish(Bool(data=False))
        else:
            dynamic_lookahead_distance = float(self.last_valid_ld)
            self.pub_lookahead.publish(Float32(data=float(self.last_valid_ld)))
            self.pub_lane_valid.publish(Bool(data=False))

        # 7. Visualization
        try:
            annotated_frame = result.plot()
        except Exception:
            pass

        # 점열을 그냥 이어서 차선/경로 표현
        overlay_point_path(bev_im_for_drawing, final_left_xs, final_left_ys, color=(255, 0, 0), thickness=2)
        overlay_point_path(bev_im_for_drawing, final_right_xs, final_right_ys, color=(0, 0, 255), thickness=2)
        overlay_point_path(
            bev_im_for_drawing,
            self.tracked_center_path['xs'],
            self.tracked_center_path['ys'],
            color=(0, 255, 0),
            thickness=3
        )

        # 점 시각화
        overlay_points(bev_im_for_drawing, left_fit_xs, left_fit_ys, color=(255, 0, 0), radius=2)
        overlay_points(bev_im_for_drawing, right_fit_xs, right_fit_ys, color=(0, 0, 255), radius=2)
        overlay_points(bev_im_for_drawing, center_fit_xs, center_fit_ys, color=(255, 255, 255), radius=2)

        if goal_point_bev is not None:
            cv2.circle(bev_im_for_drawing, goal_point_bev, 10, (0, 255, 255), -1)

        if lane_detected_bool:
            y_offset_px = int(self.y_offset_m / self.m_per_pixel_y)
            true_origin_pt = (int(self.bev_w / 2), self.bev_h - 1 + y_offset_px)

            radius_x_px = int(dynamic_lookahead_distance / self.m_per_pixel_x)
            radius_y_px = int(dynamic_lookahead_distance / self.m_per_pixel_y)

            cv2.ellipse(
                bev_im_for_drawing,
                true_origin_pt,
                (radius_x_px, radius_y_px),
                0,
                0,
                360,
                (255, 255, 255),
                2
            )

        steer_text = f"Steer: {steer_deg:.1f} deg" if steer_deg is not None else "Steer: N/A"
        cv2.putText(bev_im_for_drawing, steer_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Lane Detected: {lane_detected_bool}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(bev_im_for_drawing, f"Lookahead: {dynamic_lookahead_distance:.2f}m", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        left_class = left_lane_tracked['class_name'] or "N/A"
        right_class = right_lane_tracked['class_name'] or "N/A"
        cv2.putText(
            bev_im_for_drawing,
            f"Left: {left_class}  Right: {right_class}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        original_with_drivable_area = self.draw_drivable_area_on_original(
            im0s,
            final_left_xs,
            final_left_ys,
            final_right_xs,
            final_right_ys,
            self.tracked_center_path['xs'],
            self.tracked_center_path['ys'],
        )

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.format = "jpeg"

        success, encoded_img = cv2.imencode('.jpg', original_with_drivable_area)
        if success:
            msg.data = encoded_img.tobytes()
            self.pub_drivable_area.publish(msg)

        #cv2.imshow("Original Camera View", original_with_drivable_area)
        cv2.imshow("Final Path & Logs (on BEV)", bev_im_for_drawing)
        cv2.imshow("Roboflow Detections (on BEV)", annotated_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()

    default_weights = _default_resource_path('best_ver2.pt')
    default_params = _default_resource_path('bev_params_0803.npz')
    default_calib = _default_calibration_path()

    parser.add_argument('--weights', default=default_weights, help='Path to model weights')
    parser.add_argument('--param-file', default=default_params, help='Path to BEV parameters file')
    parser.add_argument('--calib-file', default=default_calib, help='Path to camera calibration pkl file')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.8)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', default='0')
    parser.add_argument(
        '--topic',
        type=str,
        default='/camera/lane/raw/compressed',
        help='ROS 2 Image Topic',
    )

    opt, _ = parser.parse_known_args()

    node = LaneFollowerNode(opt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
