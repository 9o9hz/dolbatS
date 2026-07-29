#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


# ============================================================
# 사용자 설정
# ============================================================

# 현재 작업 디렉터리와 관계없이 utils 및 저장소 루트를 찾는다.
UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent

# 촬영한 사진이 들어 있는 폴더
IMAGE_DIR = UTILS_DIR / "calibration_images"

# 실제 차선 카메라 운용 해상도. 다른 해상도의 이미지는 리사이즈하지 않고 제외한다.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
EXPECTED_IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# 체커보드 내부 코너 개수: (가로, 세로)
# 예: 가로 10칸 x 세로 7칸 체커보드라면 내부 코너는 9 x 6
# 현재 (10, 7) 설정에는 일반적으로 가로 11칸 x 세로 8칸 보드가 필요하다.
CHECKERBOARD_SIZE = (10, 7)

# 체커보드 한 칸의 실제 크기(mm)
# 출력물을 자로 직접 측정한 값을 넣는 것이 가장 정확함
SQUARE_SIZE_MM = 25.0

# ROS 패키지가 실제로 설치하는 config 파일을 바로 갱신한다.
OUTPUT_FILE = (
    PROJECT_ROOT
    / "src"
    / "lane_vision_pkg"
    / "config"
    / "camera_calibration.npz"
)

# 코너 검출 결과와 보정 예시를 저장할 폴더
DEBUG_DIR = UTILS_DIR / "calibration_debug"

# 이 수보다 적으면 결과를 저장하지 않는다.
MIN_VALID_IMAGES = 10

# 이미지별 재투영 RMSE가 이 값을 넘으면 해당 이미지를 제외하고 다시 보정한다.
MAX_VIEW_RMSE_PX = 1.0

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_checkerboard_corners(
    image: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """
    체커보드 내부 코너를 찾는다.

    먼저 검출 성능이 좋은 findChessboardCornersSB()를 사용하고,
    실패하면 기존 findChessboardCorners() 방식으로 다시 시도한다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 명암 차이를 조금 개선
    equalized = cv2.equalizeHist(gray)

    # 개선된 SB 방식
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            equalized,
            CHECKERBOARD_SIZE,
            flags=sb_flags,
        )

        if found and corners is not None:
            return True, corners.astype(np.float32)

    # 기존 방식으로 재시도
    normal_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        equalized,
        CHECKERBOARD_SIZE,
        flags=normal_flags,
    )

    if not found or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )

    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    return True, refined


def make_object_points() -> np.ndarray:
    """
    체커보드 내부 코너들의 실제 좌표를 생성한다.

    체커보드는 평면이므로 z 좌표는 모두 0이다.
    """
    columns, rows = CHECKERBOARD_SIZE

    object_points = np.zeros(
        (columns * rows, 3),
        dtype=np.float32,
    )

    object_points[:, :2] = (
        np.mgrid[0:columns, 0:rows]
        .T
        .reshape(-1, 2)
    )

    object_points *= SQUARE_SIZE_MM

    return object_points


def calculate_reprojection_errors(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    rotation_vectors: tuple[np.ndarray, ...],
    translation_vectors: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
) -> tuple[float, float, list[float]]:
    """
    전체 점별 평균 오차, 전체 RMSE, 사진별 RMSE를 픽셀 단위로 계산한다.

    cv2.norm(..., NORM_L2) / N은 RMSE가 아니므로 사용하지 않는다.
    """
    if len(object_points_list) != len(image_points_list):
        raise ValueError("3D 점 목록과 2D 점 목록의 길이가 다릅니다.")

    if (
        len(rotation_vectors) != len(object_points_list)
        or len(translation_vectors) != len(object_points_list)
    ):
        raise ValueError("캘리브레이션 자세 벡터 수가 이미지 수와 다릅니다.")

    individual_rmse: list[float] = []
    all_squared_distances: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []

    for index, object_points in enumerate(object_points_list):
        projected_points, _ = cv2.projectPoints(
            object_points,
            rotation_vectors[index],
            translation_vectors[index],
            camera_matrix,
            distortion_coefficients,
        )

        observed = np.asarray(
            image_points_list[index],
            dtype=np.float64,
        ).reshape(-1, 2)
        projected = np.asarray(
            projected_points,
            dtype=np.float64,
        ).reshape(-1, 2)

        if observed.shape != projected.shape:
            raise ValueError(
                f"{index}번 이미지의 관측점과 투영점 shape가 다릅니다: "
                f"{observed.shape} != {projected.shape}"
            )

        residual = observed - projected
        squared_distances = np.sum(residual * residual, axis=1)
        distances = np.sqrt(squared_distances)

        view_rmse = float(np.sqrt(np.mean(squared_distances)))
        individual_rmse.append(view_rmse)
        all_squared_distances.append(squared_distances)
        all_distances.append(distances)

    if not all_squared_distances:
        raise ValueError("재투영 오차를 계산할 이미지가 없습니다.")

    squared = np.concatenate(all_squared_distances)
    distances = np.concatenate(all_distances)

    mean_error = float(np.mean(distances))
    global_rmse = float(np.sqrt(np.mean(squared)))

    if not (
        np.isfinite(mean_error)
        and np.isfinite(global_rmse)
        and np.all(np.isfinite(individual_rmse))
    ):
        raise ValueError("재투영 오차에 NaN 또는 무한대가 포함되어 있습니다.")

    return mean_error, global_rmse, individual_rmse


def collect_image_paths() -> list[Path]:
    """
    calibration_images 폴더에서 이미지 파일을 대소문자 구분 없이 찾는다.
    """
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"'{IMAGE_DIR}' 폴더가 없습니다.\n"
            "640x480으로 촬영한 체커보드 사진을 넣어주세요."
        )

    image_paths = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]

    return sorted(
        image_paths,
        key=lambda path: path.name.casefold(),
    )


def prepare_debug_directory() -> None:
    """
    이번 실행에서 만드는 파일만 정리해 이전 결과가 섞이지 않게 한다.
    """
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    for path in DEBUG_DIR.iterdir():
        if not path.is_file():
            continue

        if (
            path.name.startswith("detected_")
            and path.suffix.lower() == ".jpg"
        ) or path.name == "original_vs_undistorted.jpg":
            path.unlink()


def write_image_checked(path: Path, image: np.ndarray) -> None:
    """OpenCV 이미지 저장 실패를 즉시 오류로 처리한다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")


def save_detection_preview(
    image: np.ndarray,
    corners: np.ndarray,
    image_path: Path,
    image_index: int,
) -> None:
    """
    검출된 코너를 그린 이미지를 calibration_debug 폴더에 저장한다.
    """
    preview = image.copy()

    cv2.drawChessboardCorners(
        preview,
        CHECKERBOARD_SIZE,
        corners,
        True,
    )

    output_path = (
        DEBUG_DIR
        / f"detected_{image_index:04d}_{image_path.stem}.jpg"
    )
    write_image_checked(output_path, preview)


def project_relative_path(path: Path) -> str:
    """NPZ 메타데이터에는 Ubuntu에서 읽기 쉬운 POSIX 경로를 저장한다."""
    resolved = path.resolve()

    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def save_npz_atomically(
    output_path: Path,
    **arrays: object,
) -> Path | None:
    """
    임시 파일을 완성한 뒤 교체하고 기존 결과는 previous 파일로 보관한다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous_path: Path | None = None
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}_",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez(temporary_file, **arrays)

        # NamedTemporaryFile의 0600 권한이 최종 파일에 남지 않게 한다.
        # ROS가 별도 사용자/서비스로 실행돼도 읽을 수 있는 config 권한이다.
        temporary_path.chmod(0o644)

        if output_path.exists():
            previous_path = output_path.with_name(
                f".{output_path.stem}.previous{output_path.suffix}"
            )
            shutil.copy2(output_path, previous_path)

        temporary_path.replace(output_path)

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return previous_path


def calculate_dataset_coverage(
    image_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, float]:
    """
    보드 중심 위치와 크기 변화를 수치화한다.

    캘리브레이션 가능 여부를 임의의 기준으로 막지는 않고, 촬영 구도가
    한곳에 몰렸을 때 사용자가 다시 촬영할 수 있도록 경고한다.
    """
    if not image_points_list:
        raise ValueError("데이터셋 분포를 계산할 코너가 없습니다.")

    width, height = image_size
    centers: list[np.ndarray] = []
    board_area_ratios: list[float] = []

    for corners in image_points_list:
        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        centers.append(np.mean(points, axis=0))

        hull = cv2.convexHull(points)
        area = float(cv2.contourArea(hull))
        board_area_ratios.append(area / float(width * height))

    center_array = np.asarray(centers, dtype=np.float64)
    area_array = np.asarray(board_area_ratios, dtype=np.float64)

    center_span_x = float(
        (np.max(center_array[:, 0]) - np.min(center_array[:, 0]))
        / width
    )
    center_span_y = float(
        (np.max(center_array[:, 1]) - np.min(center_array[:, 1]))
        / height
    )

    min_area_ratio = float(np.min(area_array))
    max_area_ratio = float(np.max(area_array))
    area_scale_ratio = float(
        max_area_ratio / max(min_area_ratio, np.finfo(np.float64).eps)
    )

    return {
        "center_span_x_ratio": center_span_x,
        "center_span_y_ratio": center_span_y,
        "min_board_area_ratio": min_area_ratio,
        "max_board_area_ratio": max_area_ratio,
        "board_area_scale_ratio": area_scale_ratio,
    }


def print_dataset_coverage(coverage: dict[str, float]) -> None:
    """촬영 위치와 크기 다양성을 출력하고 부족한 경우 경고한다."""
    center_span_x = coverage["center_span_x_ratio"]
    center_span_y = coverage["center_span_y_ratio"]
    area_scale_ratio = coverage["board_area_scale_ratio"]

    print()
    print("촬영 분포")
    print("-" * 65)
    print(f"보드 중심 X 이동 범위 : {center_span_x * 100.0:.1f}%")
    print(f"보드 중심 Y 이동 범위 : {center_span_y * 100.0:.1f}%")
    print(f"보드 면적 크기 비율   : {area_scale_ratio:.2f}배")

    if center_span_x < 0.25:
        print(
            "[경고] 보드 중심의 좌우 이동 범위가 좁습니다. "
            "화면 좌우 가장자리에서도 촬영하세요."
        )

    if center_span_y < 0.20:
        print(
            "[경고] 보드 중심의 상하 이동 범위가 좁습니다. "
            "화면 위아래에서도 촬영하세요."
        )

    if area_scale_ratio < 1.5:
        print(
            "[경고] 보드 크기 변화가 작습니다. "
            "카메라와의 거리를 바꿔 촬영하세요."
        )


def calibrate() -> None:
    image_paths = collect_image_paths()

    if not image_paths:
        raise FileNotFoundError(
            f"'{IMAGE_DIR}' 폴더에서 사진을 찾지 못했습니다.\n"
            "640x480으로 촬영한 사진을 utils/calibration_images 폴더에 넣어주세요."
        )

    prepare_debug_directory()

    print("=" * 65)
    print("카메라 캘리브레이션 시작")
    print("=" * 65)
    print(f"사진 폴더        : {IMAGE_DIR.resolve()}")
    print(f"전체 사진 수     : {len(image_paths)}")
    print(f"필수 영상 해상도 : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"체커보드 내부 코너: {CHECKERBOARD_SIZE}")
    print(f"한 칸 크기       : {SQUARE_SIZE_MM} mm")
    print(f"최소 유효 사진 수: {MIN_VALID_IMAGES}")
    print(f"사진별 RMSE 한계 : {MAX_VIEW_RMSE_PX:.2f} px")
    print("=" * 65)

    base_object_points = make_object_points()

    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    valid_paths: list[Path] = []
    failed_records: list[tuple[Path, str]] = []

    expected_corner_count = CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1]

    for image_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[읽기 실패] {image_path.name}")
            failed_records.append((image_path, "이미지 읽기 실패"))
            continue

        current_size = (image.shape[1], image.shape[0])

        if current_size != EXPECTED_IMAGE_SIZE:
            print(
                f"[해상도 불일치] {image_path.name}: "
                f"{current_size[0]}x{current_size[1]}, "
                f"필수 해상도: {IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            )
            failed_records.append(
                (
                    image_path,
                    f"해상도 {current_size[0]}x{current_size[1]}",
                )
            )
            continue

        found, corners = find_checkerboard_corners(image)

        if found and corners is not None:
            corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

            if (
                len(corners) != expected_corner_count
                or not np.all(np.isfinite(corners))
            ):
                print(f"[코너값 오류] {image_path.name}")
                failed_records.append(
                    (image_path, "코너 수 또는 좌표값 오류")
                )
                continue

            object_points_list.append(base_object_points.copy())
            image_points_list.append(corners)
            valid_paths.append(image_path)

            save_detection_preview(
                image,
                corners,
                image_path,
                image_index,
            )
            print(f"[검출 성공] {image_path.name}")

        else:
            failed_records.append((image_path, "체커보드 검출 실패"))
            print(f"[검출 실패] {image_path.name}")

    detected_count = len(valid_paths)

    print("=" * 65)
    print(f"검출 성공: {detected_count}장")
    print(f"제외/실패: {len(failed_records)}장")

    if detected_count < MIN_VALID_IMAGES:
        raise RuntimeError(
            f"640x480 유효 사진이 {detected_count}장뿐입니다. "
            f"최소 {MIN_VALID_IMAGES}장이 필요합니다.\n"
            "CHECKERBOARD_SIZE와 입력 사진 해상도를 확인하세요."
        )

    image_size = EXPECTED_IMAGE_SIZE
    rejected_outliers: list[tuple[Path, float]] = []

    # RMSE가 큰 사진을 한 장씩 제거하고 다시 보정한다.
    while True:
        (
            rms_error,
            camera_matrix,
            distortion_coefficients,
            rotation_vectors,
            translation_vectors,
        ) = cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            image_size,
            None,
            None,
        )

        (
            mean_error,
            calculated_global_rmse,
            individual_errors,
        ) = calculate_reprojection_errors(
            object_points_list,
            image_points_list,
            rotation_vectors,
            translation_vectors,
            camera_matrix,
            distortion_coefficients,
        )

        if not np.isfinite(rms_error):
            raise RuntimeError("OpenCV 캘리브레이션 RMS가 유효하지 않습니다.")

        if not np.isclose(
            float(rms_error),
            calculated_global_rmse,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(
                "OpenCV RMS와 직접 계산한 RMS가 일치하지 않습니다: "
                f"{float(rms_error):.8f} != "
                f"{calculated_global_rmse:.8f}"
            )

        worst_index = int(np.argmax(individual_errors))
        worst_error = individual_errors[worst_index]

        if worst_error <= MAX_VIEW_RMSE_PX:
            break

        if len(valid_paths) - 1 < MIN_VALID_IMAGES:
            raise RuntimeError(
                f"{valid_paths[worst_index].name}의 재투영 RMSE가 "
                f"{worst_error:.3f}px이지만, 이 사진을 제외하면 "
                f"최소 {MIN_VALID_IMAGES}장보다 적어집니다.\n"
                "사진을 더 촬영한 뒤 다시 실행하세요."
            )

        rejected_path = valid_paths.pop(worst_index)
        rejected_outliers.append((rejected_path, worst_error))
        object_points_list.pop(worst_index)
        image_points_list.pop(worst_index)

        print(
            f"[고오차 제외 후 재보정] {rejected_path.name}: "
            f"RMSE {worst_error:.6f} px"
        )

    valid_count = len(valid_paths)

    if camera_matrix.shape != (3, 3):
        raise RuntimeError(
            f"camera_matrix shape가 잘못됐습니다: {camera_matrix.shape}"
        )

    if not (
        np.all(np.isfinite(camera_matrix))
        and np.all(np.isfinite(distortion_coefficients))
    ):
        raise RuntimeError("캘리브레이션 결과에 NaN 또는 무한대가 있습니다.")

    if camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        raise RuntimeError("계산된 초점거리가 0 이하입니다.")

    coverage = calculate_dataset_coverage(
        image_points_list,
        image_size,
    )
    print_dataset_coverage(coverage)

    # alpha=0: 검은 영역 최소화
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        image_size,
        alpha=0.0,
        newImgSize=image_size,
    )

    if not np.all(np.isfinite(new_camera_matrix)):
        raise RuntimeError("새 카메라 행렬에 NaN 또는 무한대가 있습니다.")

    # 사진별 오차 출력
    sorted_results = sorted(
        zip(valid_paths, individual_errors),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("사진별 재투영 RMSE: 높은 순서")
    print("-" * 65)

    for image_path, error in sorted_results:
        warning = (
            "  <-- 한계값 근접"
            if error > MAX_VIEW_RMSE_PX * 0.8
            else ""
        )
        print(f"{image_path.name:<35} {error:.6f} px{warning}")

    # 보정 전후 비교 예시 저장
    sample_image = cv2.imread(str(valid_paths[0]))

    if sample_image is None:
        raise RuntimeError(
            f"비교 이미지를 다시 읽지 못했습니다: {valid_paths[0]}"
        )

    undistorted = cv2.undistort(
        sample_image,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix,
    )

    comparison = np.hstack((sample_image, undistorted))

    cv2.putText(
        comparison,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        comparison,
        "Undistorted",
        (image_size[0] + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    comparison_path = DEBUG_DIR / "original_vs_undistorted.jpg"
    write_image_checked(comparison_path, comparison)

    # 모든 계산과 디버그 이미지 저장이 성공한 뒤 최종 NPZ를 교체한다.
    previous_file = save_npz_atomically(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        new_camera_matrix=new_camera_matrix,
        roi=np.asarray(roi, dtype=np.int32),
        image_width=np.int32(image_size[0]),
        image_height=np.int32(image_size[1]),
        checkerboard_columns=np.int32(CHECKERBOARD_SIZE[0]),
        checkerboard_rows=np.int32(CHECKERBOARD_SIZE[1]),
        square_size_mm=np.float32(SQUARE_SIZE_MM),
        rms_error=np.float64(rms_error),
        global_reprojection_rmse=np.float64(
            calculated_global_rmse
        ),
        mean_reprojection_error=np.float64(mean_error),
        individual_reprojection_errors=np.asarray(
            individual_errors,
            dtype=np.float64,
        ),
        reprojection_error_metric=np.asarray(
            "mean=per_corner_euclidean_px;"
            "individual=per_view_rmse_px"
        ),
        calibration_model=np.asarray(
            "opencv_pinhole_brown_conrady"
        ),
        detected_image_count=np.int32(detected_count),
        rejected_outlier_count=np.int32(len(rejected_outliers)),
        rejected_outlier_paths=np.asarray(
            [
                project_relative_path(path)
                for path, _ in rejected_outliers
            ],
            dtype=np.str_,
        ),
        rejected_outlier_rmse=np.asarray(
            [error for _, error in rejected_outliers],
            dtype=np.float64,
        ),
        valid_image_paths=np.asarray(
            [project_relative_path(path) for path in valid_paths],
            dtype=np.str_,
        ),
        failed_image_paths=np.asarray(
            [
                project_relative_path(path)
                for path, _ in failed_records
            ],
            dtype=np.str_,
        ),
        failed_image_reasons=np.asarray(
            [reason for _, reason in failed_records],
            dtype=np.str_,
        ),
        center_span_x_ratio=np.float64(
            coverage["center_span_x_ratio"]
        ),
        center_span_y_ratio=np.float64(
            coverage["center_span_y_ratio"]
        ),
        min_board_area_ratio=np.float64(
            coverage["min_board_area_ratio"]
        ),
        max_board_area_ratio=np.float64(
            coverage["max_board_area_ratio"]
        ),
        board_area_scale_ratio=np.float64(
            coverage["board_area_scale_ratio"]
        ),
    )

    print()
    print("=" * 65)
    print("캘리브레이션 완료")
    print("=" * 65)
    print(f"사용된 사진 수       : {valid_count}")
    print(f"고오차 제외 사진 수  : {len(rejected_outliers)}")
    print(f"영상 해상도          : {image_size[0]} x {image_size[1]}")
    print(f"전체 재투영 RMSE     : {rms_error:.6f} px")
    print(f"점별 평균 거리 오차  : {mean_error:.6f} px")
    print(f"캘리브레이션 파일    : {OUTPUT_FILE.resolve()}")

    if previous_file is not None:
        print(f"이전 파일 백업       : {previous_file.resolve()}")

    print(f"코너 검출 확인 폴더  : {DEBUG_DIR.resolve()}")
    print(f"보정 비교 이미지     : {comparison_path.resolve()}")
    print()
    print("Camera Matrix")
    print(camera_matrix)
    print()
    print("Distortion Coefficients")
    print(distortion_coefficients)
    print()
    print(f"ROI: {roi}")
    print("=" * 65)

    if rms_error < 0.3:
        print("[평가] 매우 양호")
    elif rms_error < 0.5:
        print("[평가] 양호")
    elif rms_error < 1.0:
        print(
            "[평가] 사용 가능. 촬영 분포와 RMSE가 높은 사진을 "
            "확인하면 더 좋아질 수 있습니다."
        )
    else:
        print(
            "[평가] 오차가 큽니다. calibration_debug의 검출 결과와 "
            "오차가 큰 사진을 확인하세요."
        )

    if rejected_outliers:
        print()
        print("고오차로 제외한 사진")
        print("-" * 65)
        for rejected_path, error in rejected_outliers:
            print(f"- {rejected_path.name}: {error:.6f} px")

    if failed_records:
        print()
        print("검출 전에 제외되거나 실패한 사진")
        print("-" * 65)
        for failed_path, reason in failed_records:
            print(f"- {failed_path.name}: {reason}")


def main() -> int:
    try:
        calibrate()

    except KeyboardInterrupt:
        print()
        print("[중단] 사용자가 캘리브레이션을 중단했습니다.")
        return 130

    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as error:
        print()
        print("[오류]")
        print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
