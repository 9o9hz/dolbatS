# 주행 설정·데이터 생성 도구

이 폴더의 파일은 실시간 주행 중에는 사용되지 않는다. 카메라
캘리브레이션, BEV 설정, rosbag 프레임 확인 및 YOLO 학습 데이터 생성을
위한 오프라인 도구와 작업 산출물을 모아 둔다.

주요 도구:

- `capture_calibration_images.py`: 640×480 체커보드 이미지 수집
- `calibrate_from_images.py`: 카메라 캘리브레이션 생성 및
  `src/lane_vision_pkg/config/camera_calibration.npz` 갱신
- `webcam_bev_drag_checkerboard_live.py`: 실제 비율 BEV 설정 생성
- `captured_frame_bev_parallel_roi.py`: 캡처 프레임 기반 BEV ROI 설정
- `rosbag_frame_capture_viewer.py`: rosbag 프레임 확인·캡처
- `rosbag_bev_triple_viewer.py`: 원본/ROI/BEV 비교
- `rosbag_bev_metric_roi_editor.py`: metric BEV ROI 편집
- `export_rosbag_bev_dataset.py`: YOLO 라벨링용 BEV 데이터셋 생성

도구끼리의 Python import와 기본 데이터 경로는 이 `utils/` 폴더를
기준으로 유지된다. 실제 주행 노드는 이 폴더를 import하지 않는다.

## 실시간 ROS BEV 설정 + 체커보드 검사

`webcam_bev_drag_checkerboard_live.py` 하나에서 usb_cam 영상 정지, BEV 네 점
선택, 25mm 체커보드의 정사각 비율·간격·직각 검사를 함께 수행한다. 목적은
주행 카메라의 원근 영상을 바닥 기준 BEV로 펴고, 실제 정사각형 체커보드가
BEV에서도 정사각형·등간격·직각으로 보이는지 수치로 검증하는 것이다.

### 준비물

- Logitech C920 웹캠(현재 일반 영상 장치 `/dev/video2`)
- `640×480`, 30 FPS usb_cam 설정
- 내부 코너 `10×7`, 실제 한 칸 `25mm × 25mm` 체커보드
- `utils/camera_calibration.npz`

체커보드는 반드시 실제 주행 바닥과 같은 평면에 평평하게 놓는다. 상자나
기울어진 판 위에 놓으면 그 평면에만 맞는 BEV가 만들어져 도로에서는 틀어진다.

### 실행

```bash
# 터미널 1: Logitech C920 (/dev/video2)
source /opt/ros/humble/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 \
  -p image_width:=640 -p image_height:=480 -p framerate:=30.0 \
  -p io_method:=mmap -p pixel_format:=mjpeg2rgb \
  -p camera_name:=lane_camera -p frame_id:=lane_camera \
  -r image_raw:=/camera/lane/raw \
  -r image_raw/compressed:=/camera/lane/raw/compressed \
  -r camera_info:=/camera/lane/camera_info
```

```bash
# 터미널 2: BEV 설정 + 체커보드 검사
source /opt/ros/humble/setup.bash
python3 utils/webcam_bev_drag_checkerboard_live.py \
  --topic /camera/lane/raw/compressed \
  --transport compressed
```

`SPACE`로 화면을 정지한 뒤 `LB, RB, LT, RT` 순서로 클릭한다. RB의 y는
LB에, RT의 y는 LT에 자동 정렬된다. 원본 영상 주변의 회색 확장 캔버스도
선택할 수 있어 음수 또는 640x480 바깥 좌표로 확장 BEV를 만들 수 있다.
BEV 화면에서 `X/Y` 비율이 1에 가깝고
간격 CV와 직각 오차가 작을수록 올바른 BEV다. `S`는 품질 검사를 통과한
설정만 NPZ와 분석 보고서로 저장한다.

### 조작

- `SPACE`: 실시간 화면 정지/재개
- 왼쪽 클릭: `LB → RB → LT → RT` 선택 또는 기존 점 드래그
- 오른쪽 클릭: 마지막 점 취소
- `R`: 네 점과 검사 결과 초기화
- `S`: 강한 체커보드 재검사 후 저장
- `Q` 또는 `Esc`: 종료

네 점이 완성되면 체커보드 검출 성공 여부와 관계없이 우선 640×640 직접 BEV를
표시한다. 선택용 원본과 `BEV Snapshot + Checkerboard Inspection`은 점을
찍은 사진으로 고정되어 ROI를 조절하며 체커보드 수치를 비교할 수 있다.
`Live Undistorted Camera`와 `Live BEV Preview`는 동일한 선택 좌표를 최신
카메라 프레임에 적용해 함께 갱신된다. 원본 체커보드 평면까지 검출되면 실제
25mm 배율을 이용한 metric BEV를 계산한다.

### BEV 검사 화면 해석

| 표시 | 의미 | 좋은 값 |
|---|---|---:|
| `Checkerboard: DETECTED` | 10×7 내부 코너 검출 성공 | `DETECTED` |
| `Physical square` | 체커보드 실제 한 칸 크기 | `25.0 mm` |
| `X px/square` | 한 칸의 평균 가로 픽셀 길이 | Y와 동일 |
| `Y px/square` | 한 칸의 평균 세로 픽셀 길이 | X와 동일 |
| `X/Y CV` | 가로·세로 칸 간격 불균일 정도 | 각각 0%에 가까움 |
| `X/Y square difference` | 한 칸의 가로·세로 크기 차이 | 0%에 가까움 |
| `Grid orthogonality error` | 격자가 90°에서 벗어난 정도 | 0°에 가까움 |

`QUALITY PASS` 기준은 다음과 같다.

- 한 칸의 가로/세로 크기 차이: 2% 이하
- 가로 간격 CV: 5% 이하
- 세로 간격 CV: 5% 이하
- 격자 직각 오차: 2° 이하
- metric BEV의 목표 25mm 배율 오차: 3% 이하

예를 들어 X/Y 차이가 0.01%이고 CV가 2%대라도 직각 오차가 11°이면
`QUALITY FAIL`이다. 이때는 상단 또는 하단 좌우 점의 x를 조절해 격자를
직각으로 맞춘다. CV가 크면 체커보드가 휘었거나 바닥과 다른 평면에 있는지
먼저 확인한다.

### 확장 BEV

회색 여백을 클릭하면 왼쪽·위쪽은 음수 좌표, 오른쪽은 x=640 이상, 아래쪽은
y=480 이상으로 저장된다. homography와 NPZ는 이 좌표를 그대로 지원한다.
다만 카메라가 촬영하지 않은 영역의 영상 정보가 새로 생기는 것은 아니므로,
ROI를 영상 밖으로 크게 확장하면 BEV 가장자리가 검게 나타날 수 있다.

### 저장 결과

- `utils/bev_params_y_auto.npz`: 주행용 BEV 변환 설정. 기존
  `src/drive_pkg/resource/bev_params_0731.npz`와 동일한 네 키
  (`src_points`, `dst_points`, `warp_w`, `warp_h`)와 동일한 자료형/배열
  구조로 저장되어 `drive_pkg`에서 그대로 읽을 수 있다.
- `utils/bev_preview_y_auto.jpg`: 순수 BEV 미리보기
- `utils/bev_checkerboard_analysis.jpg`: 코너와 품질 지표가 표시된 BEV
- `utils/bev_checkerboard_report.json`: 품질 수치와 설정 보고서
- `utils/bev_checkerboard_spacing.csv`: 체커보드 각 칸의 간격 측정값

선택 원본 좌표, homography, 실제 크기, 체커보드 품질 수치 같은 상세 정보는
주행 NPZ의 호환성을 위해 NPZ에 중복 저장하지 않고 TXT·JSON·CSV 결과에
보존한다. `src_points`에는 미리보기와 실제 변환에 적용된 유효 좌표가 들어간다.

# ROS 카메라 BEV 체커보드 왜곡 검사

ROS 카메라 화면에 `10x7` 내부 코너 체커보드를 놓고 현재 BEV NPZ의
비율·간격·직각 왜곡을 실시간으로 검사한다.

```bash
cd /home/hanjingyu/dolbatS
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 utils/ros_bev_checkerboard_distortion.py \
  --topic /camera/lane/raw/compressed \
  --transport compressed \
  --bev src/lane_vision_pkg/config/bev_params_y_auto.npz \
  --calibration src/lane_vision_pkg/config/camera_calibration.npz
```

화면에서 `S`를 누르면 분석 영상과 JSON 보고서를
`bev_distortion_result/`에 저장하고, `Q` 또는 `Esc`로 종료한다.

## 캡처 이미지에서 BEV ROI 생성

표준 흑백 체커보드의 모든 내부 코너를 검출하고, BEV 변환 후 각 셀의
가로/세로 비율이 1에 얼마나 가까운지로 X축 비율과 잔여 왜곡을 계산한다.

```bash
python3 utils/captured_frame_bev_parallel_roi.py \
  --image <체커보드가_보이는_캡처.jpg> \
  --calibration utils/camera_calibration.npz \
  --checkerboard 10 7 \
  --output utils/bev_params_checkerboard.npz
```

`--checkerboard` 값은 칸 개수가 아니라 내부 코너의 `가로 세로` 개수다.
저장 보고서에는 모든 셀의 정사각형 비율 오차, 칸 크기 CV, 가로·세로
간격 CV와 최대 직각 오차가 포함된다.
