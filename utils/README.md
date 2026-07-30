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
