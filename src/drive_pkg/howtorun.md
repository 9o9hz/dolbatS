# drive_pkg 모듈형 파이프라인

`drive_pkg`의 기본 실행 경로는 `drive_main` 하나의 통합 노드다. 카메라
이미지 구독 콜백 안에서 검출 → 경로 생성 → Pure Pursuit 제어가 토픽 왕복
없이 순서대로 호출된다(예전처럼 3개 노드로 나눠 토픽으로 연결하고 싶을 때를
대비해 `lane_detect`/`path_plan`/`pure_pursuit` 개별 실행도 그대로 남아
있다). `path_visualizer`는 여전히 별도 노드로, `drive_main`이 발행하는
디버그 토픽을 구독해 한 창에 시각화한다.

```text
/image_raw/compressed
  -> drive_main (undistort -> BEV/YOLO 검출 -> 경로 생성 -> Pure Pursuit)
  -> /cmd_vel
```

전체 파라미터는 `config/drive_pipeline.yaml`에서 `drive_main:` 블록 하나로
조정한다(예전 `lane_detect:`/`path_plan:`/`pure_pursuit:` 세 블록이 이
블록으로 합쳐졌다. `path_visualizer:` 블록은 그대로다).

## 전체 실행

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drive_pkg drive_pipeline.launch.py
```

소스 트리의 YAML을 바로 지정하려면:

```bash
ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

가중치/BEV 파라미터/캘리브레이션 파일을 커맨드라인에서 바로 바꾸고 싶으면
`drive_main`을 직접 실행하며 `--weights`/`--bev-params`/`--calib-file`을
쓴다(이 값들은 YAML의 `model_path`/`bev_params`/`calib_file`보다 낮은
우선순위의 기본값으로 들어가므로, `--ros-args --params-file`을 같이 주면
YAML 값이 최종 적용된다):

```bash
ros2 run drive_pkg drive_main --weights /path/to/best.pt \
  --bev-params /path/to/bev_params.npz \
  --ros-args --params-file config/drive_pipeline.yaml
```

기존 명령과의 호환을 위해 `drive_pipeline`/`yolo_lane_driver` 실행 커맨드도
여전히 남아 있으며, 내부적으로 `drive_main`과 동일하게 동작한다.

```bash
ros2 run drive_pkg yolo_lane_driver --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

## 노드별 실행 (구조 디버깅용)

`lane_detect`/`path_plan`/`pure_pursuit`는 검출·경로 생성·제어 로직을
각각 독립된 토픽 연결 노드로 실행하고 싶을 때(예: 특정 단계만 따로 확인,
장애 격리, 재시작 필요 시) 쓴다. `drive_main`과 정확히 같은 코드를
호출하므로(각 노드는 `drive_main`이 쓰는 것과 같은 `LaneDetectorCore` /
`SegmentationLaneProcessor` / `PurePursuitController`를 감싸는 얇은
wrapper) 결과는 동일하다. 세 터미널에서 공통 YAML을 넘겨 개별 실행한다.

```bash
ros2 run drive_pkg lane_detect --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

```bash
ros2 run drive_pkg path_plan --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

```bash
ros2 run drive_pkg pure_pursuit --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

이 개별 노드들은 여전히 `config/drive_pipeline.yaml`의 옛 블록 이름
(`lane_detect:`/`path_plan:`/`pure_pursuit:`)을 참조하므로, 개별 실행용
YAML을 따로 두거나 필요한 블록만 별도 파일로 분리해서 쓴다.

## 카메라 undistort (옵션)

`drive_main`은 `calib_file`(기본값: 워크스페이스 루트의
`camera_calibration.npz`, 소스 트리에서 실행할 때만 자동으로 잡힌다)을
로드해 BEV 변환 전에 렌즈 왜곡을 보정한다. 파일이 없거나 형식이 맞지 않으면
경고만 남기고 undistort 없이 그대로 진행하므로 노드 기동에는 영향이 없다.
`use_undistort: false`로 아예 끄거나, `calib_file`을 다른 경로로 지정할
수 있다.

## 미션 훅 (자리만 마련, 미구현)

`drive_main.LaneDriveNode`에 `preprocess_frame()`(undistort 직후, BEV/검출
직전 호출)과 `postprocess_result()`(Pure Pursuit 계산 직후 호출) 두 훅이
비어 있는 상태로 존재한다. 이후 장애물회피/신호등/수직주차 같은 미션
로직을 여기에 얹을 예정이며, 이번 단계에서는 구조만 마련하고 실제 판정은
구현하지 않았다.

## 주요 토픽

| 단계 | 토픽 | 타입 | 내용 |
| --- | --- | --- | --- |
| 입력 | `/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| 검출 | `/lane/detection/mask/compressed` | `sensor_msgs/CompressedImage` | BEV 이진 차선 마스크(PNG) |
| 검출 | `/lane/detection/segmentation/compressed` | `sensor_msgs/CompressedImage` | YOLO 시각화 |
| 검출 | `/lane/detection/status` | `std_msgs/String` | 검출 개수·추론 시간 JSON |
| 계획 | `/lane/path` | `nav_msgs/Path` | `base_link` 기준 metric 경로 |
| 계획 | `/lane/path/debug/compressed` | `sensor_msgs/CompressedImage` | 생성 경로 시각화 |
| 계획 | `/lane/path/status` | `std_msgs/String` | 경로 유효성·fallback JSON |
| 계획 | `/which/lane` | `std_msgs/String` | `lane_1`, `lane_2`, `unknown` |
| 제어 | `/cmd_vel` | `geometry_msgs/Twist` | 주행 속도·각속도 명령 |
| 제어 | `/lane/control/status` | `std_msgs/String` | 조향·LD·목표 속도 JSON |
| 차량 피드백 | `/vehicle/current_steering_angle` | `std_msgs/Float32` | Arduino가 보고한 실제 조향각 |

`pure_pursuit.enable_drive`(`drive_main`에서는 `enable_drive`) 기본값은
`false`다. 이때 계산 결과는 `/lane/control/status`에서 확인할 수 있지만
`/cmd_vel`에는 정지 명령만 발행한다. 경로와 조향 방향을 검증한 뒤에만
YAML 값을 `true`로 바꾼다.

각 단계 확인 예:

```bash
ros2 topic hz /lane/detection/mask/compressed
ros2 topic echo /lane/path/status
ros2 topic echo /lane/control/status
```

이미지는 `rqt_image_view`에서 검출/경로 디버그 토픽을 선택해 확인한다.
