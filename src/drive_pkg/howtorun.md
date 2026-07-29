# drive_pkg 2노드 파이프라인

`drive_pkg`는 두 개의 필수 노드로 나뉜다: `drive_main`(인식 + 경로 생성,
`/lane/path` 발행)과 `pure_pursuit`(그 경로를 구독해 제어, 조향각/정규화
throttle을 `/auto_steer_angle`/`/auto_throttle`로 발행).
시각화는 별도 노드 없이 `pure_pursuit` 프로세스 안에 통합돼 있다
(main5.py 스타일 로컬 cv2 창) — `drive_main`이 발행하는 디버그 토픽 4개와
자기 자신의 제어 상태(토픽 왕복 없이 직접 전달)를 모아 한 창에 표시하며,
`local_display` 파라미터로 켜고 끈다.

```text
/image_raw/compressed
  -> drive_main (undistort -> BEV/YOLO 검출 -> 경로 생성)
  -> /lane/path
  -> pure_pursuit (Pure Pursuit 제어)
  -> /auto_steer_angle, /auto_throttle
```

전체 파라미터는 `config/drive_pipeline.yaml`에서 `drive_main:`/`pure_pursuit:`
두 블록으로 나뉘어 있다. 검출·경로 생성 관련 파라미터는 `drive_main:`에,
조향/속도 제어와 통합 시각화(옛 `path_visualizer:` 블록의 모든 키) 관련
파라미터는 `pure_pursuit:`에 있다.

경로 생성 내부 알고리즘(`lane_processing.py`)은 자매 프로젝트
`yolotl_ros2`의 `main5.py`(검출 bbox 단위 최대 성분 추출, 프레임 간 좌/우
트래킹, b-spline 공간 평활화 + 기존 EMA 시간축 평활화, Pure Pursuit
lookahead 탐색)를 기반으로 한다. 실선/점선 판별과 1차선/2차선(`which_lane`)
판정, 실선 우선 히스테리시스는 이 대회 규정 특화 로직이라 그대로 유지된다.
`scipy`(b-spline)가 런타임 의존성으로 추가돼 있다.

## 전체 실행

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drive_pkg drive_pipeline.launch.py
```

`drive_pipeline.launch.py`는 `drive_main`, `pure_pursuit` 두 노드를 함께
띄운다. `pure_pursuit`의 `local_display`(기본값 true)가 켜져 있으면 통합
시각화 창도 이때 함께 뜬다.

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

`pure_pursuit`는 별도 프로세스로 직접 실행해야 실제로 `/auto_steer_angle`/
`/auto_throttle`이 발행된다 (`drive_main`은 더 이상 조향/속도를 계산하지
않는다):

```bash
ros2 run drive_pkg pure_pursuit --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

`pure_pursuit`는 `local_display: true`(기본값)일 때 segmentation | BEV
제어 | 검출선+경로 세 패널을 합친 cv2 창을 띄운다. 창에서 `q` 또는 ESC를
누르면 노드가 종료된다. 디스플레이가 없는 환경(헤드리스)에서는
`local_display: false`로 꺼야 한다.

## 카메라 undistort (옵션)

`drive_main`은 `calib_file`(기본값: 워크스페이스 루트의
`camera_calibration.npz`, 소스 트리에서 실행할 때만 자동으로 잡힌다)을
로드해 BEV 변환 전에 렌즈 왜곡을 보정한다. 파일이 없거나 형식이 맞지 않으면
경고만 남기고 undistort 없이 그대로 진행하므로 노드 기동에는 영향이 없다.
`use_undistort: false`로 아예 끄거나, `calib_file`을 다른 경로로 지정할
수 있다.

## 미션 훅 (자리만 마련, 미구현)

두 노드에 각각 훅이 있다:

- `drive_main.LaneDriveNode`: `preprocess_frame()`(undistort 직후, BEV/검출
  직전 호출), `postprocess_path()`(경로 생성 직후, `/lane/path` 발행 직전
  호출 — 정지선/장애물 회피처럼 경로 자체를 바꾸는 미션 로직의 자리).
- `pure_pursuit.PurePursuitNode`: `postprocess_command()`(Pure Pursuit
  계산 직후, `/auto_steer_angle`/`/auto_throttle` 발행 직전 호출 —
  신호등/수직주차처럼 조향·속도 결과를 바꾸는 미션 로직의 자리).

둘 다 현재는 pass-through이며, 이번 단계에서는 구조만 마련하고 실제 판정은
구현하지 않았다.

## 주요 토픽

| 단계 | 토픽 | 타입 | 내용 |
| --- | --- | --- | --- |
| 입력 | `/image_raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| 검출 | `/lane/detection/mask/compressed` | `sensor_msgs/CompressedImage` | BEV 이진 차선 마스크(PNG) |
| 검출 | `/lane/detection/segmentation/compressed` | `sensor_msgs/CompressedImage` | YOLO 시각화 |
| 검출 | `/lane/detection/status` | `std_msgs/String` | 검출 개수·추론 시간 JSON |
| 검출 | `/lane/detection/instances` | `std_msgs/String` | 검출별 bbox·confidence JSON (`drive_main` 내부에서 bbox 단위 추출에 사용) |
| 계획 | `/lane/path` | `nav_msgs/Path` | `base_link` 기준 metric 경로 (`drive_main` -> `pure_pursuit`) |
| 계획 | `/lane/path/debug/compressed` | `sensor_msgs/CompressedImage` | 생성 경로 시각화 |
| 계획 | `/lane/path/status` | `std_msgs/String` | 경로 유효성·fallback JSON |
| 계획 | `/which/lane` | `std_msgs/String` | `lane_1`, `lane_2`, `unknown` |
| 제어 | `/auto_steer_angle` | `std_msgs/Float32` | 조향각(deg) (`pure_pursuit` 발행) |
| 제어 | `/auto_throttle` | `std_msgs/Float32` | 정규화 throttle(`-1.0~1.0`, 음수=후진) (`pure_pursuit` 발행) |
| 제어 | `/lane/control/status` | `std_msgs/String` | 조향·LD·목표 속도 JSON (`pure_pursuit` 발행) |
| 차량 피드백 | `/vehicle/current_steering_angle` | `std_msgs/Float32` | Arduino가 보고한 실제 조향각 |

항상 실제 주행 명령을 `/auto_steer_angle`/`/auto_throttle`에 발행한다
(dry-run 모드 없음). `pure_pursuit`가 빈 경로(`poses` 없음)를 받으면
자동으로 `throttle=0.0`을 낸다 — `drive_main`이 검출/경로 생성에 실패하면
빈 `/lane/path`를 발행해서 이 경로로 정지시킨다. 별도의 경로-끊김
워치독(`path_timeout_sec`)은 없다.

각 단계 확인 예:

```bash
ros2 topic hz /lane/detection/mask/compressed
ros2 topic echo /lane/path/status
ros2 topic echo /auto_steer_angle
ros2 topic echo /auto_throttle
```

이미지는 `rqt_image_view`에서 검출/경로 디버그 토픽을 선택해 확인한다.
