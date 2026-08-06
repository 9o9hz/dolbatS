# drive_pkg 모듈형 파이프라인 (구현 B)

이 저장소에는 자율주행 구현이 두 개 있으며 둘 다 유지한다.

- 구현 A: `lane_vision_pkg` 단일 노드형
- 구현 B: 이 문서의 `drive_pkg` 3노드 모듈형

두 구현은 기본적으로 같은 `/lane/path`와 `/cmd_vel`을 사용하므로 동시에
실행하지 않는다. 한 구현을 종료한 뒤 다른 구현을 실행한다.

`drive_pkg`는 두 개의 필수 노드로 나뉜다: `drive_main`(인식 + 경로 생성,
`/lane/path` 발행)과 `pure_pursuit`(그 경로를 구독해 차선 후보 조향각과
유효 여부를 `/control/candidate/lane/*`로 발행).
시각화는 별도 노드 없이 `pure_pursuit` 프로세스 안에 통합돼 있다
(main5.py 스타일 로컬 cv2 창) — `drive_main`이 발행하는 디버그 토픽 4개와
자기 자신의 제어 상태(토픽 왕복 없이 직접 전달)를 모아 한 창에 표시하며,
`local_display` 파라미터로 켜고 끈다.

```text
/camera/lane/raw/compressed
  -> lane_detect
  -> /lane/detection/mask/compressed
  -> path_plan
  -> /lane/path
  -> pure_pursuit (Pure Pursuit 제어)
  -> /control/candidate/lane/{steer_angle,valid}
```

전체 파라미터는 `config/drive_pipeline.yaml`에서 `drive_main:`/`pure_pursuit:`
두 블록으로 나뉘어 있다. 검출·경로 생성 관련 파라미터는 `drive_main:`에,
조향 제어와 통합 시각화(옛 `path_visualizer:` 블록의 모든 키) 관련
파라미터는 `pure_pursuit:`에 있다.

경로 생성 내부 알고리즘(`lane_processing.py`)은 자매 프로젝트
`yolotl_ros2`의 `main5.py`(검출 bbox 단위 최대 성분 추출, 프레임 간 좌/우
트래킹, b-spline 공간 평활화 + 기존 EMA 시간축 평활화, Pure Pursuit
lookahead 탐색)를 기반으로 한다. 실선/점선 판별과 1차선/2차선(`which_lane`)
판정, 실선 우선 히스테리시스는 이 대회 규정 특화 로직이라 그대로 유지된다.
`scipy`(b-spline)가 런타임 의존성으로 추가돼 있다.

## 전체 실행

카메라와 세 노드를 함께 안전 모드로 실행:

```bash
ros2 launch drive_pkg drive_pipeline_full.launch.py \
  enable_drive:=false \
  launch_serial_bridge:=false
```

실차 저속 실행:

```bash
ros2 launch drive_pkg drive_pipeline_full.launch.py \
  enable_drive:=true \
  launch_serial_bridge:=true \
  serial_port:=/dev/ttyUSB0
```

실행 직후에는 정지 상태다. 같은 터미널에서 스페이스바를 누르면 출발하고
다시 누르면 즉시 정지한다. `Ctrl+C`는 전체 노드를 종료한다.

카메라를 이미 별도로 실행 중이면 세 노드만 실행:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch drive_pkg drive_pipeline.launch.py
```

`drive_pipeline.launch.py`는 `drive_main`, `pure_pursuit` 두 노드를 함께
띄운다. 최종 제어는 별도 `mission_manager` launch가 발행한다.
`pure_pursuit`의 `local_display`(기본값 true)가 켜져 있으면 통합 시각화
창도 이때 함께 뜬다.

소스 트리의 YAML을 바로 지정하려면:

```bash
ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

## 개선된 Pure Pursuit 적용 및 확인

현재 개선판은 기존 `pure_pursuit` 노드 안에 구현되어 있다. 새 노드나 launch,
토픽을 추가하지 않았으므로 실행 명령은 그대로이며, 반드시 실행 중인 기존
launch를 `Ctrl+C`로 종료한 뒤 다시 시작해야 새 파라미터와 Python 코드가
로드된다. 현재 워크스페이스는 소스가 install 영역에 연결된 개발 설치이므로
이 변경을 적용하기 위한 별도 패키지 빌드는 하지 않는다.

개선판의 핵심 기본값은
`src/drive_pkg/config/drive_pipeline.yaml`의 `pure_pursuit.ros__parameters`
블록에 있다.

```yaml
minimum_path_preview_m: 0.30
lookahead_search_mode: "continuous_arc_length"

curvature_tracking_enabled: true
curvature_tracking_gain: 0.20
curvature_tracking_sample_gap_m: 0.15
curvature_tracking_preview_m: 0.45
curvature_tracking_min_samples: 3
curvature_tracking_max_mad_1pm: 0.25
curvature_tracking_max_correction_1pm: 0.20
curvature_tracking_sign_guard_1pm: 0.05
curvature_tracking_min_deficit_1pm: 0.01
```

실차/카메라 입력에 적용:

```bash
cd /home/tak/dolbatS
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

0803 bag에 안전하게 적용(최종 조향 토픽 대신 `/debug/control/...` 사용):

```bash
cd /home/tak/dolbatS
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
ros2 launch drive_pkg drive_rosbag.launch.py \
  bag_path:=/home/tak/Desktop/bags/rosbag2_2026_08_03-14_53_47 \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml \
  show_visualizer:=true
```

적용된 파라미터 확인:

```bash
ros2 param get /pure_pursuit minimum_path_preview_m
ros2 param get /pure_pursuit curvature_tracking_preview_m
ros2 param get /pure_pursuit curvature_tracking_max_mad_1pm
ros2 param get /pure_pursuit curvature_tracking_min_deficit_1pm
```

각각 `0.3`, `0.45`, `0.25`, `0.01`이 나와야 한다. 제어 상태 확인:

```bash
ros2 topic echo /lane/control/status
```

상태 JSON의 주요 항목은 다음과 같다.

- `lookahead_m`: 속도·곡률·필터로 결정한 기존 동적 Ld
- `target_search_lookahead_m`: 최소 가시 경로 preview를 반영한 실제 목표점 탐색 거리
- `target_path_preview_m`: 경로 첫 점부터 목표점까지 확보된 실제 호길이
- `target_path_curvature_samples`: 곡률 중앙값에 사용한 표본 수
- `target_path_curvature_mad_1pm`: 표본 곡률의 MAD(작을수록 일관됨)
- `curvature_tracking_deficit_1pm`: PP에 부족한 REF 곡률의 절댓값 차이
- `curvature_tracking_reason`: `applied`, `unstable_preview_curvature`,
  `fallback_path`, `opposite_direction_guard`, `pp_already_sufficient`,
  `curvature_deficit_below_guard` 등 보정 적용/차단 이유

시각화 중간 패널에서도 `LOOK-AHEAD/SEARCH/TARGET`, `VISIBLE PATH PREVIEW`,
`CURVATURE QUALITY: N=... MAD=...`를 확인할 수 있다.

튜닝은 다음 순서를 권장한다.

1. 기본값으로 bag과 저속 실차에서 `target_path_preview_m`, MAD, raw 조향을 저장한다.
2. 커브 진입 목표점이 여전히 지나치게 가깝다면
   `minimum_path_preview_m`만 `0.30 -> 0.35`로 소폭 올린다.
3. 곡률 품질이 안정적인데도 보정량이 부족한 경우에만
   `curvature_tracking_gain`을 `0.20 -> 0.25`로 올린다.
4. `steering_gain`을 먼저 크게 올리거나 MAD 한계를 무작정 넓히지 않는다.

코드를 지우지 않고 기존 Pure Pursuit에 가깝게 롤백하려면 다음 두 값만
변경하고 노드를 재시작한다.

```yaml
minimum_path_preview_m: 0.0
curvature_tracking_enabled: false
```

기존 명령과의 호환을 위해 아래 명령도 세 노드를 한 프로세스에서
실행한다. 노드별 장애 격리와 재시작이 필요하면 launch 실행을 권장한다.

```bash
ros2 run drive_pkg yolo_lane_driver --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

## 노드별 실행

세 터미널에서 공통 YAML을 넘겨 개별 실행할 수 있다.

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

## 확장 경계

두 노드에 각각 훅이 있다:

- `drive_main.LaneDriveNode`: `preprocess_frame()`(undistort 직후, BEV/검출
  직전 호출), `postprocess_path()`(경로 생성 직후, `/lane/path` 발행 직전
  호출 — 정지선/장애물 회피처럼 경로 자체를 바꾸는 미션 로직의 자리).
`pure_pursuit`에는 신호등·장애물·주차 판단을 넣지 않는다.
mission_manager가 candidate와 detector 결과를 구독해 최종 조향과 throttle을
결정한다. 차선 후보의 마지막 유효값은 유지하지만, 활성화된 장애물 후보가
invalid이면 과거 풀조향값을 사용하지 않고 정지한다.

## 주요 토픽

| 단계 | 토픽 | 타입 | 내용 |
| --- | --- | --- | --- |
| 입력 | `/camera/lane/raw/compressed` | `sensor_msgs/CompressedImage` | 카메라 영상 |
| 검출 | `/lane/detection/mask/compressed` | `sensor_msgs/CompressedImage` | BEV 이진 차선 마스크(PNG) |
| 검출 | `/lane/detection/segmentation/compressed` | `sensor_msgs/CompressedImage` | YOLO 시각화 |
| 검출 | `/lane/detection/status` | `std_msgs/String` | 검출 개수·추론 시간 JSON |
| 검출 | `/lane/detection/instances` | `std_msgs/String` | 검출별 bbox·confidence JSON (`drive_main` 내부에서 bbox 단위 추출에 사용) |
| 계획 | `/lane/path` | `nav_msgs/Path` | `base_link` 기준 metric 경로 (`drive_main` -> `pure_pursuit`) |
| 계획 | `/lane/path/debug/compressed` | `sensor_msgs/CompressedImage` | 생성 경로 시각화 |
| 계획 | `/lane/path/status` | `std_msgs/String` | 경로 유효성·fallback JSON |
| 계획 | `/which/lane` | `std_msgs/String` | `lane_1`, `lane_2`, `unknown` |
| 후보 | `/control/candidate/lane/steer_angle` | `std_msgs/Float32` | 차선 후보 조향각(deg) |
| 후보 | `/control/candidate/lane/valid` | `std_msgs/Bool` | 현재 path 기반 후보 유효 여부 |
| 입력 | `/auto_throttle` | `std_msgs/Float32` | 최종 throttle 피드백(동적 LD 계산) |
| 제어 | `/lane/control/status` | `std_msgs/String` | 후보 조향·LD JSON (`pure_pursuit` 발행) |
| 차량 피드백 | `/vehicle/angle` | `std_msgs/Float32` | Arduino가 보고한 실제 조향각 |

`pure_pursuit`는 실제 주행 명령이나 throttle 후보를 발행하지 않는다.
빈 경로를 받으면 `valid=false`를 발행하고 마지막 조향값은 상태 표시용으로
유지한다.

각 단계 확인 예:

```bash
ros2 topic hz /lane/detection/mask/compressed
ros2 topic echo /lane/path/status
ros2 topic echo /control/candidate/lane/steer_angle
ros2 topic echo /control/candidate/lane/valid
```

이미지는 `rqt_image_view`에서 검출/경로 디버그 토픽을 선택해 확인한다.
