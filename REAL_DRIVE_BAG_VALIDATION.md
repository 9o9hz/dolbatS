# Pure Pursuit 실차 주행 및 rosbag 검증 실행 가이드

이 문서는 `/home/tak/dolbatS`의 현재 실행 구조를 기준으로 한다.
패키지를 새로 빌드하지 않고, 터미널별 명령을 위에서부터 그대로 실행한다.

## 0. 반드시 먼저 확인할 사항

- 차량 바퀴가 사람이나 장애물을 향하지 않도록 한다.
- 최초 확인은 차량을 들어 올리거나 즉시 비상 정지가 가능한 상태에서 한다.
- `drive_pipeline.launch.py`는 `drive_main`과 `pure_pursuit`를 함께 실행한다.
  따라서 `pure_pursuit`를 별도 터미널에서 추가 실행하지 않는다.
- Serial Bridge의 명령 timeout도 `0.3초`로 강제 적용한다.
- 실제 출발은 터미널 4에서 스페이스바를 누를 때만 한다.

모든 터미널에서 공통으로 사용하는 환경 설정은 다음과 같다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
```

---

## 1. 터미널 1 — 차선 카메라 실행

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 \
  -p pixel_format:=mjpeg2rgb \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0 \
  -r image_raw:=/camera/lane/raw \
  -r image_raw/compressed:=/camera/lane/raw/compressed \
  -r camera_info:=/camera/lane/camera_info
```

카메라 장치가 `/dev/video2`가 아니면 실제 장치 번호로 변경한다.

별도 터미널에서 카메라 입력을 확인할 수 있다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
ros2 topic hz /camera/lane/raw/compressed
```

약 30 Hz가 확인된 다음 단계로 넘어간다.

---

## 2. 터미널 2 — 차선 검출 및 Pure Pursuit 실행

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

이 명령 하나가 다음 두 노드를 실행한다.

```text
/drive_main
/pure_pursuit
```

다음 명령을 추가로 실행하면 안 된다.

```text
ros2 run drive_pkg pure_pursuit
python3 /home/tak/dolbatS/src/drive_pkg/pure_pursuit.py
두 번째 drive_pipeline.launch.py
```

노드와 경로 발행을 확인한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 node list
ros2 topic hz /lane/path
ros2 topic echo --once /lane/control/status
```

---

## 3. 터미널 3 — Mission Manager 실행

현재 `mission_manager.yaml`에 저장된 파라미터를 그대로 적용한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 launch mission_manager_pkg mission_manager.launch.py \
  params_file:=/home/tak/dolbatS/src/mission_manager_pkg/config/mission_manager.yaml
```

실제로 적용된 값을 확인하려면 다른 터미널에서 다음을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
ros2 param get /mission_manager drive_enabled_default
```

---

## 4. 터미널 4 — 주행 활성화 키보드 스위치

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 run control_pkg keyboard_drive_toggle
```

시작 직후 다음 문구를 확인한다.

```text
DRIVE DISABLED
```

- 스페이스바 1회: 주행 활성화
- 스페이스바 다시 1회: 즉시 주행 비활성화
- `Ctrl+C`: 토글 노드 종료 및 `drive/enabled=false` 발행

아직 스페이스바를 누르지 않는다.

---

## 5. 터미널 5 — 실차 검증 bag 기록 시작

다음 명령은 실행 시각을 포함한 새 폴더를 만들고, YAML·Git 상태·모델 및
BEV 파일 체크섬을 저장한 후 rosbag 기록을 시작한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=/home/tak/Desktop/bags/pp_curve_baseline_${RUN_ID}

mkdir -p "$RUN_ROOT/meta"

cp /home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml \
  "$RUN_ROOT/meta/drive_pipeline.yaml"

cp /home/tak/dolbatS/src/mission_manager_pkg/config/mission_manager.yaml \
  "$RUN_ROOT/meta/mission_manager.yaml"

cp /home/tak/dolbatS/src/control_pkg/config/serial_bridge.yaml \
  "$RUN_ROOT/meta/serial_bridge.yaml"

git -C /home/tak/dolbatS rev-parse HEAD \
  > "$RUN_ROOT/meta/git_commit.txt"

git -C /home/tak/dolbatS status --short \
  > "$RUN_ROOT/meta/git_status.txt"

git -C /home/tak/dolbatS diff \
  > "$RUN_ROOT/meta/uncommitted_changes.patch"

sha256sum \
  /home/tak/dolbatS/src/drive_pkg/resource/best.pt \
  /home/tak/dolbatS/src/drive_pkg/resource/bev_params_0803.npz \
  /home/tak/dolbatS/camera_calibration.npz \
  > "$RUN_ROOT/meta/resource_checksums.txt"

printf '%s\n' "$RUN_ROOT" | tee /tmp/dolbats_last_drive_bag_path.txt

ros2 bag record \
  -o "$RUN_ROOT/bag" \
  /camera/lane/raw/compressed \
  /camera/lane/camera_info \
  /lane/path \
  /lane/path/status \
  /lane/control/status \
  /lane/detection/status \
  /lane/detection/instances \
  /which/lane \
  /control/candidate/lane/steer_angle \
  /control/candidate/lane/valid \
  /auto_steer_angle \
  /auto_throttle \
  /vehicle/angle \
  /vehicle/drive_pwm \
  /drive/enabled \
  /mission_state \
  /mission_manager/status
```

다음과 같은 메시지가 나타나면 기록 준비가 된 것이다.

```text
Recording...
```

이 터미널은 기록이 끝날 때까지 그대로 둔다.

### 문제 구간의 시각화 토픽도 같이 기록하려면

짧은 디버그 주행에서는 위 `ros2 bag record` 토픽 목록 끝에 다음을 추가한다.

```text
/lane/detection/bev/compressed
/lane/detection/segmentation/compressed
/lane/path/debug/compressed
```

장시간 주행에서는 CPU·디스크 부하와 bag 용량 증가 때문에 기본 기록 목록을
권장한다. raw 카메라가 저장되므로 시각화 결과는 나중에 다시 생성할 수 있다.

---

## 6. 터미널 6 — Arduino Serial Bridge 실행

Serial Bridge는 모든 인식·제어·기록 노드가 준비된 뒤 마지막에 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 launch control_pkg serial_bridge.launch.py \
  params_file:=/home/tak/dolbatS/src/control_pkg/config/serial_bridge.yaml \
  serial_port:=/dev/ttyACM0 \
  command_timeout_sec:=0.3
```

Arduino 포트가 `/dev/ttyUSB0`이면 `serial_port`를 변경한다.

다른 터미널에서 실제 조향각이 들어오는지 확인한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 topic hz /vehicle/angle
ros2 topic echo --once /vehicle/angle
```

`/vehicle/angle`이 발행되지 않으면 실제 조향 응답을 비교할 수 없으므로 주행을
시작하지 말고 Arduino 연결과 telemetry를 먼저 확인한다.

---

## 7. 런타임 파라미터 저장

bag이 기록되는 동안 새 터미널에서 실행한다. 터미널 5가 저장한 최근 경로를
불러오므로 경로를 다시 입력할 필요가 없다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

RUN_ROOT=$(cat /tmp/dolbats_last_drive_bag_path.txt)

ros2 param dump /drive_main \
  > "$RUN_ROOT/meta/drive_main_runtime.yaml"

ros2 param dump /pure_pursuit \
  > "$RUN_ROOT/meta/pure_pursuit_runtime.yaml"

ros2 param dump /mission_manager \
  > "$RUN_ROOT/meta/mission_manager_runtime.yaml"

ros2 param dump /serial_bridge \
  > "$RUN_ROOT/meta/serial_bridge_runtime.yaml"
```

---

## 8. 실제 주행 시작

아래 항목을 모두 확인한다.

```text
[ ] 카메라 약 30 Hz
[ ] /lane/path 발행 확인
[ ] /lane/control/status 발행 확인
[ ] /auto_throttle=0 확인
[ ] rosbag Recording 확인
[ ] /vehicle/angle telemetry 확인
[ ] 비상 정지 담당자 준비
```

확인이 끝나면 터미널 4에서 스페이스바를 한 번 눌러 주행을 시작한다.

권장 기록 순서는 다음과 같다.

```text
1. 정지 상태 3~5초
2. 직선 진입
3. 완만한 곡선
4. 급곡선
5. S자 또는 곡률 부호가 바뀌는 구간
6. 정지
```

같은 코스를 같은 방향과 비슷한 속도로 최소 3회 반복한다. 한 번의 장시간
bag보다 다음과 같이 실행별로 분리하는 편이 분석하기 쉽다.

```text
pp_curve_baseline_run01
pp_curve_baseline_run02
pp_curve_baseline_run03
```

---

## 9. 주행 및 기록 종료 순서

종료 순서를 지킨다.

1. 터미널 4에서 스페이스바를 눌러 `DRIVE DISABLED`로 만든다.
2. `/auto_throttle`이 0인지 확인한다.
3. 정지 상태를 약 3초 더 기록한다.
4. 터미널 5의 bag 기록에서 `Ctrl+C`를 한 번 누른다.
5. bag 종료가 완료된 후 Serial Bridge와 나머지 노드를 종료한다.

최종 throttle 확인:

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash
ros2 topic echo --once /auto_throttle
```

bag을 강제 종료하지 않는다. 정상적으로 `Ctrl+C`를 사용해야
`metadata.yaml`이 완성된다.

---

## 10. 저장된 bag 확인

새 터미널에서 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

RUN_ROOT=$(cat /tmp/dolbats_last_drive_bag_path.txt)

ros2 bag info "$RUN_ROOT/bag"
find "$RUN_ROOT" -maxdepth 2 -type f | sort
```

다음 토픽의 메시지 수가 반드시 0보다 커야 한다.

```text
/camera/lane/raw/compressed
/lane/path
/lane/control/status
/control/candidate/lane/steer_angle
/auto_steer_angle
/auto_throttle
/vehicle/angle
```

`/vehicle/angle`이 0개이면 해당 bag으로 Arduino 실제 조향 추종 성능을 평가할
수 없다.

---

## 11. 저장한 bag을 안전하게 재생

Serial Bridge와 실제 Mission Manager를 종료한 상태에서 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

RUN_ROOT=$(cat /tmp/dolbats_last_drive_bag_path.txt)

ros2 launch drive_pkg drive_rosbag.launch.py \
  bag_path:="$RUN_ROOT/bag" \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml \
  rate:=1.0 \
  show_visualizer:=true
```

`drive_rosbag.launch.py`는 카메라 토픽만 재생하고 후보 조향 출력을
`/debug/control/candidate/lane/*`로 보내므로 실제 Mission Manager에 연결되지
않는다. 반복 재생되므로 종료할 때 `Ctrl+C`를 누른다.

직접 재생해야 한다면 저장된 전체 토픽을 재생하지 말고 카메라만 선택한다.

```bash
ros2 bag play "$RUN_ROOT/bag" \
  --topics /camera/lane/raw/compressed
```

전체 bag을 그대로 재생하면서 `drive_pipeline`을 켜면 저장된 `/lane/path`와
현재 노드가 생성하는 `/lane/path`가 동시에 발행되어 검증 결과가 오염된다.

---

## 12. 곡률 추종 개선 적용 순서

첫 실차 주행에서는 현재 Pure Pursuit 조향 출력을 변경하지 않고 baseline을
기록한다.

먼저 `/lane/control/status`에 다음 진단값만 추가하는 것을 권장한다.

```text
target_path_curvature_1pm
pure_pursuit_curvature_1pm
curvature_tracking_error_1pm
target_curvature_valid
target_curvature_sample_count
path_arc_length_m
path_first_distance_m
```

각 오차는 다음과 같이 분리한다.

```text
레퍼런스 목표점 곡률: k_target
Pure Pursuit 명령 곡률: k_pp
Arduino 실제 곡률: k_actual = tan(vehicle_angle) / wheelbase

제어 알고리즘 오차 = k_target - k_pp
조향 실행단 오차 = k_pp - k_actual
전체 오차 = k_target - k_actual
```

판단 순서는 다음과 같다.

1. `/lane/path`와 `k_target`이 불안정하면 경로 생성부터 수정한다.
2. `k_pp`는 충분하지만 `/vehicle/angle`이 따라가지 못하면 Arduino 조향
   보정부터 수정한다.
3. 경로와 Arduino가 정상인데 `k_pp`만 곡선에서 지속적으로 부족할 때만
   제한적 곡률 혼합을 적용한다.
4. 곡률 혼합은 gain `0.1`부터 시작하고 동일 코스에서 다시 3회 기록한다.
5. baseline과 수정 버전을 같은 raw 입력의 offline replay 및 별도 실차
   주행으로 모두 비교한다.

다음처럼 곡률 조향각을 Pure Pursuit 조향각에 그대로 더하면 안 된다.

```text
delta = delta_pp + atan(wheelbase * k_target)
```

정상 원호에서는 Pure Pursuit가 이미 곡률을 포함하므로 이중 조향이 된다.
제어 변경은 baseline bag 분석 후 제한된 곡률 차이만 혼합하는 방식으로
진행해야 한다.

---

## 13. 핵심 비상 정지 방법

가장 먼저 사용할 방법:

```text
터미널 4에서 스페이스바
```

토글 터미널을 사용할 수 없으면 새 터미널에서 다음을 실행한다.

```bash
source /opt/ros/humble/setup.bash
source /home/tak/dolbatS/install/setup.bash

ros2 topic pub --once \
  /drive/enabled \
  std_msgs/msg/Bool \
  "{data: false}"
```

ROS 통신 자체가 끊겼다면 차량 전원을 차단한다. `command_timeout_sec:=0.3`은
ROS 명령이 끊겼을 때 구동 정지를 보내지만, 하드웨어 비상 정지를 대체하지
않는다.
