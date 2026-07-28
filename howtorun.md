# dolbatS 차선 추종 자율주행 실행

현재 주행 흐름은 다음과 같다.

```text
/image_raw/compressed
  -> lane_detect
  -> /lane/detection/mask/compressed
  -> path_plan
  -> /lane/path/control
  -> pure_pursuit
  -> /cmd_vel
  -> serial_bridge
  -> Arduino
```

`drive_pipeline.launch.py`가 `lane_detect`, `path_plan`,
`path_visualizer`, `pure_pursuit` 노드를 한 번에 실행한다.

> 이 문서는 ROS 2 Humble과 현재 워크스페이스의 설치가 완료된 상태를
> 전제로 한다. 각 명령은 별도 터미널에서 실행한다.

## 0. 모든 터미널 공통 준비

```bash
cd ~/dolbatS
source /opt/ros/humble/setup.bash
source ~/dolbatS/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

카메라와 Arduino 장치 이름을 먼저 확인한다.

```bash
ls -l /dev/video*
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

## 1. Arduino 시리얼 브리지 실행

`serial_bridge`는 `/cmd_vel`을 구독하여 Arduino 주행 명령으로 변환한다.

```bash
ros2 run control_pkg serial_bridge \
  --serial-port /dev/ttyACM0 \
  --baudrate 115200
```

장치가 `/dev/ttyACM0` 등으로 잡혔다면 `--serial-port` 값을 변경한다.

발행되는 차량 상태:

```text
/vehicle/current_steering_angle
/vehicle/current_speed
/ultrasonic/left_distance
/ultrasonic/right_distance
```

## 2. 차선 카메라 실행

현재 `lane_detect`의 기본 입력은 `/image_raw/compressed`이다.

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0 \
  -p pixel_format:=mjpeg2rgb
```

카메라 토픽을 확인한다.

```bash
ros2 topic hz /image_raw/compressed
```

토픽이 없다면 카메라 장치 번호와 `usb_cam` 출력 토픽을 먼저 확인한다.

## 3. 주행 파라미터 확인

설정 파일:

```text
/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

실제 차량을 움직이기 전에는 다음 값을 유지한다.

```yaml
pure_pursuit:
  ros__parameters:
    enable_drive: false
```

주요 확인 항목:

```yaml
lane_detect:
  ros__parameters:
    model_path: "/home/tak/lane_yolo_project/weight/best1.pt"
    image_topic: "/image_raw/compressed"

path_plan:
  ros__parameters:
    pixels_per_meter: 600.0
    lane_width_m: 0.90
    bev_reference_forward_offset_m: 1.04
    max_fallback_sec: 0.45

pure_pursuit:
  ros__parameters:
    speed_mps: 0.18
    wheelbase_m: 0.545
    max_steer_deg: 18.0
    max_path_age_sec: 0.40
    max_fallback_sec: 0.45
    path_timeout_sec: 0.50
```

## 4. 차선 주행 파이프라인 실행

```bash
ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml
```

실행되는 노드:

```text
/lane_detect
/path_plan
/path_visualizer
/pure_pursuit
```

같은 기능을 실행하는 호환 명령 `yolo_lane_driver`를 launch와 동시에
실행하면 안 된다. 중복 노드와 중복 `/cmd_vel` 발행이 발생할 수 있다.

## 5. Dry-run 상태 확인

`enable_drive: false` 상태에서 다음 항목을 확인한다.

```bash
ros2 topic hz /lane/detection/mask/compressed
ros2 topic echo /lane/detection/status
ros2 topic echo /lane/path/status
ros2 topic echo /lane/path/control
ros2 topic echo /lane/control/status
ros2 topic echo /cmd_vel
```

정상 흐름:

```text
/lane/detection/status  -> YOLO 검출 결과 존재
/lane/path/status       -> path_valid: true
/lane/control/status    -> planned/commanded 조향각과 look-ahead 계산
/cmd_vel                -> linear.x와 angular.z가 0인 정지 명령
```

비주얼라이저의 `ACTUAL STEERING`은
`/vehicle/current_steering_angle` 피드백이며, 시리얼 브리지가 실행되지
않거나 1초 이상 피드백이 없으면 `N/A`로 표시된다.

디버그 이미지는 `rqt_image_view`에서 확인한다.

```bash
ros2 run rqt_image_view rqt_image_view
```

확인할 이미지 토픽:

```text
/lane/detection/segmentation/compressed
/lane/path/debug/compressed
```

## 6. 실제 주행 활성화

카메라 방향, 경로 중심, 조향 부호를 충분히 확인한 뒤 파이프라인을
종료한다. `drive_pipeline.yaml`에서 다음 값을 변경한다.

```yaml
pure_pursuit:
  ros__parameters:
    enable_drive: true
```

그다음 4번의 launch 명령으로 파이프라인을 다시 실행한다.

현재 구현은 실행 중 `ros2 param set`으로 `enable_drive`를 바꿔도 내부
제어 변수에 반영되지 않는다. YAML 수정 후 노드를 재시작해야 한다.

실제 주행 중에는 다음 값을 계속 확인한다.

```bash
ros2 topic echo /lane/control/status
ros2 topic echo /cmd_vel
```

## 7. 정지

주행 파이프라인 터미널에서 `Ctrl+C`를 누르면 `pure_pursuit`가 정지
`Twist`를 발행한다. 그다음 `serial_bridge`를 종료한다.

경로가 비거나 `path_timeout_sec` 동안 새 경로가 들어오지 않아도
`pure_pursuit`가 정지 명령을 발행한다.

rosbag 입력을 사용할 때는 기록 시각과 제어 시각을 일치시켜야 한다.

```bash
ros2 bag play <bag-directory> --clock
ros2 launch drive_pkg drive_pipeline.launch.py \
  params_file:=/home/tak/dolbatS/src/drive_pkg/config/drive_pipeline.yaml \
  use_sim_time:=true
```

## 주의사항

- 현재 구성은 차선 추종 주행이다.
- 장애물 검출 및 초음파 이벤트는 아직 `/cmd_vel` 제어와 연결되지 않았다.
- `detect_pkg`를 실행하는 것만으로 차량이 장애물 앞에서 자동 정지하지 않는다.
- `move_with_arrow.py` 등의 수동 제어 코드와 자율주행을 동시에 실행하지 않는다.
- `enable_drive: true`는 차량을 띄우고 바퀴 방향을 검증한 뒤 사용한다.
