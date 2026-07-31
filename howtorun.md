


ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 \
  -p pixel_format:=mjpeg2rgb \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0 \
  -r image_raw:=/camera/lane/raw \
  -r image_raw/compressed:=/camera/lane/raw/compressed \
  -r camera_info:=/camera/lane/camera_info

ros2 launch drive_pkg drive_pipeline.launch.py

ros2 launch mission_manager_pkg mission_manager.launch.py


ros2 launch control_pkg serial_bridge.launch.py

  ros2 launch detect_pkg obstacle_detection.launch.py
ros2 run camera_pkg traffic_light_camera_publisher --camera-index 4

ros2 run detect_pkg traffic_light_detection


```
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video2 \
  -p pixel_format:=mjpeg2rgb \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0




## Mission manager

```bash
ros2 launch mission_manager_pkg mission_manager.launch.py
```

차선·장애물 후보와 detector 결과를 받아 `lane`, `traffic_light`,
`obstacle` 중 하나를 선택한다. 차선 candidate의 마지막 유효값은
timeout 없이 유지한다. 노드 시작 후 유효한 차선 candidate를 한 번도
받지 않았다면 최종 조향과 throttle을 0으로 발행한다.

장애물 회피 노드는 YOLO의 `/detect/obstacle/detected` 감지와 좌우 뒤
초음파의 감지→해제를 각각 상태로 저장한다. 두 상태가 모두 준비되면
뒤 초음파가 감지했던 방향으로 즉시 회피를 시작한다. 회피가 시작되면
`/detect/obstacle/avoidance_active`를 회전 완료까지 유지한다.
회피 중 obstacle candidate가 invalid가 되면 과거 풀조향값을 유지하지 않고
최종 조향과 throttle을 모두 0으로 발행한다.

```bash
ros2 topic echo /mission_state
ros2 topic echo /mission_manager/status
ros2 topic echo /auto_steer_angle
ros2 topic echo /auto_throttle
```

## 카메라와 detector

```bash
ros2 run camera_pkg traffic_light_camera_publisher --camera-index 0
ros2 run detect_pkg traffic_light_detection
```

신호등 detector 기본 입력은 `sensor_msgs/Image` 타입의
`/camera/traffic_light/raw`이며 결과만 발행한다.

```text
/detect/traffic_light/detected
/detect/traffic_light/color
/detect/traffic_light/confidence
```

차선 카메라와 장애물 detector:

```bash
ros2 run camera_pkg lane_camera_publisher --camera-index 1
ros2 launch detect_pkg obstacle_detection.launch.py
```

`obstacle_detection.launch.py`는 장애물 YOLO와 4개 초음파 센서를 사용하는
장애물 회피 candidate publisher를 함께 실행한다.

```text
/detect/obstacle/detected                  YOLO 객체 검출 여부
/detect/obstacle/avoidance_active          obstacle mission 활성화
/control/candidate/obstacle/steer_angle    장애물 후보 조향각(deg)
/control/candidate/obstacle/valid          현재 후보 유효 여부
/detect/avoidance/status                   회피 상태 JSON
/ultrasonic/left/front                     왼쪽 앞 거리(cm)
/ultrasonic/left/rear                      왼쪽 뒤 거리(cm)
/ultrasonic/right/front                    오른쪽 앞 거리(cm)
/ultrasonic/right/rear                     오른쪽 뒤 거리(cm)
```

상태 전이:

```text
차선 주행
  -> YOLO 객체 검출 상태를 독립적으로 래치
  -> 좌/우 뒤 초음파의 "장애물 감지 후 해제" 상태를 독립적으로 래치
  -> 두 상태가 모두 준비되는 순간 avoidance_active=true
  -> 뒤 초음파가 감지했던 방향으로 즉시 풀조향 후보 발행
  -> 반대 방향 앞 초음파 거리가 감소한 뒤 증가
  -> avoidance_active=false, lane candidate로 복귀
```

YOLO 검출과 초음파 감지→해제의 순서는 상관없다. 먼저 들어온 조건은
상태로 유지되고, 나머지 조건까지 충족되는 콜백에서 바로 풀조향한다.
예를 들어 왼쪽 뒤 센서로 장애물을 잡았다면 왼쪽 풀조향을 발행하고
오른쪽 앞 센서의 거리 감소→증가를 기다린다. 오른쪽 뒤에서 시작한
경우에는 반대로 오른쪽 풀조향과 왼쪽 앞 센서를 사용한다.

회피 상태는 다음 토픽에서 확인한다.

```bash
ros2 topic echo /detect/avoidance/status
```

상태 JSON의 `yolo_latched`와 `rear_obstacle_state`에서 두 조건의 진행
상태를 각각 확인할 수 있다. `rear_obstacle_state`는
`wait_detection` → `detected` → `cleared_after_detection` 순서로 변한다.

임계값, 풀조향각, 거리 증감 판정과 timeout은
`src/detect_pkg/config/obstacle_detector.yaml`에서 조정한다.
뒤 센서의 `-1.0`은 에코 없음이므로 `rear_no_echo_is_clear: true`일 때
연속 프레임 조건을 만족하면 장애물 해제로 처리한다.

## Arduino serial bridge

권장 실차 실행은 설치된
`share/control_pkg/config/serial_bridge.yaml`을 자동으로 읽는 launch
방식이다. 소스 설정 파일은
`src/control_pkg/config/serial_bridge.yaml`에 있다.

```bash
ros2 launch control_pkg serial_bridge.launch.py
```

포트 등 자주 바꾸는 값은 YAML보다 우선하는 launch argument로 지정한다.

```bash
ros2 launch control_pkg serial_bridge.launch.py \
  serial_port:=/dev/ttyACM0 \
  baudrate:=115200 \
  command_timeout_sec:=0.4
```

YAML을 적용하지 않는 단독 디버깅에는 기존 직접 실행도 사용할 수 있다.

```bash
ros2 run control_pkg serial_bridge \
  --serial-port /dev/ttyACM0 --baudrate 115200
```

bridge는 최종 `/auto_steer_angle`, `/auto_throttle`을 구독한다. 두 명령이
모두 `command_timeout_sec` 안에 갱신된 경우에만 구동 heartbeat를 Arduino로
보내며, 하나라도 stale이면 구동을 정지한다. 기본값은 0.3초이고 timeout
때 조향은 마지막 목표를 유지한다. 포트가 없거나 USB가 분리되면 노드는
종료되지 않고 재연결한다.

Arduino는 유효한 `D,...` 명령이 500ms 동안 없으면 별도 watchdog으로
구동 모터를 정지한다.

차량 telemetry:

```text
/vehicle/current_steering_angle  실제 조향각(deg)
/vehicle/drive_pwm               명령 PWM(-255~255), 실제 측정 속도 아님
/ultrasonic/left/front           왼쪽 앞 거리(cm)
/ultrasonic/left/rear            왼쪽 뒤 거리(cm)
/ultrasonic/right/front          오른쪽 앞 거리(cm)
/ultrasonic/right/rear           오른쪽 뒤 거리(cm)
```

실차 구동 전에는 바퀴를 지면에서 띄우고 조향 부호, 전후진 방향,
stale 정지, USB 분리 후 정지를 먼저 확인한다.
