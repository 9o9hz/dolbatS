# dolbatS 실행 요약

상세 차선 파이프라인 파라미터와 토픽은
`src/drive_pkg/howtorun.md`를 기준으로 한다.

## 공통 준비

```bash
cd ~/dolbatS
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 통합 준비 구성

```bash
ros2 launch drive_pkg drive_pipeline.launch.py
```

이 launch는 `drive_main`과 `pure_pursuit`를 실행한다. Pure Pursuit 출력은
다음 lane candidate 토픽이다.

```text
/control/candidate/lane/steer_angle  std_msgs/Float32 (deg)
/control/candidate/lane/valid        std_msgs/Bool
```

`mission_manager`가 후보와 인식 결과를 판단해 최종
`/auto_steer_angle`, `/auto_throttle`을 발행한다.

후보 확인:

```bash
ros2 topic echo /control/candidate/lane/steer_angle
ros2 topic echo /control/candidate/lane/valid
```

## Mission manager

```bash
ros2 launch mission_manager_pkg mission_manager.launch.py
```

차선·장애물 후보와 detector 결과를 받아 `lane`, `traffic_light`,
`obstacle` 중 하나를 선택한다. 차선 candidate의 마지막 유효값은
timeout 없이 유지한다. 노드 시작 후 유효한 차선 candidate를 한 번도
받지 않았다면 최종 조향과 throttle을 0으로 발행한다.

장애물 판단에는 YOLO의 프레임별 `/detect/obstacle/detected` 대신
초음파 감지부터 회전 완료까지 유지되는
`/detect/obstacle/avoidance_active`를 사용한다. 회피 중 obstacle
candidate가 invalid가 되면 과거 풀조향값을 유지하지 않고 최종 조향과
throttle을 모두 0으로 발행한다.

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

`obstacle_detection.launch.py`는 장애물 YOLO, 초음파 이벤트 필터, 장애물
회피 candidate publisher를 함께 실행한다.

```text
/detect/obstacle/enable                    YOLO 추론 활성화
/detect/obstacle/avoidance_active          obstacle mission 활성화
/control/candidate/obstacle/steer_angle    장애물 후보 조향각(deg)
/control/candidate/obstacle/valid          현재 후보 유효 여부
/detect/avoidance/status                   회피 상태 JSON
```

상태 전이:

```text
차선 주행
  -> 좌/우 초음파 임계값 이하 연속 감지
  -> avoidance_active=true, YOLO ON, 직진 조향 후보 0도
  -> 감지됐던 쪽 초음파에서 장애물이 사라짐
  -> 같은 방향 풀조향 후보
  -> 해당 센서 거리가 감소한 뒤 증가
  -> avoidance_active=false, YOLO OFF, lane candidate로 복귀
```

YOLO bbox 하단 중앙점과 차량 폭의 BEV 결과는 다음 토픽에서 확인한다.

```bash
ros2 topic echo /detect/obstacle/bev_footprint
ros2 topic echo /detect/avoidance/status
```

시각화는 `rqt_image_view`에서 `/detect/obstacle/bev_view`를 선택한다.
임계값, 풀조향각, 거리 증감 판정과 timeout은
`src/detect_pkg/config/obstacle_detector.yaml`에서 조정한다.

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
/ultrasonic/left_distance        cm
/ultrasonic/right_distance       cm
```

실차 구동 전에는 바퀴를 지면에서 띄우고 조향 부호, 전후진 방향,
stale 정지, USB 분리 후 정지를 먼저 확인한다.
