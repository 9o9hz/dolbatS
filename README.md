# dolbatS

## 명령어 체계
돌쇠는 다음과 같은 명령어 체계를 가지고 움직입니다. `Serial` 통신을 통해 `115200 baudrate`로 명령어를 전송받습니다. 모든 명령어의 끝은 `\n`으로 끝나야 합니다.


```
D,DIR,SPEED\n
S,ANGLE\n
```

첫 번째 명령어는 돌쇠를 앞뒤로 움직이게 하는 명령어입니다. 

`DIR`은 방향이며 전진은 `F`, 후진은 `R`, 정지는 `S`입니다.

`SPEED`가 뜻하는 것은 속도입니다. 돌쇠는 `0`부터 `255`까지 속도를 가집니다.

만일 돌쇠를 앞으로 128의 속도만큼 가게 하고 싶으면, 다음과 같이 명령어를 작성하면 됩니다.

```
D,F,128\n
```

두 번째 명령어는 돌쇠 앞바퀴를 회전시키는 명령어입니다.

`ANGLE`이 뜻하는 것은 돌쇠 앞바퀴의 목표 각도의 소수점 첫째 자리까지의 값입니다. 전방을 향하게 할 때 `0`입니다. 왼쪽이 양수, 오른쪽이 음수입니다.

만일 돌쇠의 앞바퀴를 왼쪽으로 15.3도만큼 움직이고 싶다면, 다음과 같이 명령어를 작성하면 됩니다.

```
S,15.3\n
```

자세한 내용은 `steering_test.py` 파일 안에 `send_steer()`, `send_drive()` 함수를 참조하면 됩니다.

Arduino Mega의 초음파 핀은 `ECHO/TRIG` 순서로 왼쪽 앞 22/23번,
왼쪽 뒤 24/25번, 오른쪽 앞 26/27번, 오른쪽 뒤 28/29번입니다.
아두이노는 10ms마다
`signed_drive_pwm,현재조향각,왼쪽앞cm,왼쪽뒤cm,오른쪽앞cm,오른쪽뒤cm`
형식으로 최신 상태를 송출합니다. 첫 값은 실제 측정 속도가 아니라 현재
구동 명령 PWM이며 범위는 `-255~255`입니다(양수=전진, 음수=후진).
초음파 센서는 상호 간섭을 줄이기 위해 네 개를 하나씩 순차 측정하며,
측정 실패 또는 범위 초과는 `-1.0`으로 표시합니다.

유효한 `D,...` 명령이 500ms 동안 들어오지 않으면 Arduino watchdog이
구동 PWM을 0으로 만들고 정지한다. 새 유효 명령이 오면 정상 제어로
복귀하며, timeout 때 조향 목표는 갑자기 중앙으로 바꾸지 않고 유지한다.

```
128,5.2,35.2,40.1,41.8,38.7
-80,-10.1,-1.0,120.4,80.2,-1.0
```

## ROS2 카메라 인식 토픽 발행

`traffic_light_camera_publisher`는 `/camera/traffic_light/raw`, `lane_camera_publisher`는 `/camera/lane/raw`를 발행합니다. `obstacle_detector_publisher`는 차선 카메라 토픽을 구독해 `dolsoi-model-v2.pt`로 객체를 찾고 감지 여부, 박스와 하단 중심 좌표를 발행합니다.

## ROS2 차선 주행 파이프라인

통합 구성에서 `pure_pursuit`는 최종 차량 명령이 아니라 차선 후보만
발행합니다.

```text
lane_detect -> /lane/detection/mask/compressed
            -> path_plan -> /lane/path
                         -> pure_pursuit
                              -> /control/candidate/lane/steer_angle
                              -> /control/candidate/lane/valid

mission_manager -> /auto_steer_angle, /auto_throttle
                -> serial_bridge -> Arduino
```

`mission_manager`만 최종 조향과 throttle을 30Hz로 발행한다. 장애물 회피
candidate publisher는 아직 구현되지 않았다. 따라서 장애물이 검출됐지만
유효한 장애물 candidate를 한 번도 받지 못한 경우에는 정지한다.

통합 실행은 세 터미널에서 다음 순서로 시작한다.

```bash
ros2 launch drive_pkg drive_pipeline.launch.py
ros2 launch mission_manager_pkg mission_manager.launch.py
ros2 launch control_pkg serial_bridge.launch.py
```

`serial_bridge`의 권장 실차 실행은 YAML 설정을 자동 적용하는 launch 방식이다.

```bash
ros2 launch control_pkg serial_bridge.launch.py
```

설정 파일은 `src/control_pkg/config/serial_bridge.yaml`이며, 포트는 다음처럼
덮어쓸 수 있다.

```bash
ros2 launch control_pkg serial_bridge.launch.py \
  serial_port:=/dev/ttyACM0
```

`ros2 run control_pkg serial_bridge`는 YAML이 자동 적용되지 않는 단독
디버깅 방식이다.

파라미터는
`src/drive_pkg/config/drive_pipeline.yaml` 한 곳에서 노드별로 관리합니다.
차선 후보 구성은 `ros2 launch drive_pkg drive_pipeline.launch.py`를
사용한다. 상세 토픽과 실행 예시는
`src/drive_pkg/howtorun.md`를 참고하세요.

필요 패키지:

```
sudo apt install python3-colcon-common-extensions
python3 -m pip install -r requirements.txt
```

실행(각각 별도 터미널):

```
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 run camera_pkg traffic_light_camera_publisher --camera-index 0
```

```
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run camera_pkg lane_camera_publisher --camera-index 1
```

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run detect_pkg obstacle_detector_publisher
```

카메라 인덱스를 실행 인자로 지정하려면:

```
ros2 run camera_pkg lane_camera_publisher --camera-index 1
```

디버그 창으로 카메라 화면을 보려면:

```
ros2 run detect_pkg obstacle_detector_publisher --debug-window
```

만약 `install/setup.bash`가 없고 `install/setup.sh`만 있다면 `python3-colcon-common-extensions`가 빠져 있을 가능성이 큽니다. 설치한 뒤 새 터미널에서 다시 빌드하세요.

`AttributeError: _ARRAY_API not found` 또는 `ImportError: numpy.core.multiarray failed to import`가 나오면 NumPy 2.x와 Ubuntu의 matplotlib/OpenCV 바이너리가 충돌한 것입니다. 아래처럼 NumPy를 1.x로 낮춘 뒤 다시 실행하세요.

```
python3 -m pip install --force-reinstall "numpy<2" "opencv-python<4.12"
colcon build
source install/setup.bash
```

기본 토픽:

| 토픽 | 타입 | 내용 |
| --- | --- | --- |
| `/camera/traffic_light/raw` | `sensor_msgs/Image` | 신호등 카메라 raw BGR 프레임 |
| `/camera/lane/raw` | `sensor_msgs/Image` | 차선 카메라 raw BGR 프레임 |
| `/camera/lane/detection_view/compressed` | `sensor_msgs/CompressedImage` | 차선 카메라의 JPEG 압축 장애물 감지 영상 |
| `/detect/traffic_light/detected` | `std_msgs/Bool` | 신호등 감지 여부 |
| `/detect/traffic_light/color` | `std_msgs/String` | `red`, `yellow`, `green`, `none` |
| `/detect/traffic_light/confidence` | `std_msgs/Float32` | 선택한 신호등 검출 confidence, 미검출은 `0.0` |
| `/detect/obstacle/detected` | `std_msgs/Bool` | 감지 여부. 매 프레임 발행 |
| `/detect/obstacle/bbox` | `std_msgs/Float32MultiArray` | 감지된 경우에만 `[center_x, center_y, width, height]` 발행 |
| `/detect/obstacle/ultrasonic_enabled` | `std_msgs/Bool` | 화면 왼쪽 절반의 YOLO 대상이 사라진 뒤 초음파 판정을 활성화 |
| `/ultrasonic/left/front` | `std_msgs/Float32` | 왼쪽 앞 초음파 거리(cm) |
| `/ultrasonic/left/rear` | `std_msgs/Float32` | 왼쪽 뒤 초음파 거리(cm) |
| `/ultrasonic/right/front` | `std_msgs/Float32` | 오른쪽 앞 초음파 거리(cm) |
| `/ultrasonic/right/rear` | `std_msgs/Float32` | 오른쪽 뒤 초음파 거리(cm) |
| `/vehicle/drive_pwm` | `std_msgs/Float32` | Arduino의 구동 명령 PWM(`-255~255`), 실제 측정 속도 아님 |
| `/detect/obstacle_event` | `std_msgs/Int8MultiArray` | 초음파 장애물 상태가 바뀔 때만 `[event, avoid_direction]` 발행 |

## ROS2 초음파 장애물 이벤트

`ultrasonic_obstacle_event`는 `/detect/obstacle/ultrasonic_enabled`가 참일
때만 좌우 거리 한 쌍을 한 프레임으로 처리합니다.
기본값으로 40cm 이하가 3프레임 연속되면 감지 이벤트를, 양쪽 모두
45cm 초과가 3프레임 연속되면 해제 이벤트를 발행합니다. `-1.0`, NaN,
무한대는 유효한 거리로 세지 않으며, 센서 오류만으로 해제 이벤트가
발생하지 않습니다.

이벤트 데이터의 코드는 다음과 같습니다.

| 배열 위치 | 값 | 의미 |
| --- | --- | --- |
| `data[0]` | `1` / `0` | 장애물 감지 / 장애물 사라짐 |
| `data[1]` | `1` / `-1` | 왼쪽 회피 / 오른쪽 회피 |

감지와 그에 대응하는 해제 이벤트에는 같은 회피 방향이 들어갑니다. 따라서
다음 회피 로직은 마지막 이벤트의 `data[1]`을 저장해 두었다가 사용할 수
있습니다. 왼쪽 센서가 막히면 오른쪽, 오른쪽 센서가 막히면 왼쪽을
선택하며 양쪽이 모두 막히면 더 여유 있는 쪽을 선택합니다.

실행:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select control_pkg
source install/setup.bash
ros2 run control_pkg ultrasonic_obstacle_event
```

임계값과 연속 프레임 수 변경:

```bash
ros2 run control_pkg ultrasonic_obstacle_event --ros-args \
  -p detect_threshold_cm:=35.0 \
  -p clear_threshold_cm:=40.0 \
  -p consecutive_frames:=5
```

이벤트 확인:

```bash
ros2 topic echo /detect/obstacle_event
```

주요 파라미터:

```
ros2 run camera_pkg traffic_light_camera_publisher --ros-args \
  -p camera_index:=0 \
  -p raw_image_topic:=/camera/traffic_light/raw
```

```
ros2 run detect_pkg obstacle_detector_publisher --ros-args \
  -p confidence_threshold:=0.5 \
  -p raw_image_topic:=/camera/lane/raw \
  -p show_window:=true
```
