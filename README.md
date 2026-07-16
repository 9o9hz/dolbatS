# dolbatS

## 명령어 체계
돌쇠는 다음과 같은 명령어 체계를 가지고 움직입니다. `Serial` 통신을 통해 `9600 baudrate`로 명령어를 전송받습니다. 모든 명령어의 끝은 `\n`으로 끝나야 합니다.


```
D,DIR,SPEED\n
S,ANGLE\n
```

첫 번째 명령어는 돌쇠를 앞뒤로 움직이게 하는 명령어입니다. 

`DIR`이 뜻하는 것은 방향으로, 앞으로 가기 위해서는 `F`, 뒤로 가기 위해서는 `B`를 작성하면 됩니다.

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

아두이노는 300ms마다 현재 상태를 `angle,speed` 형식으로 송출합니다. 조향각은 소수점 첫째 자리까지 표시하고, 전진 속도는 양수, 후진 속도는 음수, 정지는 `0`으로 표시합니다.

```
-12.3,100
0.0,-80
```

## ROS2 카메라 인식 토픽 발행

`traffic_light_camera_publisher`는 `/camera/traffic_light/raw`, `lane_camera_publisher`는 `/camera/lane/raw`를 발행합니다. `obstacle_detector_publisher`는 신호등 카메라 토픽을 구독해 `dolsoi-model-v2.pt`로 객체를 찾고 감지 여부, 박스와 하단 중심 좌표를 발행합니다.

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
| `/camera/traffic_light/detection_view` | `sensor_msgs/Image` | 신호등 카메라의 RViz용 감지 영상 |
| `/camera/lane/raw` | `sensor_msgs/Image` | 차선 카메라 raw BGR 프레임 |
| `/detect/obstacle/detected` | `std_msgs/Bool` | 감지 여부. 매 프레임 발행 |
| `/detect/obstacle/bbox` | `std_msgs/Float32MultiArray` | 감지된 경우에만 `[center_x, center_y, width, height]` 발행 |
| `/detect/obstacle/bottom_center` | `std_msgs/Float32MultiArray` | 감지된 경우에만 바운딩 박스 하단 중심 `[x, y]` 발행 |

주요 파라미터:

```
ros2 run camera_pkg traffic_light_camera_publisher --ros-args \
  -p camera_index:=0 \
  -p raw_image_topic:=/camera/traffic_light/raw
```

```
ros2 run detect_pkg obstacle_detector_publisher --ros-args \
  -p confidence_threshold:=0.5 \
  -p raw_image_topic:=/camera/traffic_light/raw \
  -p show_window:=true
```
