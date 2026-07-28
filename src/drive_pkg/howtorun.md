# drive_pkg 모듈형 파이프라인

`drive_pkg`는 하나의 ROS 2 패키지 안에서 세 노드가 토픽으로 연결된다.

```text
/image_raw/compressed
  -> lane_detect
  -> /lane/detection/mask/compressed
  -> path_plan
  -> /lane/path
  -> pure_pursuit
  -> /cmd_vel
```

각 노드는 따로 실행·교체·확인할 수 있다. 전체 파라미터는
`config/drive_pipeline.yaml`에서 노드별로 조정한다.

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

`pure_pursuit.enable_drive` 기본값은 `false`다. 이때 계산 결과는
`/lane/control/status`에서 확인할 수 있지만 `/cmd_vel`에는 정지 명령만
발행한다. 경로와 조향 방향을 검증한 뒤에만 YAML 값을 `true`로 바꾼다.

각 단계 확인 예:

```bash
ros2 topic hz /lane/detection/mask/compressed
ros2 topic echo /lane/path/status
ros2 topic echo /lane/control/status
```

이미지는 `rqt_image_view`에서 검출/경로 디버그 토픽을 선택해 확인한다.
