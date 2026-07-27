# drive_pkg 실행

## 1. USB 카메라 압축 토픽 실행

카메라 장치 번호는 `ls -l /dev/video*`로 확인한다.

```bash
source /opt/ros/humble/setup.bash

ros2 run usb_cam usb_cam_node_exe \
  --ros-args \
  -r __ns:=/camera1 \
  -p video_device:=/dev/video2 \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0 \
  -p pixel_format:=mjpeg2rgb
```

USB 카메라 입력 토픽:

```text
/camera1/image_raw/compressed
```

## 2. drive_pkg 빌드

```bash
cd /home/tak/dolbatS
source /opt/ros/humble/setup.bash
colcon build --packages-select drive_pkg
source install/setup.bash
```

## 3. 차선 세그멘테이션 및 경로 생성

기본값은 실제 주행 명령이 비활성화된 dry-run이다.

```bash
ros2 run drive_pkg yolo_lane_driver --display
```

파라미터를 한 파일에서 조정하려면
`config/yolo_lane_driver.yaml`을 수정하고 다음처럼 실행한다:

```bash
ros2 run drive_pkg yolo_lane_driver --ros-args \
  --params-file /home/tak/dolbatS/src/drive_pkg/config/yolo_lane_driver.yaml
```

YAML의 `enable_drive` 기본값은 안전을 위해 `false`이다. 실제 주행 시에만
`true`로 변경한다. YAML 수정 후에는 노드만 재시작하면 된다.

모델은 차선 종류를 별도 클래스로 구분하지 않으므로, 같은 곡선으로 묶인
분리 마스크가 `dashed_piece_threshold`개 이상이면 점선으로 판정한다.
`prefer_solid_when_dashed: true`이면 점선과 함께 검출된 연속 실선을
우선 기준 경계로 사용한다.

차량 중심에 가장 가까운 좌·우 차선을 비교하여 왼쪽이 실선이고 오른쪽이
점선이면 `/which/lane`에 `lane_1`, 왼쪽이 점선이고 오른쪽이 실선이면
`lane_2`를 발행한다. 두 차선이 모두 같은 종류이거나 한쪽만 검출되면
차선 번호를 발행하지 않는다.

현재 기본 차량 형상 및 동적 LD 설정:

```bash
ros2 run drive_pkg yolo_lane_driver \
  --lane-width-m 0.90 \
  --bev-reference-forward-offset-m 1.04 \
  --lookahead-min-m 1.1 \
  --lookahead-max-m 2.5 \
  --display
```

`bev-reference-forward-offset-m`는 후륜축 중심에서 BEV 영상 하단 기준점
(현재 차량 앞코)까지의 전방 거리이다. 동적 LD는 조향 요구각 0도에서
최댓값, 최대 조향각에서 최솟값이 되도록 선형으로 조정된다.

BEV 변환은 기본적으로 `resource/bev_params_7.npz`의 `src_points`,
`dst_points`, `warp_w`, `warp_h`를 사용한다. 다른 파일을 사용할 때는:

```bash
ros2 run drive_pkg yolo_lane_driver \
  --bev-params /path/to/bev_params.npz \
  --display
```

ROS 파라미터로 설정하려면:

```bash
ros2 run drive_pkg yolo_lane_driver --ros-args \
  -p image_topic:=/image_raw/compressed \
  -p model_path:=/home/tak/lane_yolo_project/weight/best1.pt \
  -p bev_params:=/home/tak/dolbatS/src/drive_pkg/resource/bev_params_7.npz \
  -p confidence:=0.25 \
  -p display:=true
```

OpenCV 창:

```text
BEV segmentation
BEV generated path
```

출력 토픽:

```text
/lane/path
/lane/yolo_drive/segmentation/compressed
/lane/yolo_drive/path/compressed
/lane/yolo_drive/status
/which/lane
/cmd_vel
```

rosbag을 사용할 때 기본 입력 토픽은 다음과 같다.

```text
/image_raw/compressed
```

## 4. rqt_image_view로 확인

```bash
ros2 run rqt_image_view rqt_image_view
```

다음 두 압축 토픽을 각각 확인한다.

```text
/lane/yolo_drive/segmentation/compressed
/lane/yolo_drive/path/compressed
```

## 5. 실제 주행 활성화

경로와 조향값을 충분히 확인한 후에만 사용한다.

```bash
ros2 run drive_pkg yolo_lane_driver \
  --enable-drive \
  --speed-mps 0.20 \
  --display
```
