# T1. lane_camera
```
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

# T2. traffic_light_camera

```
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video4 \
  -p pixel_format:=mjpeg2rgb \
  -p image_width:=640 \
  -p image_height:=480 \
  -p framerate:=30.0 \
  -r image_raw:=/camera/traffic_light/raw \
  -r image_raw/compressed:=/camera/traffic_light/raw/compressed \
  -r camera_info:=/camera/traffic_light/camera_info
```

# T3. serial_bridge
```
ros2 launch control_pkg serial_bridge.launch.py
```
 

# T4. drive_pkg
```
ros2 launch drive_pkg drive_pipeline.launch.py 
```
```
ros2 run drive_pkg yolotl
```

# T5. mission_manager
```
  ros2 launch mission_manager_pkg mission_manager.launch.py
```

# T6. keyboard_toggle
```
ros2 run control_pkg keyboard_drive_toggle 
```

# T7. obstacle
아래 두 launch는 대체 실행 방식이므로 동시에 실행하지 않는다.

```
ros2 launch detect_pkg obstacle_sonic_end.launch.py
```
```
ros2 launch detect_pkg obstacle_yolo_end.launch.py
```

YOLO bbox가 화면 가운데에서 설정 크기에 도달할 때마다 왼쪽, 오른쪽을
번갈아 풀조향하고, `yolo_end`와 같은 bbox 경계 통과 조건으로 종료하는 방식:

```
ros2 launch detect_pkg obstacle_yolo_only.launch.py
```

YOLO bbox 경계 또는 전방 초음파 임계값 중 하나를 먼저 만족하면 회피를
종료하는 OR 방식:

```
ros2 launch detect_pkg obstacle_fusion_end.launch.py
```


# T8. traffic_light
```
ros2 run detect_pkg traffic_light_detection
```



ros2 run plotjuggler plotjuggler -l /home/j/dolbatS/plotJuggler.xml --start_streamer "ROS2 Topic Subscriber"
