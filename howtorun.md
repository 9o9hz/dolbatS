# T1. lane_camera
```
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video4 \
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
  -p video_device:=/dev/video2 \
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
ros2 launch drive_pkg drive_pipeline.launch.py initial_lane:=lane_2


ros2 run drive_pkg drive_main --ros-args \
  --params-file /home/j/dolbatS/src/drive_pkg/config/drive_pipeline.yaml \
  -p initial_lane:=lane_1
```
```

# 시간 주행 yolotl
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
L: 2차선, R: 1차선

```
ros2 launch detect_pkg obstacle_yolo_only.launch.py avoid_direction:=L
ros2 launch detect_pkg obstacle_yolo_only.launch.py avoid_direction:=R
```

# T8. traffic_light
```
ros2 run detect_pkg traffic_light_detection
```



ros2 run plotjuggler plotjuggler -l /home/j/dolbatS/plotJuggler.xml --start_streamer "ROS2 Topic Subscriber"


# 1. 현재 살아있는 ROS 2 프로세스 확인 
ps aux | grep -i ros2

# 2. 데몬 재시작으로 discovery 캐시 초기화
ros2 daemon stop
ros2 daemon start
