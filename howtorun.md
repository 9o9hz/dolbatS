# T1. camera
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

# T2. serial_bridge
```
ros2 launch control_pkg serial_bridge.launch.py
```

# T3. drive_pkg
```
ros2 launch drive_pkg drive_pipeline.launch.py
```

# T4. mission_manager
```
ros2 launch mission_manager_pkg mission_manager.launch.py
```

# T5. keyboard_toggle
```
ros2 run control_pkg keyboard_drive_toggle 
```

# T6. obstacle
```
ros2 launch detect_pkg obstacle_detection.launch.py
```

# T7. traffic_light
```
ros2 run detect_pkg traffic_light_detection
```