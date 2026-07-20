# clone 하고 초기 설정
```
cd /home/$(whoami)/dolbatS
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

# 기본 준비
```
cd /home/$(whoami)/dolbatS
deactivate 2>/dev/null || true
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
sudo chmod 666 /dev/tty*
```

# Publisher
## `/camera/traffic_light/*`

```bash
ros2 run camera_pkg traffic_light_camera_publisher --camera-index 0
```

## `/camera/lane/raw`
```bash
ros2 run camera_pkg lane_camera_publisher \
  --camera-index 1
```

## `/detect/obstacle/*`
```bash
ros2 run detect_pkg obstacle_detector_publisher \
  --debug-window
```

# `/cmd_vel`
```bash
ros2 run control_pkg serial_bridge \
  --serial-port /dev/ttyUSB0 \
  --baudrate 115200
```
