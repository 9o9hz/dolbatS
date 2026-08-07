"""Run modular lane driving with camera and optional Arduino bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    enable_drive = LaunchConfiguration("enable_drive")
    initial_lane = LaunchConfiguration("initial_lane")
    default_params = PathJoinSubstitution(
        [FindPackageShare("drive_pkg"), "config", "drive_pipeline.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("video_device", default_value="/dev/video0"),
            DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("enable_drive", default_value="false"),
            DeclareLaunchArgument(
                "initial_lane",
                default_value="auto",
                description="Initial lane: auto, lane_1, or lane_2.",
            ),
            DeclareLaunchArgument(
                "launch_serial_bridge", default_value="false"
            ),
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name="modular_lane_camera",
                output="screen",
                parameters=[
                    {
                        "video_device": ParameterValue(
                            LaunchConfiguration("video_device"), value_type=str
                        ),
                        "image_width": 640,
                        "image_height": 480,
                        "framerate": 30.0,
                        "io_method": "mmap",
                        "pixel_format": "mjpeg2rgb",
                        "camera_name": "lane_camera",
                        "frame_id": "lane_camera",
                    }
                ],
                remappings=[
                    ("image_raw", "/camera/lane/raw"),
                    (
                        "image_raw/compressed",
                        "/camera/lane/raw/compressed",
                    ),
                    ("camera_info", "/camera/lane/camera_info"),
                ],
            ),
            Node(
                package="drive_pkg",
                executable="lane_detect",
                name="lane_detect",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="drive_pkg",
                executable="path_plan",
                name="path_plan",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "initial_lane": ParameterValue(
                            initial_lane, value_type=str
                        )
                    },
                ],
            ),
            Node(
                package="drive_pkg",
                executable="pure_pursuit",
                name="pure_pursuit",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "enable_drive": ParameterValue(
                            enable_drive, value_type=bool
                        )
                    },
                ],
            ),
            Node(
                package="control_pkg",
                executable="serial_bridge",
                name="modular_serial_bridge",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("launch_serial_bridge")
                ),
                parameters=[
                    {
                        "serial_port": ParameterValue(
                            LaunchConfiguration("serial_port"), value_type=str
                        ),
                        "baud_rate": 115200,
                        "cmd_vel_topic": "/cmd_vel",
                        "wheelbase": 0.545,
                        "max_steer_angle": 26.5,
                        "command_timeout": 0.5,
                    }
                ],
            ),
            Node(
                package="control_pkg",
                executable="keyboard_drive_toggle",
                name="keyboard_drive_toggle",
                output="screen",
            ),
        ]
    )
