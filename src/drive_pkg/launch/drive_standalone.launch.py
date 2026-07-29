"""Lane-only test launch.

Unlike the integrated ``drive_pipeline.launch.py``, this explicitly maps the
lane candidate outputs to the final vehicle command topics. Do not run it
together with a future mission_manager.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [FindPackageShare("drive_pkg"), "config", "drive_pipeline.yaml"]
    )
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="YAML file shared by the standalone lane nodes.",
            ),
            Node(
                package="drive_pkg",
                executable="drive_main",
                name="drive_main",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="drive_pkg",
                executable="pure_pursuit",
                name="pure_pursuit",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "steer_angle_topic": "/auto_steer_angle",
                        "throttle_topic": "/auto_throttle",
                    },
                ],
            ),
        ]
    )
