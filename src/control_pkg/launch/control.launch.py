from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = (
        get_package_share_directory("control_pkg") + "/config/control.yaml"
    )
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            Node(
                package="control_pkg",
                executable="serial_bridge",
                name="serial_bridge",
                parameters=[config_file],
                output="screen",
            ),
        ]
    )
