from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = (
        get_package_share_directory("detect_pkg")
        + "/config/obstacle_yolo_only.yaml"
    )
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            Node(
                package="detect_pkg",
                executable="obstacle_detector_publisher",
                name="obstacle_detector_publisher",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="detect_pkg",
                executable="yolo_obstacle_yolo_only",
                name="yolo_obstacle_yolo_only",
                parameters=[config_file],
                output="screen",
            ),
        ]
    )
