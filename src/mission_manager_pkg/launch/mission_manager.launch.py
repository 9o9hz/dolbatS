"""Launch mission_manager with its installed YAML configuration."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [
            FindPackageShare("mission_manager_pkg"),
            "config",
            "mission_manager.yaml",
        ]
    )
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="mission_manager ROS parameter YAML file.",
            ),
            Node(
                package="mission_manager_pkg",
                executable="mission_manager",
                name="mission_manager",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
