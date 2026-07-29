"""Launch drive_main (perception + path) and pure_pursuit (control). The
debug visualizer runs in-process inside pure_pursuit (see
pure_pursuit.py's ``local_display`` parameter)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    default_params = PathJoinSubstitution(
        [
            FindPackageShare("drive_pkg"),
            "config",
            "drive_pipeline.yaml",
        ]
    )
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="YAML file shared by all drive pipeline nodes.",
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
                parameters=[params_file],
            ),
        ]
    )
