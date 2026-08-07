"""Launch drive_main (perception + path) and pure_pursuit (control). The
debug visualizer runs in-process inside pure_pursuit (see
pure_pursuit.py's ``local_display`` parameter)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
    initial_lane = LaunchConfiguration("initial_lane")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="YAML file shared by all drive pipeline nodes.",
            ),
            DeclareLaunchArgument(
                "initial_lane",
                default_value="auto",
                description="Initial lane: auto, lane_1, or lane_2.",
            ),
            Node(
                package="drive_pkg",
                executable="drive_main",
                name="drive_main",
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
                parameters=[params_file],
            ),
        ]
    )
