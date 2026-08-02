"""Run the lane pipeline safely against a compressed-image rosbag."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    bag_path = LaunchConfiguration("bag_path")
    rate = LaunchConfiguration("rate")
    show_visualizer = LaunchConfiguration("show_visualizer")
    default_params = PathJoinSubstitution(
        [FindPackageShare("drive_pkg"), "config", "drive_pipeline.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag_path",
                description="Path to a rosbag2 directory.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
            ),
            DeclareLaunchArgument("rate", default_value="1.0"),
            DeclareLaunchArgument(
                "show_visualizer",
                default_value="true",
            ),
            Node(
                package="drive_pkg",
                executable="drive_main",
                name="drive_main",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "local_display": False,
                        "image_topic": "/camera/lane/raw/compressed",
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
                        "local_display": ParameterValue(
                            show_visualizer,
                            value_type=bool,
                        ),
                        # Never feed a live mission manager during playback.
                        "steer_angle_topic": (
                            "/debug/control/candidate/lane/steer_angle"
                        ),
                        "candidate_valid_topic": (
                            "/debug/control/candidate/lane/valid"
                        ),
                    },
                ],
            ),
            TimerAction(
                period=3.0,
                actions=[
                    ExecuteProcess(
                        cmd=[
                            "ros2",
                            "bag",
                            "play",
                            bag_path,
                            "--rate",
                            rate,
                            "--loop",
                            "--topics",
                            "/camera/lane/raw/compressed",
                        ],
                        output="screen",
                    )
                ],
            ),
        ]
    )
