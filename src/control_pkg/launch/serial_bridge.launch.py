"""Launch serial_bridge with its installed YAML configuration."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _optional_override(context, argument_name, parameter_name, value_type):
    raw_value = LaunchConfiguration(argument_name).perform(context).strip()
    if not raw_value:
        return {}
    try:
        return {parameter_name: value_type(raw_value)}
    except ValueError as exc:
        type_name = value_type.__name__
        raise ValueError(
            f"launch argument {argument_name} must be {type_name}, "
            f"got {raw_value!r}"
        ) from exc


def _launch_setup(context):
    overrides = {}
    overrides.update(
        _optional_override(
            context, "serial_port", "serial_port", str
        )
    )
    # The public launch argument follows the existing CLI spelling, while
    # the ROS parameter is named baud_rate in serial_bridge.py and YAML.
    overrides.update(
        _optional_override(context, "baudrate", "baud_rate", int)
    )
    for name in (
        "command_timeout_sec",
        "reconnect_interval_sec",
        "serial_startup_delay",
    ):
        overrides.update(
            _optional_override(context, name, name, float)
        )

    return [
        Node(
            package="control_pkg",
            executable="serial_bridge",
            name="serial_bridge",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                overrides,
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    default_params_file = PathJoinSubstitution(
        [
            FindPackageShare("control_pkg"),
            "config",
            "serial_bridge.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="serial_bridge ROS parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "serial_port",
                default_value="",
                description="Override YAML serial_port when non-empty.",
            ),
            DeclareLaunchArgument(
                "baudrate",
                default_value="",
                description="Override YAML baud_rate (integer) when set.",
            ),
            DeclareLaunchArgument(
                "command_timeout_sec",
                default_value="",
                description=(
                    "Override YAML command_timeout_sec (double) when set."
                ),
            ),
            DeclareLaunchArgument(
                "reconnect_interval_sec",
                default_value="",
                description=(
                    "Override YAML reconnect_interval_sec (double) when set."
                ),
            ),
            DeclareLaunchArgument(
                "serial_startup_delay",
                default_value="",
                description=(
                    "Override YAML serial_startup_delay (double) when set."
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
