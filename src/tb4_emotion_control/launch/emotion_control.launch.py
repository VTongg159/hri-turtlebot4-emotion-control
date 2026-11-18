"""Webcam emotion detector + motion controller, optionally with the Gazebo TurtleBot 4.

    ros2 launch tb4_emotion_control emotion_control.launch.py start_sim:=true
    ros2 launch tb4_emotion_control emotion_control.launch.py backend:=fer
    ros2 launch tb4_emotion_control emotion_control.launch.py camera_source:=/path/video.mp4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("tb4_emotion_control")
    params = PathJoinSubstitution([pkg, "config", "emotion_control.yaml"])
    use_sim_time = ParameterValue(LaunchConfiguration("start_sim"), value_type=bool)
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_sim", default_value="false"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("backend", default_value="auto"),
            DeclareLaunchArgument("camera_source", default_value="0"),
            DeclareLaunchArgument("show_window", default_value="true"),
            DeclareLaunchArgument("cmd_vel_stamped", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution([pkg, "launch", "sim.launch.py"])),
                launch_arguments={"headless": LaunchConfiguration("headless")}.items(),
                condition=IfCondition(LaunchConfiguration("start_sim")),
            ),
            Node(
                package="tb4_emotion_control",
                executable="emotion_detector_node",
                name="emotion_detector",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params,
                    {
                        # "0" must stay a string (it may also be a video path)
                        "camera_source": ParameterValue(
                            LaunchConfiguration("camera_source"), value_type=str
                        ),
                        "backend": ParameterValue(LaunchConfiguration("backend"), value_type=str),
                        "show_window": ParameterValue(
                            LaunchConfiguration("show_window"), value_type=bool
                        ),
                    },
                ],
            ),
            Node(
                package="tb4_emotion_control",
                executable="emotion_motion_node",
                name="emotion_motion",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params,
                    {
                        "use_sim_time": use_sim_time,
                        "cmd_vel_stamped": ParameterValue(
                            LaunchConfiguration("cmd_vel_stamped"), value_type=bool
                        ),
                    },
                ],
            ),
        ]
    )
