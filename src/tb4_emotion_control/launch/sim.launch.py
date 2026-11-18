"""TurtleBot 4 in Gazebo Harmonic on an open 1 m grid (no Nav2 needed for this lab)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = FindPackageShare("tb4_emotion_control")
    desc = FindPackageShare("nav2_minimal_tb4_description")
    model = PathJoinSubstitution([desc, "urdf", "standard", "turtlebot4.urdf.xacro"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "world", default_value=PathJoinSubstitution([pkg, "worlds", "emotion_arena.sdf"])
            ),
            ExecuteProcess(
                cmd=["gz", "sim", "-r", "-s", LaunchConfiguration("world")],
                output="screen",
                sigterm_timeout="2",
                sigkill_timeout="2",
            ),
            ExecuteProcess(
                cmd=["gz", "sim", "-g"],
                condition=UnlessCondition(LaunchConfiguration("headless")),
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "robot_description": ParameterValue(
                            Command(["xacro", " ", model]), value_type=str
                        ),
                    }
                ],
            ),
            # fixed demo cameras in the world -> ROS images (used by demo_recorder)
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="demo_camera_bridge",
                arguments=[
                    "/demo_cam/perspective@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/demo_cam/top@sensor_msgs/msg/Image[gz.msgs.Image",
                ],
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("nav2_minimal_tb4_sim"), "launch", "spawn_tb4.launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim_time": "true",
                    "robot_name": "turtlebot4",
                    "x_pose": "0.0",
                    "y_pose": "0.0",
                    "z_pose": "0.01",
                    "yaw": "0.0",
                }.items(),
            ),
        ]
    )
