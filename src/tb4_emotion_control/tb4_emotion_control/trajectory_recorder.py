"""Log /odom together with the active emotion/action to CSV for later plotting.

    ros2 run tb4_emotion_control trajectory_recorder --ros-args -p output:=evidence/run.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .odometry import yaw_from_quaternion
from .ros_utils import ignore_further_interrupts, init_ros


class TrajectoryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_recorder")
        out = Path(self.declare_parameter("output", "trajectory.csv").value).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.file = out.open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["t", "x", "y", "yaw", "v", "w", "label", "action"])
        self.label, self.action = "", ""
        self.t0: float | None = None  # node clock (sim time with use_sim_time)
        self.rows = 0
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/emotion/action", self._on_action, 10)
        self.get_logger().info(f"recording to {out}")

    def _now(self) -> float:
        """Node clock in seconds: simulation time when use_sim_time is set, else wall time."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_action(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.label = data.get("label", "")
        self.action = data.get("action", "")

    def _on_odom(self, msg: Odometry) -> None:
        if self.t0 is None:
            self.t0 = self._now()
        pose, twist = msg.pose.pose, msg.twist.twist
        q = pose.orientation
        self.writer.writerow(
            [
                f"{self._now() - self.t0:.3f}",
                f"{pose.position.x:.4f}",
                f"{pose.position.y:.4f}",
                f"{yaw_from_quaternion(q.x, q.y, q.z, q.w):.4f}",
                f"{twist.linear.x:.3f}",
                f"{twist.angular.z:.3f}",
                self.label,
                self.action,
            ]
        )
        self.rows += 1
        if self.rows % 50 == 0:
            self.file.flush()

    def close(self) -> None:
        self.file.close()
        self.get_logger().info(f"wrote {self.rows} samples")


def main(args=None) -> None:
    init_ros(args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ignore_further_interrupts()
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
