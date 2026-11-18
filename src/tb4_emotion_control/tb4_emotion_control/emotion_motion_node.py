"""/emotion/label -> TurtleBot motion primitive -> /cmd_vel at a fixed rate.

Topics:
    sub /emotion/label   std_msgs/String     stable label from the detector
    sub /emotion/estop   std_msgs/Bool       latched e-stop
    sub /odom            nav_msgs/Odometry   closes the loop for one-shot moves
    pub /cmd_vel         geometry_msgs/Twist (or TwistStamped, see parameter)
    pub /emotion/action  std_msgs/String     JSON {action, linear, angular, progress, label}

The TurtleBot 4 base drops velocity commands older than 0.5 s, so the command
is republished at ``control_rate`` (20 Hz) even when it does not change.
All timing uses the node clock, so with ``use_sim_time`` the manoeuvres are
defined in simulation seconds and do not depend on the host's load.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import Bool, String

from .emotions import VALID_LABELS
from .motion import MotionController, default_primitives
from .odometry import ProgressTracker, yaw_from_quaternion
from .ros_utils import ignore_further_interrupts, init_ros


class EmotionMotionNode(Node):
    def __init__(self) -> None:
        super().__init__("emotion_motion")
        p = self.declare_parameter
        primitives = default_primitives(
            arc_linear=float(p("arc_linear", 0.15).value),
            arc_angular=float(p("arc_angular", 0.5).value),
            nudge_speed=float(p("nudge_speed", 0.10).value),
            nudge_distance=float(p("nudge_distance", 0.15).value),
            spin_speed=float(p("spin_speed", 1.0).value),
        )
        self.controller = MotionController(
            primitives,
            input_timeout=float(p("input_timeout", 1.0).value),
            one_shot_timeout_factor=float(p("one_shot_timeout_factor", 2.5).value),
        )
        self.use_odom = bool(p("use_odom", True).value)
        self.odom_timeout = float(p("odom_timeout", 0.5).value)
        self.stamped = bool(p("cmd_vel_stamped", False).value)
        cmd_topic = p("cmd_vel_topic", "/cmd_vel").value
        rate = float(p("control_rate", 20.0).value)

        msg_type = TwistStamped if self.stamped else Twist
        self.cmd_pub = self.create_publisher(msg_type, cmd_topic, 10)
        self.action_pub = self.create_publisher(String, "/emotion/action", 10)
        self.create_subscription(String, "/emotion/label", self._on_label, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "/emotion/estop", self._on_estop, latched)
        if self.use_odom:
            self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)

        self.tracker = ProgressTracker()
        self.last_odom_time: float | None = None
        self.started_at: float | None = None
        self.last_action = None
        self.create_timer(1.0 / rate, self._step)
        self.get_logger().info(
            f"publishing {msg_type.__name__} on {cmd_topic} at {rate:.0f} Hz "
            f"(odometry feedback {'on' if self.use_odom else 'off'})"
        )

    def _now(self) -> float:
        """Node clock in seconds: simulation time when use_sim_time is set, else wall time."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- input
    def _on_label(self, msg: String) -> None:
        label = msg.data.strip().lower()
        if label not in VALID_LABELS:
            self.get_logger().warn(f"ignoring unknown label '{msg.data}'", throttle_duration_sec=5.0)
            return
        self.controller.set_label(label, self._now())

    def _on_estop(self, msg: Bool) -> None:
        self.controller.set_estop(msg.data)
        self.get_logger().warn("E-STOP engaged" if msg.data else "E-STOP released")

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.tracker.update(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        self.last_odom_time = self._now()

    # --------------------------------------------------------------- output
    def _odom_fresh(self, now: float) -> bool:
        return (
            self.use_odom
            and self.last_odom_time is not None
            and now - self.last_odom_time <= self.odom_timeout
            and self.tracker.started
        )

    def _step(self) -> None:
        now = self._now()
        fresh = self._odom_fresh(now)
        cmd = self.controller.step(
            now,
            distance=self.tracker.distance if fresh else None,
            angle=self.tracker.angle if fresh else None,
        )
        if self.controller.started_at != self.started_at:
            # A new one-shot began (or one ended): measure progress from here.
            self.started_at = self.controller.started_at
            self.tracker.reset()
        self.publish_cmd(cmd.linear, cmd.angular)
        self.action_pub.publish(
            String(
                data=json.dumps(
                    {
                        "action": cmd.action,
                        "linear": cmd.linear,
                        "angular": cmd.angular,
                        "progress": round(cmd.progress, 3),
                        "label": self.controller.label,
                        "odom": fresh,
                    }
                )
            )
        )
        if cmd.action != self.last_action:
            self.get_logger().info(
                f"[{self.controller.label}] -> {cmd.action} "
                f"(v={cmd.linear:+.2f} m/s, w={cmd.angular:+.2f} rad/s)"
            )
            self.last_action = cmd.action

    def publish_cmd(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist = twist
            self.cmd_pub.publish(msg)
        else:
            self.cmd_pub.publish(twist)


def main(args=None) -> None:
    init_ros(args)  # keeps the context alive long enough to send the final stop command
    node = EmotionMotionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ignore_further_interrupts()
        if rclpy.ok():
            for _ in range(3):  # make sure the robot stops on exit
                node.publish_cmd(0.0, 0.0)
            node.get_logger().info("sent zero velocity, shutting down")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
