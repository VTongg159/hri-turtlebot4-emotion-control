"""Drive a timed sequence of expressions for tests and demo recordings.

    ros2 run tb4_emotion_control emotion_sequence --ros-args \
        -p sequence:="happy:6,neutral:2,sad:6,neutral:2,angry:3,fear:3,surprise:8,neutral:2"

Each item is ``label:seconds``. Modes:

* ``mode:=label`` (default): publish the label on ``topic`` at 10 Hz.
  - ``topic:=/emotion/label`` feeds emotion_motion_node directly (no camera);
  - ``topic:=/emotion/override`` feeds the running emotion_detector_node as a
    manual override, so the whole detector -> motion -> robot chain runs.
* ``mode:=prompt``: publish no labels at all. The step only tells the operator
  which expression to make (shown by demo_recorder); the robot reacts to the
  real recognition result. Used for the live demo recording.

Every tick also publishes ``/emotion/demo_step`` (JSON: index, total, label,
elapsed, duration, mode, done), which demo_recorder uses for captions and
for taking exactly one snapshot per step.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .emotions import NO_FACE, VALID_LABELS
from .ros_utils import init_ros

DEFAULT_SEQUENCE = (
    "neutral:2,happy:8,neutral:2,sad:8,neutral:2,angry:3,neutral:1,"
    "fear:3,neutral:1,surprise:9,neutral:2"
)
MODES = ("label", "prompt")


def parse_sequence(text: str) -> list[tuple[str, float]]:
    steps = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, seconds = item.partition(":")
        label = label.strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"unknown label '{label}'")
        duration = float(seconds) if seconds else 3.0
        if duration <= 0:
            raise ValueError(f"duration must be positive in '{item}'")
        steps.append((label, duration))
    if not steps:
        raise ValueError("empty sequence")
    return steps


class EmotionSequence(Node):
    def __init__(self) -> None:
        super().__init__("emotion_sequence")
        p = self.declare_parameter
        self.steps = parse_sequence(p("sequence", DEFAULT_SEQUENCE).value)
        self.loop = bool(p("loop", False).value)
        self.mode = p("mode", "label").value
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.topic = p("topic", "/emotion/label").value
        self.label_pub = self.create_publisher(String, self.topic, 10) if self.mode == "label" else None
        self.step_pub = self.create_publisher(String, "/emotion/demo_step", 10)
        # After the last step: hold "neutral" as an override (robot stays still), or "none".
        self.final_label = "neutral" if self.topic.endswith("override") else NO_FACE
        self.index = 0
        self.step_start = None  # set on the first tick with a valid clock
        self.done = False
        self.get_logger().info(f"{self.mode} mode, sequence: {self.steps}")
        self.create_timer(0.1, self._tick)

    def _announce(self) -> None:
        label, duration = self.steps[self.index]
        self.get_logger().info(f"step {self.index + 1}/{len(self.steps)}: {label} for {duration:.1f} s")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.step_start).nanoseconds * 1e-9

    def _publish_step(self, label: str, elapsed: float, duration: float) -> None:
        info = {
            "index": self.index + 1,
            "total": len(self.steps),
            "label": label,
            "elapsed": round(elapsed, 2),
            "duration": duration,
            "mode": self.mode,
            "done": self.done,
        }
        self.step_pub.publish(String(data=json.dumps(info)))

    def _tick(self) -> None:
        if self.step_start is None:
            # With use_sim_time the clock reads 0 until /clock arrives; starting
            # earlier would make the first step end immediately.
            if self.get_clock().now().nanoseconds == 0:
                return
            self.step_start = self.get_clock().now()
            self._announce()
        if self.done:
            if self.label_pub:
                self.label_pub.publish(String(data=self.final_label))
            self._publish_step("", 0.0, 0.0)
            return
        if self._elapsed() >= self.steps[self.index][1]:
            self.index += 1
            if self.index >= len(self.steps):
                if not self.loop:
                    self.done = True
                    self.index = len(self.steps) - 1
                    self.get_logger().info("sequence finished")
                    return
                self.index = 0
            self.step_start = self.get_clock().now()
            self._announce()
        label, duration = self.steps[self.index]
        if self.label_pub:
            self.label_pub.publish(String(data=label))
        self._publish_step(label, self._elapsed(), duration)


def main(args=None) -> None:
    init_ros(args)
    node = EmotionSequence()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):  # publish the final label briefly so the robot stops
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
