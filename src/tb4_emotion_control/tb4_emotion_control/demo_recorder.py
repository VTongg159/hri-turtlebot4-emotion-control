"""Record the demo video and report screenshots without screen capture.

Composes, at a fixed rate, one 1920x1080 frame from
* /emotion/debug_image   detector overlay (webcam, emotion bars, robot state)
* /demo_cam/perspective  fixed 3/4-view camera in the Gazebo world
* /demo_cam/top          fixed top-down camera; the odometry trail is drawn on it
* /emotion/action        active emotion -> action (highlighted in a table)
* /emotion/demo_step     current step of emotion_sequence (caption, snapshots)
* /odom                  pose for the trail and the readout

Output in ``output_dir``:
    demo_raw.avi                 MJPG, still readable if the process is killed
    snapshots/NN_<label>_*.png   one set per step: composite, overlay, gazebo, top
    trajectory_top.png           top view with the complete trail

    ros2 run tb4_emotion_control demo_recorder --ros-args -p output_dir:=evidence/demo
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import ImageDraw
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from . import demo_graphics as g
from .emotions import EMOTIONS
from .ros_utils import ignore_further_interrupts, init_ros

W, H = 1920, 1080
HEADER = 76
# seconds into a step at which its snapshot is taken (mid-manoeuvre)
CONTINUOUS = ("happy", "sad")  # labels whose manoeuvre lasts as long as the label
SNAPSHOT_DELAY = {"happy": 5.0, "sad": 5.0, "angry": 1.2, "fear": 1.2, "surprise": 3.2, "neutral": 1.0}


def image_to_bgr(msg: Image) -> Optional[np.ndarray]:
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}.get(msg.encoding)
    if channels is None:
        return None
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // channels, channels)
    img = img[:, : msg.width, :3]
    return img[:, :, ::-1].copy() if msg.encoding.startswith("rgb") else img.copy()


class DemoRecorder(Node):
    def __init__(self) -> None:
        super().__init__("demo_recorder")
        p = self.declare_parameter
        self.out_dir = Path(p("output_dir", "evidence/demo").value).expanduser()
        self.fps = float(p("fps", 15.0).value)

        self.snap_dir = self.out_dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.out_dir / "demo_raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (W, H)
        )
        if not self.writer.isOpened():
            raise RuntimeError("cannot open video writer")

        self.overlay = self.persp = self.top = None
        self.action, self.label = "", ""
        self.linear = self.angular = self.progress = 0.0
        self.pose = (0.0, 0.0, 0.0)
        self.trail: list[tuple[float, float, str]] = []
        self.step: dict = {}
        self.snapped_steps: set[int] = set()
        self.snapped_end: set[int] = set()
        self.snapped_neutral = False
        self.snapped_neutral_end = False
        self.prev_step: tuple[int, str] | None = None  # (index, label) of the previous step
        self.frames = 0
        self.t0: float | None = None  # first tick, node clock
        self.closed = False
        self._cache: dict = {}

        self.create_subscription(Image, "/emotion/debug_image", self._img_cb("overlay"), 1)
        self.create_subscription(Image, "/demo_cam/perspective", self._img_cb("persp"), qos_profile_sensor_data)
        self.create_subscription(Image, "/demo_cam/top", self._img_cb("top"), qos_profile_sensor_data)
        self.create_subscription(String, "/emotion/action", self._on_action, 10)
        self.create_subscription(String, "/emotion/demo_step", self._on_step, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, qos_profile_sensor_data)
        self.create_timer(1.0 / self.fps, self._tick)
        self.get_logger().info(f"recording {W}x{H} @ {self.fps:.0f} fps to {self.out_dir}")


    def _now(self) -> float:
        """Node clock in seconds: simulation time when use_sim_time is set, else wall time."""
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- input
    def _img_cb(self, attr: str):
        def cb(msg: Image) -> None:
            img = image_to_bgr(msg)
            if img is not None:
                setattr(self, attr, img)
        return cb

    def _on_action(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.action = d.get("action", "")
        self.label = d.get("label", "")
        self.linear, self.angular = float(d.get("linear", 0.0)), float(d.get("angular", 0.0))
        self.progress = float(d.get("progress", 0.0))

    def _on_step(self, msg: String) -> None:
        try:
            step = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if self.step and not self.step.get("done") and step.get("index") != self.step.get("index"):
            self.prev_step = (int(self.step["index"]), self.step.get("label", ""))
        self.step = step

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.pose = (x, y, yaw)
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.004:
            self.trail.append((x, y, self.action))

    # ------------------------------------------------------------- drawing
    @staticmethod
    def _to_top_px(x: float, y: float, size: int) -> tuple[int, int]:
        u, v = g.top_project(x, y, size)
        return int(round(u)), int(round(v))

    def top_with_trail(self) -> Optional[np.ndarray]:
        if self.top is None:
            return None
        img = self.top.copy()
        size = img.shape[1]
        k = size / 720.0  # line widths were tuned for a 720 px image
        for (x0, y0, a0), (x1, y1, _) in zip(self.trail, self.trail[1:]):
            rgb = g.ACTION_RGB.get(a0)
            if rgb is not None and a0 != "stop":
                cv2.line(img, self._to_top_px(x0, y0, size), self._to_top_px(x1, y1, size), rgb[::-1],
                         max(2, int(5 * k)), cv2.LINE_AA)
        x, y, yaw = self.pose  # heading arrow on the robot
        tail = self._to_top_px(x, y, size)
        tip = self._to_top_px(x + 0.28 * math.cos(yaw), y + 0.28 * math.sin(yaw), size)
        cv2.arrowedLine(img, tail, tip, (255, 255, 255), max(3, int(6 * k)), cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, tail, tip, (40, 40, 40), max(2, int(3 * k)), cv2.LINE_AA, tipLength=0.35)
        return img

    def _step_caption(self) -> tuple[str, str]:
        if not self.step or self.step.get("done"):
            return "", ""
        label = self.step.get("label", "")
        head = f"Bước {self.step['index']}/{self.step['total']}"
        if self.step.get("mode") == "prompt":
            return head, f"Hãy thể hiện: {g.VI_EMOTION.get(label, label)}"
        return head, f"{g.VI_EMOTION.get(label, label)}  →  {g.VI_ACTION[g.EMOTION_ACTION.get(label, '')]}"

    def _header(self) -> np.ndarray:
        head, caption = self._step_caption()
        key = ("header", head, caption, self.step.get("label", ""))
        if key not in self._cache:
            img = g.canvas(W, HEADER, g.PANEL)
            d = ImageDraw.Draw(img)
            d.text((28, HEADER // 2), "Bài thực hành 08 · Nhận diện biểu cảm khuôn mặt → điều khiển TurtleBot 4",
                   font=g.font(26, "bold"), fill=g.TEXT, anchor="lm")
            if caption:
                cw, ch = g.chip_size(d, caption, size=24)
                x = W - 28 - cw
                g.chip(d, (x, (HEADER - ch) // 2), caption, g.emotion_rgb(self.step.get("label", "")), size=24)
                d.text((x - 16, HEADER // 2), head, font=g.font(24, "bold"), fill=g.MUTED, anchor="rm")
            self._cache[key] = g.to_bgr(img)
        return self._cache[key]

    def _table(self, w: int, h: int) -> np.ndarray:
        active = self.label if self.label in EMOTIONS else None
        key = ("table", active)
        if key not in self._cache:
            img = g.canvas(w, h)
            d = ImageDraw.Draw(img)
            d.text((28, 34), "Biểu cảm  →  hành động của TurtleBot", font=g.font(28, "bold"), fill=g.TEXT, anchor="lm")
            g.table_rows(d, 20, 70, w - 40, [(e, g.VI_ACTION[g.EMOTION_ACTION[e]]) for e in EMOTIONS],
                         active=active, row_h=72, size=28)
            self._cache[key] = g.to_bgr(img)
        return self._cache[key]

    def _readout(self, w: int, h: int) -> np.ndarray:
        img = g.canvas(w, h)
        d = ImageDraw.Draw(img)
        px = 28
        d.text((px, 34), "Hành động", font=g.font(22), fill=g.MUTED, anchor="lm")
        d.text((px, 74), g.VI_ACTION.get(self.action, self.action), font=g.font(34, "bold"),
               fill=g.ACTION_RGB.get(self.action, g.TEXT), anchor="lm")
        mono = g.font(22, "mono")
        x, y, yaw = self.pose
        d.text((px, 130), f"v = {self.linear:+.2f} m/s  ω = {self.angular:+.2f} rad/s", font=mono, fill=g.TEXT, anchor="lm")
        d.text((px, 164), f"x = {x:+.2f} m  y = {y:+.2f} m", font=mono, fill=g.TEXT, anchor="lm")
        d.text((px, 198), f"θ = {math.degrees(yaw):+7.1f}°", font=mono, fill=g.TEXT, anchor="lm")
        bar_y = 234
        d.rounded_rectangle((px, bar_y, w - 28, bar_y + 12), radius=6, fill=g.PANEL_HI)
        if self.progress > 0.0:
            d.rounded_rectangle((px, bar_y, px + int((w - 56) * self.progress), bar_y + 12), radius=6, fill=g.ACCENT)
        d.text((px, bar_y + 46), "Vệt quỹ đạo (camera nhìn từ trên):", font=g.font(20), fill=g.MUTED, anchor="lm")
        for i, a in enumerate(("arc_left", "arc_right", "back_off", "nudge_forward", "spin_360")):
            cx, cy = px + (i % 2) * (w // 2 - 10), bar_y + 82 + (i // 2) * 34
            d.line((cx, cy, cx + 30, cy), fill=g.ACTION_RGB[a], width=6)
            d.text((cx + 42, cy), g.VI_ACTION[a], font=g.font(20), fill=g.TEXT, anchor="lm")
        return g.to_bgr(img)

    def compose(self) -> np.ndarray:
        frame = np.empty((H, W, 3), np.uint8)
        frame[:] = g.BG[::-1]
        frame[:HEADER] = self._header()
        # left column: recognition overlay + mapping table
        frame[HEADER:HEADER + 476, :960] = g.fit_np(self.overlay, 960, 476, "đang chờ /emotion/debug_image")
        frame[HEADER + 476:, :960] = self._table(960, H - HEADER - 476)
        # right column: Gazebo 3/4 view, top view with trail, readout
        frame[HEADER:HEADER + 540, 960:] = g.fit_np(self.persp, 960, 540, "đang chờ camera Gazebo")
        top = H - HEADER - 540
        frame[HEADER + 540:, 960:960 + top] = g.fit_np(self.top_with_trail(), top, top, "đang chờ camera trên")
        frame[HEADER + 540:, 960 + top:] = self._readout(W - 960 - top, top)
        frame[HEADER:, 959:961] = (82, 74, 70)
        frame[HEADER - 1:HEADER + 1, :] = (82, 74, 70)
        return frame

    # --------------------------------------------------------------- output
    def _tick(self) -> None:
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        frame = self.compose()
        # One video frame per 1/fps of node-clock time (simulation time with use_sim_time):
        # if composing fell behind, repeat the frame so the robot moves at its true speed.
        due = int((now - self.t0) * self.fps) + 1
        for _ in range(max(1, due - self.frames)):
            self.writer.write(frame)
            self.frames += 1
        self._maybe_snapshot(frame)

    def _maybe_snapshot(self, frame: np.ndarray) -> None:
        if not self.step or self.step.get("done"):
            return
        idx, label = int(self.step["index"]), self.step.get("label", "")
        self._maybe_end_snapshot(idx, label)
        # timing comes from the sequence's own "elapsed" (its clock), not from this node
        if idx in self.snapped_steps or (label == "neutral" and self.snapped_neutral):
            return
        elapsed = float(self.step.get("elapsed", 0.0))
        if self.step.get("mode") == "prompt":
            # live: wait until the recognised label matches the prompt (fallback: end of step)
            matched = self.label == label and elapsed >= 1.0
            if not matched and elapsed < float(self.step.get("duration", 0.0)) - 0.5:
                return
        elif elapsed < SNAPSHOT_DELAY.get(label, 1.5):
            return
        self.snapped_steps.add(idx)
        self.snapped_neutral |= label == "neutral"
        stem = self.snap_dir / f"{idx:02d}_{label}"
        cv2.imwrite(f"{stem}_composite.png", frame)
        for name, img in (("overlay", self.overlay), ("gazebo", self.persp), ("top", self.top_with_trail())):
            if img is not None:
                cv2.imwrite(f"{stem}_{name}.png", img)
        self.get_logger().info(f"snapshot {stem.name}")

    def _maybe_end_snapshot(self, idx: int, label: str) -> None:
        """Raw top + 3/4 views of the robot at the end of a manoeuvre (first neutral only).

        Arcs keep moving until their label changes, so they are captured 1 s into
        the following step, once the robot has settled at the end of its path.
        """
        elapsed = float(self.step.get("elapsed", 0.0))
        if self.prev_step and self.prev_step[1] in CONTINUOUS and self.prev_step[0] not in self.snapped_end \
                and idx == self.prev_step[0] + 1 and elapsed >= 1.0:
            self._save_end(*self.prev_step)
        if label in CONTINUOUS or idx in self.snapped_end or (label == "neutral" and self.snapped_neutral_end):
            return
        if elapsed >= float(self.step.get("duration", 0.0)) - 0.4:
            self.snapped_neutral_end |= label == "neutral"
            self._save_end(idx, label)

    def _save_end(self, idx: int, label: str) -> None:
        self.snapped_end.add(idx)
        stem = self.snap_dir / f"{idx:02d}_{label}"
        if self.top is not None:
            cv2.imwrite(f"{stem}_top_end.png", self.top)
        if self.persp is not None:
            cv2.imwrite(f"{stem}_gazebo_end.png", self.persp)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.writer.release()
        top = self.top_with_trail()
        if top is not None:
            cv2.imwrite(str(self.out_dir / "trajectory_top.png"), top)
        self.get_logger().info(f"wrote {self.frames} frames ({self.frames / self.fps:.1f} s)")


def main(args=None) -> None:
    init_ros(args)  # SIGINT/SIGTERM must still finalise the video file
    node = DemoRecorder()
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
