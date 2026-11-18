"""Webcam -> facial expression -> /emotion/label, with a live OpenCV window.

Threads:
* main thread: camera capture, drawing and keyboard (OpenCV GUI needs it);
* inference thread: runs the backend on the newest frame and drops stale ones,
  so a slow model lowers the inference rate, not the video frame rate;
* executor thread: ROS callbacks and the 10 Hz label publisher, which doubles
  as the heartbeat for the motion node's watchdog.

Topics:
    pub /emotion/label        std_msgs/String    stable label (happy ... fear | none)
    pub /emotion/scores       std_msgs/String    JSON with raw + smoothed scores
    pub /emotion/estop        std_msgs/Bool      toggled with SPACE in the window
    pub /emotion/debug_image  sensor_msgs/Image  rendered overlay (publish_image:=true)
    sub /emotion/action       std_msgs/String    JSON state from emotion_motion_node
    sub /emotion/override     std_msgs/String    emotion label = manual override, "auto" = clear
    sub /emotion/demo_step    std_msgs/String    JSON from emotion_sequence; prompt mode shows
                                                 "make this expression" on the overlay
"""

from __future__ import annotations

import json
import threading
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .backends import create_backend, largest_face, resolve_backend_name
from .calibration import apply_gains, load_calibration
from .emotions import EMOTIONS, NO_FACE
from .overlay import OverlayState, blur_outside, render
from .ros_utils import ignore_further_interrupts, init_ros
from .smoother import EmotionSmoother, SmootherConfig

KEY_TO_EMOTION = {ord(str(i + 1)): e for i, e in enumerate(EMOTIONS)}
WINDOW = "HRI week 8 - emotion -> TurtleBot"
DEFAULT_CALIBRATION = "~/.ros/tb4_emotion_calibration.yaml"


class RateMeter:
    """Exponentially smoothed event rate in Hz."""

    def __init__(self, alpha: float = 0.1):
        self.alpha, self.rate, self._last = alpha, 0.0, None

    def tick(self, now: float) -> None:
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.rate = inst if self.rate == 0.0 else (1 - self.alpha) * self.rate + self.alpha * inst
        self._last = now


def to_image_msg(bgr: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(bgr).tobytes()
    return msg


class EmotionDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("emotion_detector")
        self._read_parameters()

        self.label_pub = self.create_publisher(String, "/emotion/label", 10)
        self.scores_pub = self.create_publisher(String, "/emotion/scores", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.estop_pub = self.create_publisher(Bool, "/emotion/estop", latched)
        self.image_pub = (
            self.create_publisher(Image, "/emotion/debug_image", 1) if self.publish_image else None
        )
        self.create_subscription(String, "/emotion/action", self._on_action, 10)
        self.create_subscription(String, "/emotion/override", self._on_override, 10)
        self.create_subscription(String, "/emotion/demo_step", self._on_demo_step, 10)
        self.create_timer(1.0 / self.publish_rate, self._publish_label)

        self.lock = threading.Lock()
        self.overlay = OverlayState(backend=self.backend_name)
        self.manual: Optional[str] = self.start_manual if self.start_manual in EMOTIONS else None
        self.estop = False
        self.running = True
        self._frame = None
        self._frame_id = 0
        self._frame_ready = threading.Condition(self.lock)
        self._last_image_pub = 0.0
        self.fps = RateMeter()
        self.infer_rate = RateMeter()

        self.get_logger().info(f"loading backend '{self.backend_name}' ...")
        self.backend = create_backend(
            self.backend_name,
            detector_backend=self.detector_backend,
            min_face_confidence=self.min_face_confidence,
            use_mtcnn=self.use_mtcnn,
            model_path=self.model_path,
        )
        t0 = time.monotonic()
        self.backend.warmup()
        self.get_logger().info(f"backend ready ({time.monotonic() - t0:.1f} s warm-up)")
        self.gains = self._load_gains()
        self.estop_pub.publish(Bool(data=False))

    def _read_parameters(self) -> None:
        p = self.declare_parameter
        self.backend_name = resolve_backend_name(p("backend", "auto").value)
        self.camera_source = str(p("camera_source", "0").value)
        self.frame_width = int(p("frame_width", 640).value)
        self.frame_height = int(p("frame_height", 480).value)
        self.mirror = bool(p("mirror", True).value)
        self.show_window = bool(p("show_window", True).value)
        self.publish_image = bool(p("publish_image", False).value)
        self.image_rate = float(p("image_rate", 15.0).value)
        # blur bystanders in the published/recorded image (the local window stays sharp)
        self.privacy_blur = bool(p("privacy_blur", True).value)
        self.detector_backend = p("detector_backend", "opencv").value
        self.min_face_confidence = float(p("min_face_confidence", 0.5).value)
        self.use_mtcnn = bool(p("use_mtcnn", False).value)
        self.model_path = str(p("model_path", "").value)
        self.calibration_file = str(p("calibration_file", DEFAULT_CALIBRATION).value)
        self.publish_rate = float(p("publish_rate", 10.0).value)
        # start with a manual override (e.g. "neutral" for scripted recordings)
        self.start_manual = str(p("start_manual", "").value).strip().lower() or None
        self.smoother = EmotionSmoother(
            SmootherConfig(
                alpha=float(p("ema_alpha", 0.35).value),
                min_confidence=float(p("min_confidence", 0.40).value),
                hold_time=float(p("hold_time", 0.6).value),
                face_lost_timeout=float(p("face_lost_timeout", 1.0).value),
            )
        )

    def _load_gains(self) -> Optional[dict]:
        """Per-user class gains from emotion_calibrate, if they match the backend."""
        if not self.calibration_file:
            return None
        try:
            gains, data = load_calibration(self.calibration_file)
        except Exception as exc:  # a corrupt file must not stop the demo
            self.get_logger().error(f"ignoring calibration {self.calibration_file}: {exc}")
            return None
        if not data:
            self.get_logger().info(
                "no calibration file; run 'ros2 run tb4_emotion_control emotion_calibrate' "
                "if some expressions are hard to trigger"
            )
            return None
        if data.get("backend") != self.backend_name:
            self.get_logger().warn(
                f"calibration was made for backend '{data.get('backend')}', "
                f"not '{self.backend_name}'; ignoring it"
            )
            return None
        if all(abs(g - 1.0) < 1e-9 for g in gains.values()):
            self.get_logger().info("calibration file has unit gains; nothing to apply")
            return None
        self.get_logger().info("calibration gains: " + ", ".join(f"{k}={v:.2f}" for k, v in gains.items()))
        self.overlay.backend = f"{self.backend_name} (calibrated)"
        return gains

    # ------------------------------------------------------------- ROS side
    def _on_action(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.overlay.action = data.get("action", "")
            self.overlay.linear = float(data.get("linear", 0.0))
            self.overlay.angular = float(data.get("angular", 0.0))
            self.overlay.progress = float(data.get("progress", 0.0))

    def _on_override(self, msg: String) -> None:
        label = msg.data.strip().lower()
        if label in EMOTIONS:
            self.set_manual(label)
        elif label in ("auto", ""):
            self.set_manual(None)
        else:
            self.get_logger().warn(f"ignoring override '{msg.data}'", throttle_duration_sec=5.0)

    def _on_demo_step(self, msg: String) -> None:
        try:
            step = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        prompt = step.get("label", "") if step.get("mode") == "prompt" and not step.get("done") else ""
        with self.lock:
            self.overlay.prompt = prompt

    def set_manual(self, label: Optional[str]) -> None:
        with self.lock:
            changed = label != self.manual
            self.manual = label
        if changed:
            self.get_logger().info(f"manual override: {label}" if label else "back to automatic detection")

    def current_label(self) -> str:
        with self.lock:
            return self.manual or self.smoother.stable

    def _publish_label(self) -> None:
        self.label_pub.publish(String(data=self.current_label()))

    def set_estop(self, engaged: bool) -> None:
        self.estop = engaged
        self.estop_pub.publish(Bool(data=engaged))
        self.get_logger().warn("E-STOP engaged" if engaged else "E-STOP released")

    # ------------------------------------------------------------ inference
    def inference_loop(self) -> None:
        last_id = 0  # frame ids start at 1
        last_label = None
        while self.running:
            with self._frame_ready:
                self._frame_ready.wait_for(lambda: self._frame_id != last_id or not self.running, 0.5)
                if not self.running:
                    return
                if self._frame_id == last_id:
                    continue
                frame, last_id = self._frame, self._frame_id
            try:
                face = largest_face(self.backend.analyze(frame))
                if face is not None and self.gains:
                    face.scores = apply_gains(face.scores, self.gains)
            except Exception as exc:  # keep the UI alive on a bad frame
                self.get_logger().error(f"inference failed: {exc}", throttle_duration_sec=5.0)
                face = None
            now = time.monotonic()
            self.infer_rate.tick(now)
            scores = face.scores if face else {}
            with self.lock:
                stable = self.smoother.update(scores or None, now)
                self.overlay.box = face.box if face else None
                self.overlay.raw = scores
                self.overlay.smooth = self.smoother.ema
                self.overlay.stable = stable
                self.overlay.infer_hz = self.infer_rate.rate
                payload = {
                    "face": face is not None,
                    "stable": stable,
                    "raw": {k: round(v, 3) for k, v in scores.items()},
                    "smooth": {k: round(v, 3) for k, v in self.smoother.ema.items()},
                }
            if not self.running:
                return
            try:
                self.scores_pub.publish(String(data=json.dumps(payload)))
            except Exception:  # context already shut down while stopping
                return
            if stable != last_label:
                self.get_logger().info(f"emotion: {stable}")
                last_label = stable

    # --------------------------------------------------------- camera / GUI
    def open_camera(self) -> cv2.VideoCapture:
        src = int(self.camera_source) if self.camera_source.isdigit() else self.camera_source
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera source '{self.camera_source}'")
        return cap

    def camera_loop(self) -> None:
        cap = self.open_camera()
        is_file = not self.camera_source.isdigit()
        last_status = time.monotonic()
        if self.show_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        try:
            while self.running and rclpy.ok():
                ok, frame = cap.read()
                if not ok:
                    if is_file:  # loop recorded videos for repeatable demos
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.get_logger().error("camera read failed", throttle_duration_sec=2.0)
                    time.sleep(0.05)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.monotonic()
                self.fps.tick(now)
                with self._frame_ready:
                    self._frame = frame
                    self._frame_id += 1
                    self._frame_ready.notify()

                if self.publish_image and now - self._last_image_pub >= 1.0 / self.image_rate:
                    self._last_image_pub = now
                    shown = blur_outside(frame, self.overlay.box) if self.privacy_blur else frame
                    self.image_pub.publish(to_image_msg(self._render(shown), self.get_clock().now().to_msg()))
                if self.show_window:
                    view = self._render(frame)
                    cv2.imshow(WINDOW, view)
                    self.handle_key(cv2.waitKey(1) & 0xFF)
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        self.running = False
                else:
                    if now - last_status >= 5.0:
                        last_status = now
                        self.get_logger().info(
                            f"camera {self.fps.rate:.1f} FPS, inference {self.infer_rate.rate:.1f} Hz, "
                            f"label '{self.current_label()}'"
                        )
                    if is_file:
                        time.sleep(1.0 / 30.0)
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()

    def _render(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            self.overlay.fps = self.fps.rate
            self.overlay.manual = self.manual
            self.overlay.estop = self.estop
            return render(frame, self.overlay)

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord(" "):
            self.set_estop(not self.estop)
        elif key in KEY_TO_EMOTION:
            self.set_manual(KEY_TO_EMOTION[key])
        elif key in (ord("a"), ord("0")):
            self.set_manual(None)

    def stop(self) -> None:
        self.running = False
        with self._frame_ready:
            self._frame_ready.notify_all()


def main(args=None) -> None:
    init_ros(args)  # worker threads must stop before the ROS context goes away
    node = EmotionDetectorNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_executor() -> None:
        try:
            executor.spin()
        except ExternalShutdownException:
            node.running = False

    spin_thread = threading.Thread(target=spin_executor, daemon=True)
    worker = threading.Thread(target=node.inference_loop, daemon=True)
    spin_thread.start()
    worker.start()
    try:
        node.camera_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ignore_further_interrupts()
        node.stop()
        if rclpy.ok():  # tell the motion node to stop; its watchdog is the backup
            node.label_pub.publish(String(data=NO_FACE))
            time.sleep(0.1)
        executor.shutdown()
        worker.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
