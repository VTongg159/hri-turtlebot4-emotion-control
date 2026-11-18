"""Draw the real-time camera view: face box, emotion bars and robot state.

The camera image keeps OpenCV drawing (fast); the side panel and the face
label use PIL so the Vietnamese text keeps its diacritics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import cv2
import numpy as np
from PIL import ImageDraw

from . import demo_graphics as g
from .emotions import EMOTIONS, NO_FACE

PANEL_WIDTH = 340
MIN_HEIGHT = 480


@dataclass
class OverlayState:
    box: Optional[tuple[int, int, int, int]] = None
    raw: Mapping[str, float] = field(default_factory=dict)
    smooth: Mapping[str, float] = field(default_factory=dict)
    stable: str = NO_FACE
    manual: Optional[str] = None
    action: str = ""
    linear: float = 0.0
    angular: float = 0.0
    progress: float = 0.0
    estop: bool = False
    fps: float = 0.0
    infer_hz: float = 0.0
    backend: str = ""
    prompt: str = ""  # expression the operator is asked to make (live demo)


def _camera_layer(frame: np.ndarray, s: OverlayState, label: str) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    if s.box is not None:
        x, y, bw, bh = s.box
        b, gr, r = g.emotion_rgb(label)[::-1]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (b, gr, r), 3)
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, f"{s.fps:4.1f} FPS | infer {s.infer_hz:4.1f} Hz | {s.backend}", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if s.estop:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    return img


def _face_label(d: ImageDraw.ImageDraw, s: OverlayState, w: int, h: int) -> None:
    if s.box is None:
        text = "Không thấy khuôn mặt"
        cw, ch = g.chip_size(d, text, size=20, pad=(12, 5))
        g.chip(d, ((w - cw) // 2, (h - ch) // 2), text, (60, 64, 72), fg=g.TEXT, size=20, pad=(12, 5))
        return
    if not s.raw:
        return
    top = max(EMOTIONS, key=lambda e: s.raw.get(e, 0.0))
    text = f"{g.VI_EMOTION[top]} {s.raw[top] * 100:.0f}%"
    x, y, _, _ = s.box
    _, h = g.chip_size(d, text, size=18, pad=(8, 3))
    g.chip(d, (x, max(28, y - h - 4)), text, g.emotion_rgb(top), size=18, pad=(8, 3))


def _panel(d: ImageDraw.ImageDraw, x0: int, height: int, s: OverlayState, label: str) -> None:
    px, w = x0 + 16, PANEL_WIDTH - 32
    d.rectangle((x0, 0, x0 + PANEL_WIDTH, height), fill=g.BG)
    d.text((px, 22), "Biểu cảm", font=g.font(17), fill=g.MUTED, anchor="lm")
    d.text((px, 54), g.VI_EMOTION.get(label, label), font=g.font(30, "bold"), fill=g.emotion_rgb(label), anchor="lm")
    y = 84
    if s.manual:
        g.chip(d, (px, y), "điều khiển tay / kịch bản", (255, 200, 60), size=14, pad=(8, 2))
        y += 26
    y += 8
    bar_x = px + 104
    bar_w = w - 104
    for e in EMOTIONS:
        raw, smooth = float(s.raw.get(e, 0.0)), float(s.smooth.get(e, 0.0))
        d.text((px, y + 7), g.VI_EMOTION[e], font=g.font(15, "bold" if e == label else "regular"),
               fill=g.emotion_rgb(e), anchor="lm")
        d.rounded_rectangle((bar_x, y, bar_x + bar_w, y + 14), radius=7, fill=g.PANEL_HI)
        if smooth > 0.01:
            d.rounded_rectangle((bar_x, y, bar_x + max(14, int(bar_w * smooth)), y + 14), radius=7,
                                fill=g.emotion_rgb(e))
        tick = bar_x + int(bar_w * raw)  # white tick = this frame, bar = smoothed
        d.line((tick, y - 3, tick, y + 17), fill=(255, 255, 255), width=2)
        y += 25
    y += 14
    d.text((px, y), "TurtleBot", font=g.font(17), fill=g.MUTED, anchor="lm")
    y += 30
    color = (255, 80, 80) if s.action in ("estop", "watchdog_stop") else g.ACTION_RGB.get(s.action, g.TEXT)
    d.text((px, y), g.VI_ACTION.get(s.action, s.action), font=g.font(22, "bold"), fill=color, anchor="lm")
    y += 28
    d.text((px, y), f"v = {s.linear:+.2f} m/s   ω = {s.angular:+.2f} rad/s", font=g.font(15, "mono"),
           fill=g.TEXT, anchor="lm")
    if s.progress > 0.0:
        y += 20
        d.rounded_rectangle((px, y, px + w, y + 8), radius=4, fill=g.PANEL_HI)
        d.rounded_rectangle((px, y, px + max(8, int(w * s.progress)), y + 8), radius=4, fill=g.ACCENT)
    if s.estop:
        g.chip(d, (px, height - 98), "E-STOP · SPACE để nhả", (255, 80, 80), size=15, pad=(8, 3))
    hy = height - 62
    for line in ("1 vui  2 buồn  3 bình thường  4 tức giận", "5 ngạc nhiên  6 sợ hãi  ·  A tự động",
                 "SPACE dừng khẩn cấp  ·  Q thoát"):
        d.text((px, hy), line, font=g.font(13), fill=g.MUTED, anchor="lm")
        hy += 19


def _prompt(d: ImageDraw.ImageDraw, w: int, h: int, s: OverlayState) -> None:
    if not s.prompt:
        return
    text = f"Hãy thể hiện: {g.VI_EMOTION.get(s.prompt, s.prompt)}"
    hint = g.VI_HINT.get(s.prompt, "")
    cw, ch = g.chip_size(d, text, size=26, pad=(18, 8))
    hw, hh = g.chip_size(d, hint, size=16, pad=(10, 4))
    g.chip(d, ((w - cw) // 2, h - ch - hh - 22), text, g.emotion_rgb(s.prompt), size=26, pad=(18, 8))
    if hint:
        g.chip(d, ((w - hw) // 2, h - hh - 12), hint, (30, 32, 36), fg=(255, 255, 255), size=16, pad=(10, 4))


def render(frame: np.ndarray, s: OverlayState) -> np.ndarray:
    h, w = frame.shape[:2]
    height = max(h, MIN_HEIGHT)
    label = s.manual or s.stable
    canvas = np.zeros((height, w + PANEL_WIDTH, 3), np.uint8)
    canvas[:] = g.BG[::-1]
    canvas[:h, :w] = _camera_layer(frame, s, label)
    img = g.to_pil(canvas)
    d = ImageDraw.Draw(img)
    _face_label(d, s, w, h)
    _prompt(d, w, h, s)
    _panel(d, w, height, s, label)
    return g.to_bgr(img)


def blur_outside(frame: np.ndarray, box: Optional[tuple[int, int, int, int]], margin: float = 0.12) -> np.ndarray:
    """Blur everything except the operator's face (privacy for bystanders in recordings)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    out = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (w, h), interpolation=cv2.INTER_LINEAR)
    if box is not None:
        x, y, bw, bh = box
        mx, my = int(bw * margin), int(bh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
        out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out
