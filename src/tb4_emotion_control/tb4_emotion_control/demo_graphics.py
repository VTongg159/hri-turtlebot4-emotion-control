"""Shared look for the demo video and report figures (Vietnamese text via PIL).

OpenCV's Hershey fonts cannot draw Vietnamese diacritics, so every caption in
the recorded video and the generated figures is drawn with PIL using DejaVu
Sans (it covers Vietnamese and arrows such as "→"; Noto Sans lacks the arrows).
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .emotions import COLORS

FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/noto")
FONT_FILES = {
    "regular": ("DejaVuSans.ttf", "NotoSans-Regular.ttf"),
    "bold": ("DejaVuSans-Bold.ttf", "NotoSans-Bold.ttf"),
    "mono": ("DejaVuSansMono.ttf", "NotoSansMono-Regular.ttf"),
}

VI_EMOTION = {
    "happy": "Vui vẻ",
    "sad": "Buồn",
    "neutral": "Bình thường",
    "angry": "Tức giận",
    "surprise": "Ngạc nhiên",
    "fear": "Sợ hãi",
    "none": "Không thấy mặt",
}
VI_ACTION = {
    "arc_left": "Cong sang trái",
    "arc_right": "Cong sang phải",
    "stop": "Dừng tại chỗ",
    "back_off": "Lùi nhẹ 15 cm",
    "spin_360": "Xoay một vòng",
    "nudge_forward": "Tiến nhẹ 15 cm",
    "estop": "Dừng khẩn cấp",
    "watchdog_stop": "Watchdog: dừng",
    "": "—",
}
# How to make each expression (shown with live prompts and during calibration).
VI_HINT = {
    "happy": "cười tươi, khóe miệng kéo lên, mắt híp lại",
    "sad": "trề môi xuống, khóe miệng cụp, nhíu giữa lông mày lên",
    "neutral": "mặt thả lỏng tự nhiên",
    "angry": "chau mày thật mạnh, mím chặt môi, KHÔNG nhe răng",
    "surprise": "mở to mắt, nhướng mày, há miệng tròn",
    "fear": "mở to mắt, nhướng mày, kéo miệng sang ngang (không há tròn)",
}
EMOTION_ACTION = {
    "happy": "arc_left",
    "sad": "arc_right",
    "neutral": "stop",
    "angry": "back_off",
    "surprise": "spin_360",
    "fear": "nudge_forward",
}
# trail colours per action (RGB), same palette as scripts/plot_trajectory.py
ACTION_RGB = {
    "arc_left": (230, 161, 0),
    "arc_right": (42, 111, 219),
    "back_off": (214, 39, 40),
    "nudge_forward": (44, 160, 44),
    "spin_360": (176, 79, 199),
    "stop": (158, 158, 158),
}

# Fixed top-down demo camera, must match worlds/emotion_arena.sdf (demo_camera_top).
TOP_CAM = {"x": -0.25, "y": 0.45, "height": 3.4, "fov": 0.62}


# Trails are drawn at roughly the height of the TurtleBot's top plate, which is what the
# top camera actually sees: projecting them on the floor (z = 0) would show the robot
# ~10 cm off its own path near the image edges (parallax).
ROBOT_VIEW_Z = 0.25


def top_m_per_px(size: int, z: float = 0.0, cam: dict = TOP_CAM) -> float:
    """Metres per pixel of the top camera for a horizontal plane at height ``z``."""
    return 2.0 * (cam["height"] - z) * math.tan(cam["fov"] / 2.0) / size


def top_project(x: float, y: float, size: int, z: float = ROBOT_VIEW_Z, cam: dict = TOP_CAM) -> tuple[float, float]:
    """World point (x, y, z) -> pixel (u, v) in the square top-camera image of ``size`` px."""
    m_per_px = top_m_per_px(size, z, cam)
    return size / 2 + (x - cam["x"]) / m_per_px, size / 2 - (y - cam["y"]) / m_per_px


BG = (24, 26, 30)
PANEL = (36, 39, 45)
PANEL_HI = (52, 56, 64)
TEXT = (235, 237, 240)
MUTED = (160, 166, 175)
ACCENT = (90, 200, 120)


def emotion_rgb(label: str) -> tuple[int, int, int]:
    b, g, r = COLORS.get(label, (255, 255, 255))
    return (r, g, b)


@lru_cache(maxsize=64)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    for name in FONT_FILES[weight]:
        for d in FONT_DIRS:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1])


def to_bgr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))[:, :, ::-1].copy()


def canvas(w: int, h: int, color=BG) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def fit_np(src_bgr: Optional[np.ndarray], w: int, h: int, placeholder: str = "", bg=PANEL) -> np.ndarray:
    """Letterbox a BGR image into a w x h BGR tile (fast path: OpenCV resize)."""
    import cv2  # noqa: PLC0415

    if src_bgr is None:
        tile = canvas(w, h, bg)
        ImageDraw.Draw(tile).text((w // 2, h // 2), placeholder, font=font(22), fill=MUTED, anchor="mm")
        return to_bgr(tile)
    tile = np.empty((h, w, 3), np.uint8)
    tile[:] = bg[::-1]
    s = min(w / src_bgr.shape[1], h / src_bgr.shape[0])
    rw, rh = max(1, int(src_bgr.shape[1] * s)), max(1, int(src_bgr.shape[0] * s))
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    tile[y0:y0 + rh, x0:x0 + rw] = cv2.resize(src_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    return tile


def rounded(d: ImageDraw.ImageDraw, box, fill, outline=None, width=2, radius=10) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def chip_size(d: ImageDraw.ImageDraw, label: str, size=22, pad=(14, 6)) -> tuple[int, int]:
    l, t, r, b = d.textbbox((0, 0), label, font=font(size, "bold"))
    return r - l + 2 * pad[0], b - t + 2 * pad[1]


def chip(d: ImageDraw.ImageDraw, xy, label: str, fill, fg=(20, 20, 20), size=22, pad=(14, 6)) -> int:
    """Pill-shaped label at ``xy`` (top-left); returns its width."""
    f = font(size, "bold")
    l, t, _, _ = d.textbbox((0, 0), label, font=f)
    w, h = chip_size(d, label, size, pad)
    x, y = xy
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=fill)
    d.text((x + pad[0] - l, y + pad[1] - t), label, font=f, fill=fg)
    return w


def table_rows(d: ImageDraw.ImageDraw, x: int, y: int, w: int, rows: Iterable[tuple[str, str]],
               active: Optional[str], row_h: int = 54, size: int = 26) -> int:
    """Draw ``(emotion_label, right_text)`` rows, highlighting ``active``; returns the next y."""
    for label, right in rows:
        is_active = label == active
        weight = "bold" if is_active else "regular"
        cy = y + (row_h - 8) // 2
        if is_active:
            rounded(d, (x, y, x + w, y + row_h - 8), fill=PANEL_HI, outline=emotion_rgb(label), width=3)
        d.ellipse((x + 16, cy - 9, x + 34, cy + 9), fill=emotion_rgb(label))
        d.text((x + 50, cy), VI_EMOTION[label], font=font(size, weight),
               fill=emotion_rgb(label) if is_active else TEXT, anchor="lm")
        d.text((x + int(w * 0.46), cy), "→  " + right, font=font(size, weight),
               fill=TEXT if is_active else MUTED, anchor="lm")
        y += row_h
    return y
