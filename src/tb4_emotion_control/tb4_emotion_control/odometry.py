"""Measure travel distance and accumulated rotation from odometry poses."""

from __future__ import annotations

import math
from typing import Optional


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class ProgressTracker:
    """Straight-line distance and unwrapped |yaw| swept since ``reset``.

    Rotation is integrated from successive yaw differences, so a full 360 degree
    turn reads 2*pi instead of wrapping back to 0.
    """

    def __init__(self) -> None:
        self._start: Optional[tuple[float, float]] = None
        self._last_yaw: Optional[float] = None
        self._x = self._y = 0.0
        self.angle = 0.0

    def reset(self) -> None:
        self._start = None
        self._last_yaw = None
        self.angle = 0.0

    def update(self, x: float, y: float, yaw: float) -> None:
        self._x, self._y = x, y
        if self._start is None:
            self._start = (x, y)
            self._last_yaw = yaw
            return
        self.angle += abs(wrap_angle(yaw - self._last_yaw))
        self._last_yaw = yaw

    @property
    def started(self) -> bool:
        return self._start is not None

    @property
    def distance(self) -> float:
        if self._start is None:
            return 0.0
        return math.hypot(self._x - self._start[0], self._y - self._start[1])
