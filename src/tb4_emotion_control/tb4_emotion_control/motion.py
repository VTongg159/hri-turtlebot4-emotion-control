"""Emotion -> motion primitive mapping and the execution state machine.

Each emotion owns one primitive, a constant (v, w) command:

    happy     arc to the left   v > 0, w > 0  (continuous while held)
    sad       arc to the right  v > 0, w < 0  (continuous while held)
    neutral   stop              v = 0, w = 0
    angry     short reverse     v < 0         (one-shot, ~0.15 m)
    surprise  one full turn     w != 0        (one-shot, 2*pi rad)
    fear      short advance     v > 0         (one-shot, ~0.15 m)

Two primitive kinds:

* continuous: follows the label. Robot keeps arcing while the face shows the
  expression and switches as soon as the label changes.
* one-shot: atomic. Once triggered it runs to its odometry target (or its time
  budget if odometry is unavailable). Only the e-stop or the input watchdog
  can interrupt it. Facial expressions such as surprise last well under a
  second, so letting the next frame cancel the spin would mean it never
  finishes. After a one-shot the same label must disappear and come back
  before the primitive runs again (edge-triggered).

The class is ROS-free so it can be unit-tested; the node supplies time,
odometry progress and the current label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from .emotions import EMOTIONS, NO_FACE


@dataclass(frozen=True)
class Primitive:
    name: str  # action name shown in the UI and published on /emotion/action
    linear: float  # m/s
    angular: float  # rad/s
    one_shot: bool = False
    target_distance: Optional[float] = None  # m of travel (one-shot linear moves)
    target_angle: Optional[float] = None  # rad of accumulated yaw (one-shot turns)
    duration: Optional[float] = None  # s; open-loop budget when odometry is absent

    @property
    def is_stop(self) -> bool:
        return self.linear == 0.0 and self.angular == 0.0

    def time_budget(self) -> Optional[float]:
        """Open-loop duration that reaches the target at the commanded speed."""
        if self.duration is not None:
            return self.duration
        if self.target_distance is not None and self.linear != 0.0:
            return abs(self.target_distance / self.linear)
        if self.target_angle is not None and self.angular != 0.0:
            return abs(self.target_angle / self.angular)
        return None


STOP = Primitive("stop", 0.0, 0.0)


def default_primitives(
    arc_linear: float = 0.15,
    arc_angular: float = 0.5,
    nudge_speed: float = 0.10,
    nudge_distance: float = 0.15,
    spin_speed: float = 1.0,
) -> dict[str, Primitive]:
    """Build the emotion table. Speeds stay well under TurtleBot 4 limits (0.31 m/s, 1.9 rad/s)."""
    if arc_linear <= 0 or arc_angular <= 0 or nudge_speed <= 0 or spin_speed <= 0:
        raise ValueError("speeds must be positive; directions are fixed by the mapping")
    return {
        "happy": Primitive("arc_left", arc_linear, +arc_angular),
        "sad": Primitive("arc_right", arc_linear, -arc_angular),
        "neutral": replace(STOP, name="stop"),
        "angry": Primitive(
            "back_off", -nudge_speed, 0.0, one_shot=True, target_distance=nudge_distance
        ),
        "surprise": Primitive(
            "spin_360", 0.0, spin_speed, one_shot=True, target_angle=2.0 * math.pi
        ),
        "fear": Primitive(
            "nudge_forward", nudge_speed, 0.0, one_shot=True, target_distance=nudge_distance
        ),
    }


@dataclass
class Command:
    linear: float
    angular: float
    action: str  # primitive name, or "stop", "estop", "watchdog_stop"
    progress: float = 0.0  # 0..1 for one-shots, 0 otherwise


class MotionController:
    """Choose the velocity command from the stable label, time and odometry."""

    def __init__(
        self,
        primitives: Mapping[str, Primitive] | None = None,
        input_timeout: float = 1.0,
        one_shot_timeout_factor: float = 2.5,
    ):
        self.primitives = dict(primitives or default_primitives())
        missing = set(EMOTIONS) - set(self.primitives)
        if missing:
            raise ValueError(f"missing primitives for: {sorted(missing)}")
        self.input_timeout = input_timeout
        self.one_shot_timeout_factor = one_shot_timeout_factor

        self.label: str = NO_FACE
        self.last_input_time: Optional[float] = None
        self.estop: bool = False

        self.active: Optional[Primitive] = None  # running one-shot
        self.active_label: Optional[str] = None
        self.active_start: float = 0.0
        self.consumed_label: Optional[str] = None  # one-shot waiting for re-arm
        self.rearmed = False

    # ----------------------------------------------------------------- inputs
    def set_label(self, label: str, now: float) -> None:
        self.label = label
        self.last_input_time = now
        if label != self.consumed_label:
            self.consumed_label = None
        if self.active is not None and label != self.active_label:
            self.rearmed = True  # expression went away during the one-shot

    def set_estop(self, engaged: bool) -> None:
        self.estop = engaged
        if engaged:
            self._finish_one_shot(consume=False)

    # ----------------------------------------------------------------- output
    def step(
        self,
        now: float,
        distance: Optional[float] = None,
        angle: Optional[float] = None,
    ) -> Command:
        """Advance the state machine.

        ``distance``/``angle`` are the absolute travel and accumulated |yaw|
        since the current one-shot started, measured from odometry by the node
        (None when odometry is not available). The node resets its odometry
        baseline whenever ``started_at`` changes, i.e. a new one-shot began.
        """
        if self.estop:
            return Command(0.0, 0.0, "estop")
        if self.last_input_time is None or now - self.last_input_time > self.input_timeout:
            self._finish_one_shot(consume=False)
            return Command(0.0, 0.0, "watchdog_stop")

        if self.active is not None:
            done, progress = self._one_shot_progress(now, distance, angle)
            if not done:
                p = self.active
                return Command(p.linear, p.angular, p.name, progress)
            self._finish_one_shot(consume=True)

        label = self.label
        if label == NO_FACE or label not in self.primitives:
            return Command(0.0, 0.0, "stop")
        prim = self.primitives[label]
        if prim.one_shot:
            if self.consumed_label == label:
                return Command(0.0, 0.0, "stop")
            self.active, self.active_label, self.active_start = prim, label, now
            self.rearmed = False
            return Command(prim.linear, prim.angular, prim.name, 0.0)
        return Command(prim.linear, prim.angular, prim.name)

    @property
    def started_at(self) -> Optional[float]:
        return self.active_start if self.active is not None else None

    # ---------------------------------------------------------------- helpers
    def _one_shot_progress(
        self, now: float, distance: Optional[float], angle: Optional[float]
    ) -> tuple[bool, float]:
        p = self.active
        assert p is not None
        elapsed = now - self.active_start
        budget = p.time_budget() or 0.0

        if p.target_distance is not None and distance is not None:
            progress = distance / p.target_distance
        elif p.target_angle is not None and angle is not None:
            progress = angle / p.target_angle
        else:
            progress = elapsed / budget if budget > 0 else 1.0

        # Safety: never run a closed-loop move far beyond its open-loop budget
        # (wheel slip, blocked robot, stale odometry).
        timed_out = budget > 0 and elapsed >= budget * self.one_shot_timeout_factor
        done = progress >= 1.0 or timed_out
        return done, min(max(progress, 0.0), 1.0)

    def _finish_one_shot(self, consume: bool) -> None:
        if self.active is not None and consume and not self.rearmed and self.label == self.active_label:
            self.consumed_label = self.active_label
        self.active = None
        self.active_label = None
