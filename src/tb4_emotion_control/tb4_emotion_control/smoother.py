"""Temporal filtering that turns noisy per-frame predictions into a stable label.

Per-frame expression classifiers flicker between classes, and every flicker
would otherwise jerk the robot. The filter has three stages:

1. Exponential moving average (EMA) of the probability vectors.
2. Confidence gate: the EMA winner must reach ``min_confidence``.
3. Hold time: that winner must stay on top for ``hold_time`` seconds before it
   becomes the published label (hysteresis).

When the face disappears for longer than ``face_lost_timeout`` the stable label
becomes ``NO_FACE`` so the controller can stop the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .emotions import EMOTIONS, NO_FACE, dominant


@dataclass
class SmootherConfig:
    alpha: float = 0.35  # EMA weight of the newest frame (0 < alpha <= 1)
    min_confidence: float = 0.40  # EMA probability required to accept a label
    hold_time: float = 0.6  # seconds the candidate must stay on top
    face_lost_timeout: float = 1.0  # seconds without a face before NO_FACE


@dataclass
class SmootherState:
    stable: str = NO_FACE
    candidate: str = NO_FACE
    candidate_since: float = 0.0
    last_face_time: Optional[float] = None
    ema: dict = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})


class EmotionSmoother:
    def __init__(self, config: SmootherConfig | None = None):
        self.config = config or SmootherConfig()
        if not 0.0 < self.config.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.state = SmootherState()

    @property
    def stable(self) -> str:
        return self.state.stable

    @property
    def ema(self) -> dict:
        return dict(self.state.ema)

    def reset(self) -> None:
        self.state = SmootherState()

    def update(self, scores: Optional[Mapping[str, float]], now: float) -> str:
        """Feed one frame (``None`` = no face) and return the stable label."""
        cfg, st = self.config, self.state

        if scores is None:
            if st.last_face_time is None or now - st.last_face_time >= cfg.face_lost_timeout:
                st.stable = NO_FACE
                st.candidate = NO_FACE
                st.candidate_since = now
                st.ema = {e: 0.0 for e in EMOTIONS}
            return st.stable

        # A face that (re)appears after a gap starts a fresh average.
        first_face = (
            st.last_face_time is None or now - st.last_face_time >= cfg.face_lost_timeout
        )
        st.last_face_time = now
        if first_face:
            st.ema = {e: float(scores.get(e, 0.0)) for e in EMOTIONS}
        else:
            a = cfg.alpha
            st.ema = {
                e: (1.0 - a) * st.ema[e] + a * float(scores.get(e, 0.0)) for e in EMOTIONS
            }

        label, conf = dominant(st.ema)
        if conf < cfg.min_confidence:
            # Ambiguous expression: keep the previous decision, restart the hold timer.
            st.candidate = st.stable
            st.candidate_since = now
            if st.stable == NO_FACE:
                # A face is visible but unclear. Treat it as neutral (robot stays still).
                st.stable = "neutral"
            return st.stable

        if label != st.candidate:
            st.candidate = label
            st.candidate_since = now
        if label != st.stable and now - st.candidate_since >= cfg.hold_time:
            st.stable = label
        elif st.stable == NO_FACE:
            # Face visible, candidate not yet confirmed: hold still meanwhile.
            st.stable = "neutral"
        return st.stable
