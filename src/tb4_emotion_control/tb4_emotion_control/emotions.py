"""Emotion labels shared by the detector, the smoother and the motion controller."""

from __future__ import annotations

from typing import Mapping

# The six expressions required by the assignment, in display order.
EMOTIONS: tuple[str, ...] = ("happy", "sad", "neutral", "angry", "surprise", "fear")

# Pseudo-label published when no usable face is in view.
NO_FACE = "none"

VALID_LABELS: frozenset[str] = frozenset(EMOTIONS + (NO_FACE,))

# Labels as printed on the camera overlay (OpenCV cannot draw Vietnamese diacritics).
DISPLAY_NAMES: Mapping[str, str] = {
    "happy": "Happy (Vui ve)",
    "sad": "Sad (Buon)",
    "neutral": "Neutral (Binh thuong)",
    "angry": "Angry (Tuc gian)",
    "surprise": "Surprise (Ngac nhien)",
    "fear": "Fear (So hai)",
    NO_FACE: "No face",
}

# BGR colours for the overlay.
COLORS: Mapping[str, tuple[int, int, int]] = {
    "happy": (0, 200, 255),
    "sad": (255, 140, 40),
    "neutral": (200, 200, 200),
    "angry": (40, 40, 230),
    "surprise": (230, 80, 230),
    "fear": (80, 200, 80),
    NO_FACE: (120, 120, 120),
}

_ALIASES = {
    "happiness": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprised": "surprise",
    "fearful": "fear",
    "scared": "fear",
    "calm": "neutral",
}


def normalize_label(label: str) -> str:
    """Map backend-specific spellings onto the canonical label set."""
    key = label.strip().lower()
    return _ALIASES.get(key, key)


def normalize_scores(raw: Mapping[str, float]) -> dict[str, float]:
    """Keep only the six target emotions and rescale them to sum to 1.

    DeepFace reports percentages (0-100) and FER reports fractions (0-1); both
    also include "disgust", which the assignment does not use. Dropping it and
    renormalising lets the two backends share one decision rule.
    """
    scores = {e: 0.0 for e in EMOTIONS}
    for name, value in raw.items():
        label = normalize_label(name)
        if label in scores:
            scores[label] += max(0.0, float(value))
    total = sum(scores.values())
    if total <= 0.0:
        return {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
    return {e: v / total for e, v in scores.items()}


def dominant(scores: Mapping[str, float]) -> tuple[str, float]:
    """Return the (label, score) pair with the highest score."""
    label = max(EMOTIONS, key=lambda e: scores.get(e, 0.0))
    return label, float(scores.get(label, 0.0))
