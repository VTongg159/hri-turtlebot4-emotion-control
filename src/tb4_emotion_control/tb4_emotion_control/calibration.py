"""Per-user class calibration for expression scores.

Expression CNNs trained on FER2013/FER+ see few "fear" and "sad" examples,
so for many people a fearful face scores highest on "surprise" and a sad face
on "neutral". A per-class gain (a prior correction, p_k' ~ g_k * p_k) fixes
most of that. The gains are fitted on ~3 s of score vectors recorded for each
of the six expressions of the actual operator: coordinate search over
log-gains that maximises balanced accuracy (mean per-class recall), with a
small penalty that keeps gains close to 1.

Only score vectors are stored; no images.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from .emotions import EMOTIONS

Samples = Mapping[str, Sequence[Mapping[str, float]]]  # true label -> score dicts


def apply_gains(scores: Mapping[str, float], gains: Mapping[str, float]) -> dict[str, float]:
    """Rescale scores by per-class gains and renormalise to sum to 1."""
    adj = {e: float(scores.get(e, 0.0)) * float(gains.get(e, 1.0)) for e in EMOTIONS}
    total = sum(adj.values())
    if total <= 0.0:
        return dict(scores)
    return {e: v / total for e, v in adj.items()}


def predict(scores: Mapping[str, float], gains: Mapping[str, float]) -> str:
    return max(EMOTIONS, key=lambda e: float(scores.get(e, 0.0)) * float(gains.get(e, 1.0)))


def per_class_recall(samples: Samples, gains: Mapping[str, float]) -> dict[str, float]:
    recall = {}
    for label, frames in samples.items():
        if frames:
            recall[label] = sum(predict(f, gains) == label for f in frames) / len(frames)
    return recall


def balanced_accuracy(samples: Samples, gains: Mapping[str, float]) -> float:
    r = per_class_recall(samples, gains)
    return sum(r.values()) / len(r) if r else 0.0


def confusion(samples: Samples, gains: Mapping[str, float]) -> dict[str, dict[str, int]]:
    table = {t: {p: 0 for p in EMOTIONS} for t in samples}
    for label, frames in samples.items():
        for f in frames:
            table[label][predict(f, gains)] += 1
    return table


def fit_gains(
    samples: Samples,
    log_range: tuple[float, float] = (-1.5, 3.0),
    step: float = 0.1,
    sweeps: int = 8,
    penalty: float = 0.004,
) -> dict[str, float]:
    """Fit gains (neutral fixed at 1) by coordinate search on balanced accuracy."""
    log_g = {e: 0.0 for e in EMOTIONS}
    grid = [log_range[0] + i * step for i in range(int(round((log_range[1] - log_range[0]) / step)) + 1)]

    def objective(lg: Mapping[str, float]) -> float:
        g = {e: math.exp(v) for e, v in lg.items()}
        return balanced_accuracy(samples, g) - penalty * sum(abs(v) for v in lg.values())

    best = objective(log_g)
    for _ in range(sweeps):
        improved = False
        for e in EMOTIONS:
            if e == "neutral":
                continue
            for v in grid:
                trial = dict(log_g)
                trial[e] = v
                score = objective(trial)
                if score > best + 1e-9:
                    best, log_g, improved = score, trial, True
        if not improved:
            break
    return {e: round(math.exp(v), 3) for e, v in log_g.items()}


def save_calibration(path: str | Path, gains: Mapping[str, float], meta: Mapping) -> Path:
    import yaml  # noqa: PLC0415

    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"gains": {e: float(gains.get(e, 1.0)) for e in EMOTIONS}, **dict(meta)}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return path


def load_calibration(path: str | Path) -> tuple[dict[str, float], dict]:
    """Return (gains, whole file). Missing file -> unit gains."""
    import yaml  # noqa: PLC0415

    path = Path(path).expanduser()
    if not path.is_file():
        return {e: 1.0 for e in EMOTIONS}, {}
    data = yaml.safe_load(path.read_text()) or {}
    raw = data.get("gains", {}) or {}
    gains = {e: float(raw.get(e, 1.0)) for e in EMOTIONS}
    if any(g <= 0 or not math.isfinite(g) for g in gains.values()):
        raise ValueError(f"invalid gains in {path}: {gains}")
    return gains, data
