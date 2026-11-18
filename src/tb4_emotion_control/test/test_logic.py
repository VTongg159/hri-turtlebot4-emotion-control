import math

import pytest

from tb4_emotion_control.emotion_sequence import parse_sequence
from tb4_emotion_control.emotions import EMOTIONS, NO_FACE, dominant, normalize_scores
from tb4_emotion_control.motion import MotionController, default_primitives
from tb4_emotion_control.odometry import ProgressTracker
from tb4_emotion_control.smoother import EmotionSmoother, SmootherConfig


def one_hot(label, p=0.9):
    rest = (1.0 - p) / (len(EMOTIONS) - 1)
    return {e: (p if e == label else rest) for e in EMOTIONS}


# --------------------------------------------------------------- emotions
def test_normalize_drops_disgust_and_handles_percentages():
    s = normalize_scores({"happy": 60, "sad": 10, "disgust": 30, "neutral": 10, "angry": 0,
                          "surprise": 0, "fear": 0})
    assert set(s) == set(EMOTIONS)
    assert math.isclose(sum(s.values()), 1.0)
    assert dominant(s) == ("happy", pytest.approx(0.75))


def test_normalize_accepts_fer_fractions_and_aliases():
    s = normalize_scores({"happiness": 0.1, "surprised": 0.7, "neutral": 0.2})
    assert dominant(s)[0] == "surprise"


def test_normalize_empty_is_neutral():
    assert dominant(normalize_scores({}))[0] == "neutral"


# --------------------------------------------------------------- smoother
def make_smoother(**kw):
    cfg = dict(alpha=0.5, min_confidence=0.4, hold_time=0.5, face_lost_timeout=1.0)
    cfg.update(kw)
    return EmotionSmoother(SmootherConfig(**cfg))


def test_smoother_requires_hold_time():
    sm = make_smoother()
    assert sm.update(one_hot("neutral"), 0.0) == "neutral"
    t = 0.1
    while t < 0.45:
        assert sm.update(one_hot("happy"), t) == "neutral"
        t += 0.1
    # after the EMA flips and the candidate has been on top >= hold_time
    for _ in range(10):
        t += 0.1
        out = sm.update(one_hot("happy"), t)
    assert out == "happy"


def test_smoother_ignores_single_frame_glitch():
    sm = make_smoother()
    t = 0.0
    for _ in range(10):
        sm.update(one_hot("sad"), t)
        t += 0.1
    assert sm.stable == "sad"
    sm.update(one_hot("angry", 1.0), t)  # one outlier frame
    for _ in range(3):
        t += 0.1
        sm.update(one_hot("sad"), t)
    assert sm.stable == "sad"


def test_smoother_face_lost_after_timeout():
    sm = make_smoother()
    for i in range(10):
        sm.update(one_hot("happy"), i * 0.1)
    assert sm.stable == "happy"
    assert sm.update(None, 1.2) == "happy"  # brief dropout tolerated
    assert sm.update(None, 2.0) == NO_FACE


def test_smoother_ambiguous_face_is_neutral_not_none():
    sm = make_smoother(min_confidence=0.6)
    flat = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    assert sm.update(flat, 0.0) == "neutral"


def test_smoother_rejects_bad_alpha():
    with pytest.raises(ValueError):
        EmotionSmoother(SmootherConfig(alpha=0.0))


# ----------------------------------------------------------------- motion
def run(ctrl, label, t0, t1, dt=0.05, **odom):
    cmds = []
    t = t0
    while t < t1:
        ctrl.set_label(label, t)
        cmds.append(ctrl.step(t, **odom))
        t += dt
    return cmds


def test_mapping_directions():
    p = default_primitives()
    assert p["happy"].linear > 0 and p["happy"].angular > 0  # curve left
    assert p["sad"].linear > 0 and p["sad"].angular < 0  # curve right
    assert p["neutral"].is_stop
    assert p["angry"].linear < 0 and p["angry"].angular == 0  # back off
    assert p["surprise"].linear == 0 and p["surprise"].target_angle == pytest.approx(2 * math.pi)
    assert p["fear"].linear > 0 and p["fear"].angular == 0  # forward


def test_arc_is_continuous_and_follows_label():
    c = MotionController()
    cmds = run(c, "happy", 0.0, 10.0)
    assert all(cmd.action == "arc_left" for cmd in cmds)
    c.set_label("sad", 10.0)
    assert c.step(10.0).action == "arc_right"
    c.set_label("neutral", 10.05)
    cmd = c.step(10.05)
    assert (cmd.linear, cmd.angular, cmd.action) == (0.0, 0.0, "stop")


def test_spin_is_atomic_then_needs_rearm():
    c = MotionController()
    budget = default_primitives()["surprise"].time_budget()
    assert budget == pytest.approx(2 * math.pi)
    c.set_label("surprise", 0.0)
    assert c.step(0.0).action == "spin_360"
    # the expression fades to neutral, the spin must still finish (open loop)
    cmds = run(c, "neutral", 0.05, budget - 0.1)
    assert all(cmd.action == "spin_360" for cmd in cmds)
    assert c.step(budget + 0.1).action == "stop"


def test_held_expression_triggers_one_shot_once():
    c = MotionController()
    budget = default_primitives()["surprise"].time_budget()
    cmds = run(c, "surprise", 0.0, budget + 2.0)  # surprised the whole time
    assert cmds[0].action == "spin_360" and cmds[-1].action == "stop"
    assert sum(1 for a, b in zip(cmds, cmds[1:]) if a.action != b.action) == 1
    run(c, "neutral", budget + 2.0, budget + 2.2)  # relax, then surprised again
    assert run(c, "surprise", budget + 2.2, budget + 2.3)[0].action == "spin_360"


def test_one_shot_rearms_if_expression_changed_during_it():
    c = MotionController()
    c.set_label("surprise", 0.0)
    c.step(0.0)
    run(c, "neutral", 0.05, 3.0)  # face relaxes mid-spin
    run(c, "surprise", 3.0, 6.5)  # surprised again before the spin ends
    cmds = run(c, "surprise", 6.5, 7.0)
    assert cmds[-1].action == "spin_360"  # finished one spin, started the next


def test_one_shot_uses_odometry_when_available():
    c = MotionController()
    c.set_label("fear", 0.0)
    assert c.step(0.0, distance=0.0, angle=0.0).action == "nudge_forward"
    c.set_label("fear", 0.5)
    cmd = c.step(0.5, distance=0.075, angle=0.0)
    assert cmd.action == "nudge_forward" and cmd.progress == pytest.approx(0.5)
    # odometry shows the robot lagging: keep going past the open-loop budget
    c.set_label("fear", 1.6)
    assert c.step(1.6, distance=0.12, angle=0.0).action == "nudge_forward"
    c.set_label("fear", 1.8)
    assert c.step(1.8, distance=0.151, angle=0.0).action == "stop"


def test_one_shot_safety_timeout_when_blocked():
    c = MotionController(one_shot_timeout_factor=2.0)
    c.set_label("angry", 0.0)
    assert c.step(0.0, distance=0.0).linear < 0
    cmds = run(c, "angry", 0.05, 3.1, distance=0.0)  # wheels spin, robot stuck
    assert cmds[-1].action == "stop"


def test_estop_overrides_everything():
    c = MotionController()
    c.set_label("surprise", 0.0)
    c.step(0.0)
    c.set_estop(True)
    cmd = c.step(0.1)
    assert (cmd.linear, cmd.angular, cmd.action) == (0.0, 0.0, "estop")
    c.set_estop(False)
    c.set_label("happy", 0.2)
    assert c.step(0.2).action == "arc_left"


def test_watchdog_stops_when_detector_silent():
    c = MotionController(input_timeout=1.0)
    assert c.step(0.0).action == "watchdog_stop"  # nothing received yet
    c.set_label("happy", 0.0)
    assert c.step(0.5).action == "arc_left"
    assert c.step(1.6).action == "watchdog_stop"


def test_no_face_stops():
    c = MotionController()
    c.set_label(NO_FACE, 0.0)
    assert c.step(0.0).action == "stop"


def test_rejects_non_positive_speed():
    with pytest.raises(ValueError):
        default_primitives(arc_linear=-0.1)


# --------------------------------------------------------------- odometry
def test_tracker_unwraps_full_turn():
    tr = ProgressTracker()
    n = 100
    for i in range(n + 1):
        yaw = math.atan2(math.sin(2 * math.pi * i / n), math.cos(2 * math.pi * i / n))
        tr.update(0.0, 0.0, yaw)
    assert tr.angle == pytest.approx(2 * math.pi, rel=1e-6)
    assert tr.distance == 0.0


def test_tracker_distance_and_reset():
    tr = ProgressTracker()
    tr.update(1.0, 1.0, 0.0)
    tr.update(1.0, 0.85, 0.0)
    assert tr.distance == pytest.approx(0.15)
    tr.reset()
    assert not tr.started and tr.distance == 0.0


# --------------------------------------------------------------- sequence
def test_parse_sequence():
    assert parse_sequence("happy:2, sad:1.5,neutral") == [("happy", 2.0), ("sad", 1.5), ("neutral", 3.0)]
    with pytest.raises(ValueError):
        parse_sequence("joy:2")


# --------------------------------------------------------------- backends
def test_auto_backend_resolves_to_an_installed_backend():
    from tb4_emotion_control.backends import deepface_available, hsemotion_available, resolve_backend_name

    expected = "hsemotion" if hsemotion_available() else ("deepface" if deepface_available() else "ferplus")
    assert resolve_backend_name("auto") == expected
    assert resolve_backend_name("fer") == "fer"


def test_ferplus_backend_runs_on_synthetic_frame():
    import numpy as np

    from tb4_emotion_control.backends import FerPlusBackend, default_ferplus_path

    try:
        default_ferplus_path()
    except FileNotFoundError:
        pytest.skip("FER+ model not downloaded (scripts/download_models.sh)")
    backend = FerPlusBackend()
    assert backend.analyze(np.zeros((240, 320, 3), dtype=np.uint8)) == []  # no face
    # classify a crop directly: output must be a distribution over the six emotions
    class FixedFace:
        def detectMultiScale(self, *args, **kwargs):  # noqa: N802 (OpenCV name)
            return [(100, 60, 120, 120)]

    backend._cascade = FixedFace()
    faces = backend.analyze(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert len(faces) == 1
    assert set(faces[0].scores) == set(EMOTIONS)
    assert math.isclose(sum(faces[0].scores.values()), 1.0, rel_tol=1e-6)


# ------------------------------------------------------------- calibration
def _biased_samples(n=40, seed=1):
    """Mimic FER-style bias: fear scores as surprise, sad scores as neutral."""
    import random

    rng = random.Random(seed)
    leak = {"fear": ("surprise", 0.55, 0.35), "sad": ("neutral", 0.55, 0.30)}
    samples = {}
    for label in EMOTIONS:
        frames = []
        for _ in range(n):
            s = {e: rng.uniform(0.0, 0.05) for e in EMOTIONS}
            if label in leak:
                other, p_other, p_true = leak[label]
                s[other] += p_other + rng.uniform(-0.05, 0.05)
                s[label] += p_true + rng.uniform(-0.05, 0.05)
            else:
                s[label] += 0.8
            frames.append(normalize_scores(s))
        samples[label] = frames
    return samples


def test_fit_gains_recovers_fear_and_sad():
    from tb4_emotion_control.calibration import balanced_accuracy, fit_gains, per_class_recall

    samples = _biased_samples()
    unit = {e: 1.0 for e in EMOTIONS}
    before = per_class_recall(samples, unit)
    assert before["fear"] < 0.1 and before["sad"] < 0.1
    gains = fit_gains(samples)
    after = per_class_recall(samples, gains)
    assert after["fear"] > 0.9 and after["sad"] > 0.9
    assert min(after.values()) > 0.9  # without breaking surprise / neutral
    assert balanced_accuracy(samples, gains) > balanced_accuracy(samples, unit)
    # only ratios matter: fear must gain on surprise, sad on neutral
    assert gains["neutral"] == 1.0
    assert gains["fear"] > gains["surprise"] and gains["sad"] > gains["neutral"]


def test_fit_gains_keeps_unit_gains_when_already_correct():
    from tb4_emotion_control.calibration import fit_gains

    samples = {e: [one_hot(e)] * 10 for e in EMOTIONS}
    assert fit_gains(samples) == {e: 1.0 for e in EMOTIONS}


def test_calibration_roundtrip_and_apply(tmp_path):
    from tb4_emotion_control.calibration import apply_gains, load_calibration, save_calibration

    gains = {e: 1.0 for e in EMOTIONS} | {"fear": 3.0}
    path = save_calibration(tmp_path / "c.yaml", gains, {"backend": "ferplus"})
    loaded, data = load_calibration(path)
    assert loaded == gains and data["backend"] == "ferplus"
    adj = apply_gains(one_hot("surprise", 0.5) | {"fear": 0.3}, loaded)
    assert dominant(adj)[0] == "fear" and math.isclose(sum(adj.values()), 1.0)
    assert load_calibration(tmp_path / "missing.yaml") == ({e: 1.0 for e in EMOTIONS}, {})


def test_hsemotion_backend_runs_on_synthetic_frame():
    import numpy as np

    from tb4_emotion_control.backends import HSEmotionBackend, hsemotion_available

    if not hsemotion_available():
        pytest.skip("onnxruntime or HSEmotion model not installed")
    backend = HSEmotionBackend()
    assert backend.analyze(np.zeros((240, 320, 3), dtype=np.uint8)) == []

    class FixedFace:
        def detectMultiScale(self, *args, **kwargs):  # noqa: N802 (OpenCV name)
            return [(100, 60, 120, 120)]

    backend._cascade = FixedFace()
    faces = backend.analyze(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert len(faces) == 1 and set(faces[0].scores) == set(EMOTIONS)
    assert math.isclose(sum(faces[0].scores.values()), 1.0, rel_tol=1e-6)
