"""Record the operator's six expressions and fit per-class gains.

    ros2 run tb4_emotion_control emotion_calibrate            # backend auto
    ros2 run tb4_emotion_control emotion_calibrate --backend ferplus --seconds 4

For each expression: get ready, press SPACE, hold the expression while the
bar fills (default 3 s). Keys: SPACE start/record, R redo previous, Q abort.
Only the score vectors are kept (no images). The result is written to
~/.ros/tb4_emotion_calibration.yaml, which emotion_detector_node loads.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2

from .backends import create_backend, largest_face, resolve_backend_name
from .calibration import (
    balanced_accuracy,
    confusion,
    fit_gains,
    per_class_recall,
    save_calibration,
)
from .emotions import COLORS, DISPLAY_NAMES, EMOTIONS
from .overlay import OverlayState, render

DEFAULT_OUTPUT = "~/.ros/tb4_emotion_calibration.yaml"
HINTS = {  # ASCII (OpenCV banner); the same advice as demo_graphics.VI_HINT
    "happy": "cuoi tuoi, khoe mieng keo len, mat hip lai",
    "sad": "tre moi xuong, khoe mieng cup, nhiu giua long may len",
    "neutral": "mat tha long tu nhien",
    "angry": "chau may that manh, mim chat moi, KHONG nhe rang",
    "surprise": "mo to mat, nhuong may, ha mieng tron",
    "fear": "mo to mat, nhuong may, keo mieng sang ngang (khong ha tron)",
}
WINDOW = "Emotion calibration"


def _banner(img, lines, color=(255, 255, 255)):
    y = 30
    for i, text in enumerate(lines):
        scale, thick = (0.8, 2) if i == 0 else (0.55, 1)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += 32 if i == 0 else 24


def print_report(samples, gains) -> None:
    unit = {e: 1.0 for e in EMOTIONS}
    before, after = per_class_recall(samples, unit), per_class_recall(samples, gains)
    print("\nper-class recall (fraction of frames classified correctly)")
    print(f"{'emotion':<10}{'before':>8}{'after':>8}{'gain':>8}")
    for e in EMOTIONS:
        print(f"{e:<10}{before.get(e, 0):8.2f}{after.get(e, 0):8.2f}{gains[e]:8.2f}")
    print(f"{'balanced':<10}{balanced_accuracy(samples, unit):8.2f}{balanced_accuracy(samples, gains):8.2f}")
    print("\nconfusion after calibration (rows: shown, cols: predicted)")
    print(" " * 10 + "".join(f"{e[:7]:>8}" for e in EMOTIONS))
    for t, row in confusion(samples, gains).items():
        print(f"{t:<10}" + "".join(f"{row[p]:8d}" for p in EMOTIONS))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--camera", default="0")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument(
        "--only",
        default="",
        help="comma list, e.g. fear,sad: re-record only these and reuse the other "
        "expressions from the previous run's samples file",
    )
    args, _ = ap.parse_known_args(argv)  # ignore --ros-args when run via ros2 run

    backend_name = resolve_backend_name(args.backend)
    print(f"backend: {backend_name}")
    backend = create_backend(backend_name)
    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    samples_path = Path(args.output).expanduser().with_name("tb4_emotion_calibration_samples.json")
    samples: dict[str, list] = {e: [] for e in EMOTIONS}
    order = list(EMOTIONS)
    if args.only:
        order = [e.strip().lower() for e in args.only.split(",") if e.strip()]
        unknown = set(order) - set(EMOTIONS)
        if unknown:
            raise SystemExit(f"unknown emotions: {sorted(unknown)}")
        if not samples_path.is_file():
            raise SystemExit(f"--only needs a previous full run ({samples_path} missing)")
        previous = json.loads(samples_path.read_text())
        if previous.get("backend") != backend_name:
            raise SystemExit(f"previous samples were recorded with backend '{previous.get('backend')}'")
        for e in EMOTIONS:
            if e not in order:
                samples[e] = previous["samples"].get(e, [])
        print(f"re-recording {order}, reusing the others from {samples_path}")

    idx, recording, rec_start = 0, False, 0.0
    cooldown_until = 0.0  # ignore keys right after a recording ends
    try:
        while idx < len(order):
            ok, frame = cap.read()
            if not ok:
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            face = largest_face(backend.analyze(frame))
            emotion = order[idx]
            now = time.monotonic()
            if recording and face is not None:
                samples[emotion].append(face.scores)
            state = OverlayState(
                box=face.box if face else None,
                raw=face.scores if face else {},
                smooth=face.scores if face else {},
                stable=emotion,
                backend=f"{backend_name} | calibration {idx + 1}/{len(order)}",
                action="",
            )
            view = render(frame, state)
            if recording:
                frac = min(1.0, (now - rec_start) / args.seconds)
                cv2.rectangle(view, (0, frame.shape[0] - 12), (int(frame.shape[1] * frac), frame.shape[0]),
                              COLORS[emotion], -1)
                _banner(view, [f"GIU NGUYEN: {DISPLAY_NAMES[emotion]}", f"{len(samples[emotion])} frames"],
                        COLORS[emotion])
                if frac >= 1.0:
                    recording = False
                    cooldown_until = now + 1.0
                    if len(samples[emotion]) < 10:
                        print(f"{emotion}: too few frames with a face ({len(samples[emotion])}), repeat")
                        samples[emotion].clear()
                    else:
                        print(f"{emotion}: {len(samples[emotion])} frames")
                        idx += 1
            else:
                _banner(view, [f"Bieu cam {idx + 1}/{len(order)}: {DISPLAY_NAMES[emotion]}", HINTS[emotion],
                               "SPACE: ghi   R: lam lai bieu cam truoc   Q: thoat"], COLORS[emotion])
                if face is None:
                    _banner(view, ["", "", "", "khong thay mat - nhin vao camera"], (0, 0, 255))
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise SystemExit("aborted, nothing saved")
            if recording or now < cooldown_until:
                continue  # drop key presses (incl. auto-repeat) during and just after a take
            if key == ord(" "):
                samples[emotion].clear()
                recording, rec_start = True, time.monotonic()
            elif key == ord("r") and idx > 0:
                idx -= 1
                samples[order[idx]].clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    missing = [e for e in EMOTIONS if len(samples[e]) < 10]
    if missing:
        raise SystemExit(f"not enough frames for {missing}; nothing saved, please run again")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.write_text(json.dumps({"backend": backend_name, "samples": samples}))

    gains = fit_gains(samples)
    print_report(samples, gains)
    unit = {e: 1.0 for e in EMOTIONS}
    path = save_calibration(
        args.output,
        gains,
        {
            "backend": backend_name,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "frames": {e: len(v) for e, v in samples.items()},
            "recall_before": {e: round(v, 3) for e, v in per_class_recall(samples, unit).items()},
            "recall_after": {e: round(v, 3) for e, v in per_class_recall(samples, gains).items()},
            "mean_scores": {
                t: {e: round(sum(f[e] for f in frames) / len(frames), 3) for e in EMOTIONS}
                for t, frames in samples.items()
            },
            "confusion_after": confusion(samples, gains),
        },
    )
    print(f"\nsaved {path}\nrestart emotion_detector_node (or the launch file) to use it.")


if __name__ == "__main__":
    main()
