"""Turn a record_demo.sh output folder into the final video and report figures.

    python scripts/finalize_demo.py evidence/demo [--mode scripted|live] [--report docs/report]

Steps
1. Measure every manoeuvre from trajectory.csv (radius, distance, rotation).
2. Render a title card and an end card with the measured results.
3. Assemble demo.mp4 = title + recording + end card (H.264, fades).
4. Build report figures in <dir>/figures and a LaTeX table (results.tex).
   With --report the figures and results.tex are copied into the report.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_trajectory import load, plot  # noqa: E402

from tb4_emotion_control import demo_graphics as g  # noqa: E402
from tb4_emotion_control.emotions import EMOTIONS  # noqa: E402

W, H = 1920, 1080
FPS = 15
ACTION_EMOTION = {a: e for e, a in g.EMOTION_ACTION.items()}


# ---------------------------------------------------------------- metrics
SETTLE = 1.5  # s: motion during the first part of a stop segment is deceleration (coasting)


def segments(rows: list[dict], key: str = "action") -> list[tuple[str, list[dict]]]:
    return [(k, list(seg)) for k, seg in groupby(rows, key=lambda r: r[key])]


def path_len(seg: list[dict]) -> float:
    return sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(seg, seg[1:]))


def disp(seg: list[dict]) -> float:
    return math.hypot(seg[-1]["x"] - seg[0]["x"], seg[-1]["y"] - seg[0]["y"]) if seg else 0.0


def swept_yaw(seg: list[dict]) -> float:
    return sum(abs(math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"])))
               for a, b in zip(seg, seg[1:]))


def measure(rows: list[dict]) -> dict[str, dict]:
    """Geometry of the longest execution of every action, plus coasting and idle drift."""
    out: dict[str, dict] = {}
    idle_disp, idle_yaw = [0.0], [0.0]
    segs = [(a, seg) for a, seg in segments(rows) if len(seg) >= 2 and a]
    for i, (action, seg) in enumerate(segs):
        if action == "stop":
            t0 = seg[0]["t"]
            coast = [r for r in seg if r["t"] - t0 <= SETTLE]
            idle = [r for r in seg if r["t"] - t0 >= SETTLE]
            prev = segs[i - 1][0] if i else ""
            if prev in out and "coast_deg" not in out[prev]:
                out[prev]["coast_deg"] = math.degrees(swept_yaw(coast))
                out[prev]["coast_m"] = disp(coast)
            if len(idle) >= 2:
                idle_disp.append(disp(idle))
                idle_yaw.append(math.degrees(swept_yaw(idle)))
            continue
        if action not in ACTION_EMOTION:
            continue
        if action in out and out[action]["duration"] >= seg[-1]["t"] - seg[0]["t"]:
            continue  # keep the longest execution (live runs contain short flickers)
        yaw = swept_yaw(seg)
        m = {"duration": seg[-1]["t"] - seg[0]["t"], "length": path_len(seg), "disp": disp(seg),
             "yaw_deg": math.degrees(yaw)}
        if action in ("arc_left", "arc_right"):
            m["radius"] = m["length"] / yaw if yaw > 1e-6 else float("nan")
        out[action] = m
    out["stop"] = {"disp": max(idle_disp), "yaw_deg": max(idle_yaw)}
    return out


def result_text(action: str, m: dict) -> str:
    if action in ("arc_left", "arc_right"):
        return f"R = {m['radius']:.2f} m · quay {m['yaw_deg']:.0f}° trong {m['duration']:.1f} s"
    if action in ("back_off", "nudge_forward"):
        return f"{m['disp'] * 100:.1f} cm trong {m['duration']:.2f} s"
    if action == "spin_360":
        coast = f" (+{m['coast_deg']:.0f}° quán tính)" if "coast_deg" in m else ""
        return f"{m['yaw_deg']:.1f}° trong {m['duration']:.2f} s{coast}"
    if action == "stop":
        drift = "< 1 mm" if m["disp"] < 0.001 else f"≤ {m['disp'] * 1000:.0f} mm"
        turn = "< 0.1°" if m["yaw_deg"] < 0.1 else f"≤ {m['yaw_deg']:.1f}°"
        return f"đứng yên: trôi {drift} · {turn}"
    return ""


def results_tex(metrics: dict[str, dict]) -> str:
    """Rows for the report's results table (\\input{results})."""
    spec = {
        "arc_left": ("Vui vẻ", "cong sang trái", lambda m: (f"{m['duration']:.1f}", f"{m['length']:.2f}",
                     f"$+{m['yaw_deg']:.0f}^\\circ$", f"$R={m['radius']:.2f}$ m")),
        "arc_right": ("Buồn", "cong sang phải", lambda m: (f"{m['duration']:.1f}", f"{m['length']:.2f}",
                      f"$-{m['yaw_deg']:.0f}^\\circ$", f"$R={m['radius']:.2f}$ m")),
        "stop": ("Bình thường", "dừng", lambda m: ("--", f"$<{max(m['disp'], 0.001):.3f}$",
                 f"$<{max(m['yaw_deg'], 0.1):.1f}^\\circ$", "đứng yên")),
        "back_off": ("Tức giận", "lùi nhẹ", lambda m: (f"{m['duration']:.2f}", f"\\textbf{{{m['disp']:.3f}}}",
                     "$0^\\circ$", "đích 0.150 m")),
        "spin_360": ("Ngạc nhiên", "xoay tại chỗ", lambda m: (f"{m['duration']:.2f}", f"{m['disp']:.3f}",
                     f"$\\mathbf{{{m['yaw_deg']:.1f}^\\circ}}$",
                     f"đích $360^\\circ$, $+{m.get('coast_deg', 0):.0f}^\\circ$ quán tính")),
        "nudge_forward": ("Sợ hãi", "tiến nhẹ", lambda m: (f"{m['duration']:.2f}", f"\\textbf{{{m['disp']:.3f}}}",
                          "$0^\\circ$", "đích 0.150 m")),
    }
    lines = ["% generated by scripts/finalize_demo.py from trajectory.csv"]
    for action in ("arc_left", "arc_right", "stop", "back_off", "spin_360", "nudge_forward"):
        if action in metrics:
            emo, act, fmt = spec[action]
            cells = fmt(metrics[action])
            lines.append(f"{emo} \\ra{{}} {act} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ cards
def title_card(mode: str) -> np.ndarray:
    img = g.canvas(W, H)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 170), "Bài thực hành 08 · Tương tác người – robot", font=g.font(34), fill=g.MUTED, anchor="mm")
    d.text((W // 2, 260), "Nhận diện biểu cảm khuôn mặt", font=g.font(72, "bold"), fill=g.TEXT, anchor="mm")
    d.text((W // 2, 350), "và điều khiển TurtleBot 4 qua ROS 2 /cmd_vel", font=g.font(44), fill=g.TEXT, anchor="mm")
    g.table_rows(d, 460, 440, 1000, [(e, g.VI_ACTION[g.EMOTION_ACTION[e]]) for e in EMOTIONS], active=None,
                 row_h=64, size=32)
    note = ("Kịch bản tự động: nhãn được gửi vào detector như phím 1–6, toàn bộ chuỗi detector → motion → robot đều chạy"
            if mode == "scripted" else "Nhận diện trực tiếp: robot phản ứng theo nét mặt thật của người điều khiển")
    d.text((W // 2, 900), note, font=g.font(26), fill=g.MUTED, anchor="mm")
    d.text((W // 2, 960), "Ubuntu 24.04 · ROS 2 Jazzy · Gazebo Harmonic · DeepFace / FER+ · OpenCV",
           font=g.font(24), fill=g.MUTED, anchor="mm")
    return g.to_bgr(img)


def end_card(metrics: dict[str, dict]) -> np.ndarray:
    img = g.canvas(W, H)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 130), "Kết quả đo từ odometry", font=g.font(60, "bold"), fill=g.TEXT, anchor="mm")
    y = 240
    for e in EMOTIONS:
        action = g.EMOTION_ACTION[e]
        g.rounded(d, (260, y, W - 260, y + 92), fill=g.PANEL)
        d.ellipse((290, y + 34, 314, y + 58), fill=g.emotion_rgb(e))
        d.text((336, y + 46), g.VI_EMOTION[e], font=g.font(34, "bold"), fill=g.emotion_rgb(e), anchor="lm")
        d.text((640, y + 46), g.VI_ACTION[action], font=g.font(32), fill=g.TEXT, anchor="lm")
        m = metrics.get(action)
        d.text((W - 300, y + 46), result_text(action, m) if m else "—", font=g.font(30, "bold"),
               fill=g.ACCENT if m else g.MUTED, anchor="rm")
        y += 108
    d.text((W // 2, 960), "Vận tốc: cong v = 0.15 m/s, |ω| = 0.5 rad/s · lùi/tiến 0.10 m/s · xoay 1.0 rad/s",
           font=g.font(26), fill=g.MUTED, anchor="mm")
    return g.to_bgr(img)


# ------------------------------------------------------------------ video
def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def assemble_video(out: Path, title: Path, end: Path) -> Path:
    raw = out / "demo_raw.avi"
    dst = out / "demo.mp4"
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(raw)],
        check=True, capture_output=True, text=True).stdout.strip())
    t_title, t_end = 5.0, 7.0
    ffmpeg(
        "-loop", "1", "-t", str(t_title), "-framerate", str(FPS), "-i", str(title),
        "-i", str(raw),
        "-loop", "1", "-t", str(t_end), "-framerate", str(FPS), "-i", str(end),
        "-filter_complex",
        f"[0:v]fps={FPS},format=yuv420p,fade=t=in:d=0.6,fade=t=out:st={t_title - 0.6}:d=0.6[a];"
        f"[1:v]fps={FPS},format=yuv420p,fade=t=in:d=0.4,fade=t=out:st={dur - 0.5:.2f}:d=0.5[b];"
        f"[2:v]fps={FPS},format=yuv420p,fade=t=in:d=0.6[c];"
        "[a][b][c]concat=n=3:v=1:a=0[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-threads", "4", "-crf", "20", "-movflags", "+faststart",
        str(dst),
    )
    return dst


# ---------------------------------------------------------------- figures
def snapshot(out: Path, label: str, kind: str) -> np.ndarray | None:
    files = sorted((out / "snapshots").glob(f"*_{label}_{kind}.png"))
    return cv2.imread(str(files[0])) if files else None


def first_segment(rows: list[dict], label: str) -> list[dict]:
    """Rows of the first block in which the motion node acted on ``label`` (scripted runs)."""
    for key, seg in segments(rows, "label"):
        if key == label and len(seg) >= 2:
            return seg
    return []


def longest_segment(rows: list[dict], label: str) -> list[dict]:
    """Rows of the longest block for ``label`` (live runs, where labels can flicker)."""
    segs = [seg for key, seg in segments(rows, "label") if key == label and len(seg) >= 2]
    return max(segs, key=lambda seg: seg[-1]["t"] - seg[0]["t"]) if segs else []


def action_cell(top: np.ndarray, seg: list[dict], label: str, metric: str,
                cw: int = 960, ch: int = 720) -> np.ndarray:
    """Top-view crop around one manoeuvre: trail, start, motion/heading arrow, measurement."""
    size = top.shape[1]
    xs, ys = [r["x"] for r in seg], [r["y"] for r in seg]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_w = max(0.6, (max(xs) - min(xs)) / 2 + 0.35)
    half_h = max(half_w * ch / cw, (max(ys) - min(ys)) / 2 + 0.35)
    half_w = max(half_w, half_h * cw / ch)
    u0, v0 = g.top_project(cx - half_w, cy + half_h, size)
    u1, v1 = g.top_project(cx + half_w, cy - half_h, size)
    pad = size  # allow crops that leave the image
    big = cv2.copyMakeBorder(top, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = big[int(v0) + pad:int(v1) + pad, int(u0) + pad:int(u1) + pad]
    cell = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_CUBIC)
    k = cw / (u1 - u0)
    m_to_px = k / g.top_m_per_px(size, g.ROBOT_VIEW_Z)

    def px(x: float, y: float) -> tuple[int, int]:
        u, v = g.top_project(x, y, size)
        return int(round((u - u0) * k)), int(round((v - v0) * k))

    def outlined_line(p0, p1, color, width, arrow=False):
        draw = cv2.arrowedLine if arrow else cv2.line
        extra = {"tipLength": 0.35} if arrow else {}
        draw(cell, p0, p1, (255, 255, 255), width + 6, cv2.LINE_AA, **extra)
        draw(cell, p0, p1, color, width, cv2.LINE_AA, **extra)

    action = g.EMOTION_ACTION[label]
    color = g.ACTION_RGB.get(action, (200, 200, 200))[::-1]
    start, end = seg[0], seg[-1]
    s_px, e_px = px(start["x"], start["y"]), px(end["x"], end["y"])
    if action in ("arc_left", "arc_right"):
        pts = np.array([px(r["x"], r["y"]) for r in seg], np.int32)
        cv2.polylines(cell, [pts], False, (255, 255, 255), 14, cv2.LINE_AA)
        cv2.polylines(cell, [pts], False, color, 8, cv2.LINE_AA)
    elif action in ("back_off", "nudge_forward"):
        outlined_line(s_px, e_px, color, 9, arrow=True)  # direction of travel
    elif action == "spin_360":
        r = int(0.33 * m_to_px)
        cv2.ellipse(cell, e_px, (r, r), 0, -60, 250, (255, 255, 255), 14, cv2.LINE_AA)
        cv2.ellipse(cell, e_px, (r, r), 0, -60, 250, color, 8, cv2.LINE_AA)
        a = math.radians(250)
        head = (int(e_px[0] + r * math.cos(a)), int(e_px[1] + r * math.sin(a)))
        cv2.arrowedLine(cell, (head[0] + 34, head[1] + 12), head, color, 8, cv2.LINE_AA, tipLength=1.0)
    else:  # neutral: ring around the standing robot
        r = int(0.24 * m_to_px)
        cv2.circle(cell, e_px, r, (255, 255, 255), 10, cv2.LINE_AA)
        cv2.circle(cell, e_px, r, (150, 150, 150), 5, cv2.LINE_AA)
    if action != "stop":
        cv2.circle(cell, s_px, 12, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(cell, s_px, 12, (40, 40, 40), 3, cv2.LINE_AA)
    if action in ("arc_left", "arc_right", "spin_360"):  # final heading
        tip = px(end["x"] + 0.2 * math.cos(end["yaw"]), end["y"] + 0.2 * math.sin(end["yaw"]))
        outlined_line(e_px, tip, (40, 40, 40), 4, arrow=True)

    img = g.to_pil(cell)
    d = ImageDraw.Draw(img)
    g.chip(d, (20, 20), f"{g.VI_EMOTION[label]}  →  {g.VI_ACTION[action]}", g.emotion_rgb(label), size=28)
    _, mh = g.chip_size(d, metric, size=24)
    g.chip(d, (20, ch - mh - 20), metric, (30, 32, 36), fg=(255, 255, 255), size=24)
    bar = int(0.2 * m_to_px)  # 20 cm scale bar
    bx, by = cw - 40 - bar, ch - 40
    d.rounded_rectangle((bx - 14, by - 42, cw - 26, by + 16), radius=8, fill=(30, 32, 36))
    d.line((bx, by, bx + bar, by), fill=(255, 255, 255), width=6)
    d.text((bx + bar // 2, by - 10), "20 cm", font=g.font(20, "bold"), fill=(255, 255, 255), anchor="mb")
    return g.to_bgr(img)


def actions_grid(out: Path, rows: list[dict], metrics: dict[str, dict], live: bool = False) -> np.ndarray:
    """3 x 2 grid, one cell per expression (assignment order)."""
    cw, ch, gap = 960, 720, 8
    grid = np.empty((ch * 3 + gap * 2, cw * 2 + gap, 3), np.uint8)
    grid[:] = g.BG[::-1]
    for i, e in enumerate(EMOTIONS):
        action = g.EMOTION_ACTION[e]
        top = snapshot(out, e, "top_end")
        seg = longest_segment(rows, e) if live else first_segment(rows, e)
        if top is None or not seg:
            print(f"  (no snapshot/trajectory for {e})")
            continue
        metric = result_text(action, metrics[action]) if action in metrics else ""
        x0, y0 = (i % 2) * (cw + gap), (i // 2) * (ch + gap)
        grid[y0:y0 + ch, x0:x0 + cw] = action_cell(top, seg, e, metric, cw, ch)
    return grid


def faces_grid(out: Path) -> np.ndarray | None:
    """3 x 2 grid of the recognition overlay for each prompted expression (live runs)."""
    cw, ch, gap = 960, 476, 8
    cells = [(e, snapshot(out, e, "overlay")) for e in EMOTIONS]
    if not any(img is not None for _, img in cells):
        return None
    grid = np.empty((ch * 3 + gap * 2, cw * 2 + gap, 3), np.uint8)
    grid[:] = g.BG[::-1]
    for i, (e, img) in enumerate(cells):
        x0, y0 = (i % 2) * (cw + gap), (i // 2) * (ch + gap)
        grid[y0:y0 + ch, x0:x0 + cw] = g.fit_np(img, cw, ch, "chưa có ảnh")
    return grid


def build_figures(out: Path, rows: list[dict], metrics: dict[str, dict], live: bool = False) -> Path:
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    cv2.imwrite(str(fig / "actions_grid.png"), actions_grid(out, rows, metrics, live))
    faces = faces_grid(out) if live else None
    if faces is not None:
        cv2.imwrite(str(fig / "faces_grid.png"), faces)
    for label in ("happy", "surprise"):
        comp = snapshot(out, label, "composite")
        if comp is not None:
            cv2.imwrite(str(fig / f"demo_frame_{label}.png"), comp)
    overlay = snapshot(out, "happy", "overlay")
    if overlay is not None:
        cv2.imwrite(str(fig / "interface.png"), overlay)
    if (out / "trajectory_top.png").is_file():
        shutil.copy(out / "trajectory_top.png", fig / "trajectory_top.png")
    plot(rows, fig / "trajectory_odom.png")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    ap.add_argument("--report", type=Path, help="copy figures + results.tex into this report folder")
    args = ap.parse_args()
    out = args.out.resolve()

    rows = load(out / "trajectory.csv")
    metrics = measure(rows)
    for action, m in metrics.items():
        print(f"  {action:<14} {result_text(action, m)}")

    cv2.imwrite(str(out / "title_card.png"), title_card(args.mode))
    cv2.imwrite(str(out / "end_card.png"), end_card(metrics))
    if (out / "demo_raw.avi").is_file():
        video = assemble_video(out, out / "title_card.png", out / "end_card.png")
        print(f"video: {video}")
    fig = build_figures(out, rows, metrics, live=args.mode == "live")
    (fig / "results.tex").write_text(results_tex(metrics))
    print(f"figures: {fig}")

    if args.report:
        dst = args.report / "figures"
        dst.mkdir(parents=True, exist_ok=True)
        for f in fig.iterdir():
            shutil.copy(f, (args.report if f.suffix == ".tex" else dst) / f.name)
        print(f"copied into {args.report}")


if __name__ == "__main__":
    main()
