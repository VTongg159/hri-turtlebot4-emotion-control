"""Plot a trajectory_recorder CSV, colour-coded by the active emotion action,
and print a per-segment summary (distance travelled, net rotation).

    python scripts/plot_trajectory.py evidence/trajectory.csv evidence/trajectory.png
"""

from __future__ import annotations

import csv
import math
import sys
from itertools import groupby
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "arc_left": "#e6a100",
    "arc_right": "#2a6fdb",
    "back_off": "#d62728",
    "nudge_forward": "#2ca02c",
    "spin_360": "#b04fc7",
    "stop": "#9e9e9e",
}
LABELS = {
    "arc_left": "Vui vẻ → cong sang trái",
    "arc_right": "Buồn → cong sang phải",
    "back_off": "Tức giận → lùi nhẹ",
    "nudge_forward": "Sợ hãi → tiến nhẹ",
    "spin_360": "Ngạc nhiên → xoay 1 vòng",
    "stop": "Bình thường → dừng",
}


def load(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t", "x", "y", "yaw", "v", "w"):
            r[k] = float(r[k])
    return rows


def summarize(rows: list[dict]) -> list[tuple[str, float, float, float, float]]:
    """(action, duration, path length, net displacement, |accumulated yaw|) per segment."""
    out = []
    for action, seg in groupby(rows, key=lambda r: r["action"]):
        seg = list(seg)
        if not action or len(seg) < 2:
            continue
        length = sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in zip(seg, seg[1:]))
        disp = math.hypot(seg[-1]["x"] - seg[0]["x"], seg[-1]["y"] - seg[0]["y"])
        yaw = sum(
            abs(math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"])))
            for a, b in zip(seg, seg[1:])
        )
        out.append((action, seg[-1]["t"] - seg[0]["t"], length, disp, yaw))
    return out


def plot(rows: list[dict], dst: Path) -> None:
    """Draw the odometry trajectory coloured by action and save it to ``dst``."""
    fig, ax = plt.subplots(figsize=(7, 7))
    seen = set()
    nudge_offset = {"back_off": -0.03, "nudge_forward": +0.03}  # keep collinear moves apart
    for action, seg in groupby(rows, key=lambda r: r["action"]):
        seg = list(seg)
        if action in ("", "estop", "watchdog_stop"):
            continue
        xs, ys = [r["x"] for r in seg], [r["y"] for r in seg]
        label = LABELS.get(action, action) if action not in seen else None
        seen.add(action)
        color = COLORS.get(action, "k")
        if action == "stop":
            ax.plot(xs[-1:], ys[-1:], "o", color=color, ms=4, label=label)
        elif action == "spin_360":
            ax.plot(xs[0], ys[0], "o", mfc="none", mec=color, mew=3, ms=22, label=label)
            ax.annotate("360°", (xs[0], ys[0]), xytext=(14, 12), textcoords="offset points",
                        color=color, fontweight="bold")
        elif action in nudge_offset:
            # perpendicular offset so a back-off followed by a forward nudge stays visible
            yaw = seg[0]["yaw"] + math.pi / 2
            dx, dy = nudge_offset[action] * math.cos(yaw), nudge_offset[action] * math.sin(yaw)
            ax.annotate("", xy=(xs[-1] + dx, ys[-1] + dy), xytext=(xs[0] + dx, ys[0] + dy),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
            ax.plot([], [], "-", color=color, lw=2.5, label=label)
        else:
            ax.plot(xs, ys, "-", color=color, lw=2.5, label=label)
    ax.plot(rows[0]["x"], rows[0]["y"], "ks", ms=8, label="điểm xuất phát")
    end = rows[-1]
    ax.arrow(end["x"], end["y"], 0.15 * math.cos(end["yaw"]), 0.15 * math.sin(end["yaw"]),
             width=0.01, color="k")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("x [m] (odom)")
    ax.set_ylabel("y [m] (odom)")
    ax.set_title("Quỹ đạo TurtleBot 4 theo biểu cảm khuôn mặt")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(dst, dpi=160)
    plt.close(fig)


def main() -> None:
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".png")
    rows = load(src)
    if not rows:
        sys.exit("empty trajectory")
    plot(rows, dst)
    print(f"saved {dst}")

    print(f"{'action':<15}{'time[s]':>9}{'path[m]':>9}{'disp[m]':>9}{'|yaw|[deg]':>12}")
    for action, dur, length, disp, yaw in summarize(rows):
        print(f"{action:<15}{dur:9.2f}{length:9.3f}{disp:9.3f}{math.degrees(yaw):12.1f}")


if __name__ == "__main__":
    main()
