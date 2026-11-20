"""Plot eval_place.py results: where each trial started (left) and how its cube was rotated (right).

Can also be run on its own to redraw a saved CSV:  python plot_results.py results/<tag>/trials.csv
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from demo_reach import OUTCOMES
from scene import BOX_POS

# Outcome -> (colour, marker). Every outcome has its own marker too, so it never depends on colour alone.
STYLE = {
    "OK": ("#0ca30c", "o"),
    "IK_FAIL": ("#d03b3b", "X"),
    "GRASP_MISS": ("#ec835a", "^"),
    "DROPPED": ("#c98500", "s"),
    "MISSED_BOX": ("#4a3aa7", "D"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_outcomes(rows, path, title, workspace=None):
    fig, (ax_xy, ax_yaw) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.patch.set_facecolor(SURFACE)
    n_ok = sum(r["outcome"] == "OK" for r in rows)
    fig.suptitle(f"Pick-and-place: {n_ok}/{len(rows)} OK   ({title})", color=INK, fontsize=12, x=0.01, ha="left")

    # Left: start positions on the table, seen from above, with the box and the robot base for reference.
    _style_axes(ax_xy)
    if workspace:
        (x0, x1), (y0, y1) = workspace
        ax_xy.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec=INK_2))
    ax_xy.add_patch(Rectangle((BOX_POS[0] - 0.08, BOX_POS[1] - 0.08), 0.16, 0.16, fc="#d8c7b4", ec="#8a6a4a"))
    ax_xy.annotate("box", (BOX_POS[0], BOX_POS[1]), ha="center", va="center", color=INK_2, fontsize=9)
    ax_xy.plot(0, 0, marker="P", ms=10, color=INK_2, ls="none")
    ax_xy.annotate("robot base", (0, 0), (6, 6), textcoords="offset points", color=INK_2, fontsize=9)

    # Right: cube yaw folded into one 90 deg period (a cube looks the same every 90 deg).
    _style_axes(ax_yaw)
    present = [o for o in OUTCOMES if any(r["outcome"] == o for r in rows)]
    for o in present:
        color, marker = STYLE[o]
        sel = [r for r in rows if r["outcome"] == o]
        ax_xy.scatter([r["x"] for r in sel], [r["y"] for r in sel], c=color, marker=marker, s=46,
                      edgecolors=SURFACE, linewidths=1.2, label=f"{o} ({len(sel)})", zorder=3)
        folded = [(float(r["yaw_deg"]) + 45) % 90 - 45 for r in sel]
        ax_yaw.scatter(folded, [present.index(o)] * len(sel), c=color, marker=marker, s=46,
                       edgecolors=SURFACE, linewidths=1.2, alpha=0.9, zorder=3)

    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (m)", color=INK_2)
    ax_xy.set_ylabel("y (m)", color=INK_2)
    ax_xy.set_title("Cube start position", color=INK, fontsize=10, loc="left")
    leg = ax_xy.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(INK)

    ax_yaw.set_yticks(range(len(present)), present)
    ax_yaw.set_ylim(-0.6, len(present) - 0.4)
    ax_yaw.set_xlim(-47, 47)
    ax_yaw.set_xticks([-45, -30, -15, 0, 15, 30, 45])
    ax_yaw.set_xlabel("cube yaw, folded to ±45° (deg)", color=INK_2)
    ax_yaw.set_title("Cube rotation", color=INK, fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


PLANNERS = [(True, "original system", "#eb6834"), (False, "improved system", "#2a78d6")]


def plot_yaw_sweep(rows, path, title):
    """Grasp offset and twist vs cube yaw, one line per system. Failed runs get an X."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f"Yaw sweep: how cleanly does each system grasp?   ({title})", color=INK, fontsize=12,
                 x=0.01, ha="left")
    panels = [("grasp_offset_mm", "cube offset from fingertip centre\nafter closing (mm)"),
              ("grasp_twist_deg", "cube rotated by closing\nthe fingers (deg)")]
    for ax, (key, ylabel) in zip(axes, panels):
        _style_axes(ax)
        for baseline, name, color in PLANNERS:
            sel = [r for r in rows if r["baseline"] == baseline]
            yaws = sorted({r["yaw_deg"] for r in sel})
            vals = [np.array([r[key] for r in sel if r["yaw_deg"] == y], dtype=float) for y in yaws]
            mean = np.array([np.nanmean(v) if np.any(~np.isnan(v)) else np.nan for v in vals])
            std = np.array([np.nanstd(v) if np.any(~np.isnan(v)) else np.nan for v in vals])
            ax.plot(yaws, mean, color=color, lw=2, label=name, zorder=3)
            if len(vals[0]) > 1:  # with --repeats: show the spread over noise samples
                lower = mean - std if key == "grasp_twist_deg" else np.maximum(mean - std, 0)  # a distance is >= 0
                ax.fill_between(yaws, lower, mean + std, color=color, alpha=0.18, lw=0)
            fails = [(y, np.nanmean(v) if np.any(~np.isnan(v)) else 0.0)
                     for y, v in zip(yaws, vals)
                     if any(r["outcome"] != "OK" for r in sel if r["yaw_deg"] == y)]
            if fails:
                ax.scatter(*zip(*fails), marker="X", s=70, color=color, edgecolors=INK, linewidths=0.8, zorder=4,
                           label=f"{name}: a run failed")
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    axes[1].axhline(0, color=INK_2, lw=0.8)
    axes[1].set_xlabel("cube yaw (deg); the cube repeats every 90°", color=INK_2)
    axes[1].set_xticks(range(0, 91, 15))
    leg = axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    csv_path = Path(sys.argv[1])
    with open(csv_path) as f:
        rows = [{**r, "x": float(r["x"]), "y": float(r["y"])} for r in csv.DictReader(f)]
    plot_outcomes(rows, csv_path.with_name("outcomes.png"), csv_path.parent.name)
