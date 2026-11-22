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
    "NOT_SEEN": ("#52514e", "P"),
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


METHOD_STYLE = {"stale": ("#eb6834", "stale calibration (tripod moved, nobody noticed)"),
                "marker": ("#2a78d6", "self-locating camera (4 ArUco tags)")}


def plot_calibration(rows, sweeps, path):
    """Camera calibration experiment: success rate and cube error vs each error, per method."""
    from scene import CAMERA_EYE, CAMERA_TARGET

    fig, axes = plt.subplots(2, len(sweeps), figsize=(12, 6.6), sharex="col")
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Camera calibration error: what the system tolerates (100 trials per point)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    zero = {r["method"]: r for r in rows if r["factor"] == "none"}
    distance = np.linalg.norm(np.subtract(CAMERA_EYE, CAMERA_TARGET))
    for col, (factor, (_, values, unit)) in enumerate(sweeps.items()):
        ax_ok, ax_err = axes[0, col], axes[1, col]
        for ax in (ax_ok, ax_err):
            _style_axes(ax)
        for method, (color, label) in METHOD_STYLE.items():
            pts = [zero[method]] if method in zero else []
            pts += sorted((r for r in rows if r["method"] == method and r["factor"] == factor),
                          key=lambda r: r["magnitude"])
            x = [0.0 if r["factor"] == "none" else r["magnitude"] for r in pts]
            ax_ok.plot(x, [100 * r["success"] for r in pts], color=color, lw=2, marker="o", ms=5, label=label,
                       zorder=3)
            ax_ok.fill_between(x, [100 * r["ci_lo"] for r in pts], [100 * r["ci_hi"] for r in pts], color=color,
                               alpha=0.15, lw=0)
            ax_err.plot(x, [r["perc_err_median"] for r in pts], color=color, lw=2, marker="o", ms=5, zorder=3)
            ax_err.plot(x, [r["perc_err_p90"] for r in pts], color=color, lw=1, ls="--", zorder=3)
        ax_err.axhline(10, color=INK_2, lw=0.8, ls=":")
        ax_err.annotate("~10 mm: limit from the noise sweep", (0, 10), (4, 3), textcoords="offset points",
                        color=INK_2, fontsize=8)
        if factor == "rotation":
            th = np.linspace(0.05, max(values), 50)
            ax_err.plot(th, 1000 * distance * np.tan(np.radians(th)), color=INK_2, lw=1, ls="-.",
                        label="prediction: distance × tan(angle)")
            ax_err.legend(frameon=False, fontsize=8, loc="center right")
        ax_ok.set_title(f"{factor} error", color=INK, fontsize=10, loc="left")
        ax_ok.set_ylim(-5, 105)
        ax_err.set_yscale("log")
        ax_err.set_xlabel(f"camera {factor} error ({unit})", color=INK_2)
    axes[0, 0].set_ylabel("success (%)", color=INK_2)
    axes[1, 0].set_ylabel("cube position error (mm)\nmedian, dashed = 90th percentile", color=INK_2, fontsize=9)
    leg = axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


MOTION_STYLE = {"circle": "#2a78d6", "sway": "#eb6834", "jitter": "#1baf7a"}  # palette slots 1-3


VARIANT_STYLE = {"once": (":", 1.2, "looks once"), "again": ("--", 1.4, "looks again above the cube"),
                 "track": ("-.", 1.8, "tracks the cube"), "predict": ((0, (5, 1, 1, 1)), 1.8, "predicts (v1)"),
                 "predict2": ("-", 2.6, "predicts (v2)")}


def plot_dynamic(rows, motions, path, variants=("once",), title=None):
    """Dynamic targets: success rate and target shift vs hand speed. Colour = hand motion,
    line style = planner variant (dotted: looks once; dashed: looks again; solid: tracks)."""
    variants = list(variants)
    fig, (ax_ok, ax_shift, ax_lag) = plt.subplots(1, 3, figsize=(15, 4.9))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title or "Dynamic targets: a hand moves the cube while the robot grasps it (100 trials per point)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    for ax in (ax_ok, ax_shift, ax_lag):
        _style_axes(ax)
        ax.set_xscale("log")
    for variant in [v for v in ("track", "predict", "predict2") if v in variants]:  # only tracking measures lag
        ls, lw, _ = VARIANT_STYLE[variant]
        for motion in motions:
            pts = sorted((r for r in rows if r.get("variant") == variant and r["motion"] == motion
                          and r["speed_cm_s"] > 0), key=lambda r: r["speed_cm_s"])
            if pts:
                ax_lag.plot([r["speed_cm_s"] for r in pts], [r["lag_median"] for r in pts],
                            color=MOTION_STYLE[motion], ls=ls, lw=lw, marker="o", ms=4, zorder=3)
    lag_pts = [r for r in rows if r.get("variant") == "track" and r["speed_cm_s"] > 0]
    if lag_pts:
        v = np.array(sorted({r["speed_cm_s"] for r in lag_pts}))
        ax_lag.plot(v, 10 * v * 0.2, color=INK_2, lw=0.8, ls="-.")
        ax_lag.annotate("speed × 0.2 s", (v[-1], 20 * v[-1] / 10), (-4, 6), textcoords="offset points", ha="right",
                        color=INK_2, fontsize=8)
    ax_lag.set_yscale("log")
    ax_lag.set_ylabel("gripper behind the cube at grasp height\n(mm, median)", color=INK_2, fontsize=9)
    ax_lag.set_title("Tracking: the arm lags behind", color=INK, fontsize=10, loc="left")
    for variant in variants:
        ls, lw, vlabel = VARIANT_STYLE[variant]
        sel = [r for r in rows if r.get("variant", "once") == variant]
        for motion in motions:
            color = MOTION_STYLE[motion]
            pts = sorted((r for r in sel if r["motion"] == motion), key=lambda r: r["speed_cm_s"])
            x = [r["speed_cm_s"] for r in pts]
            ax_ok.plot(x, [100 * r["success"] for r in pts], color=color, ls=ls, lw=lw, marker="o", ms=4, zorder=3)
            if variant == variants[-1]:
                ax_ok.fill_between(x, [100 * r["ci_lo"] for r in pts], [100 * r["ci_hi"] for r in pts], color=color,
                                   alpha=0.12, lw=0)
            ax_shift.plot(x, [r["shift_median"] for r in pts], color=color, ls=ls, lw=lw, marker="o", ms=4, zorder=3)
        ages = [r["look_age_s"] for r in sel if r["speed_cm_s"] > 0]
        if ages and np.median(ages) > 0.05:  # a straight-line move covers speed x (time since the look)
            v = np.array(sorted({r["speed_cm_s"] for r in sel if r["speed_cm_s"] > 0}))
            age = float(np.median(ages))
            ax_shift.plot(v, 10 * v * age, color=INK_2, ls=ls, lw=0.8, zorder=2)
            ax_shift.annotate(f"speed × {age:.1f} s ({vlabel})", (v[-1], 10 * v[-1] * age), (-4, 4),
                              textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    speeds = sorted({r["speed_cm_s"] for r in rows if r["speed_cm_s"] > 0})
    for ax in (ax_ok, ax_shift, ax_lag):
        ax.set_xticks(speeds, [f"{v:g}" for v in speeds])
        ax.minorticks_off()
        ax.set_xlabel("hand speed (cm/s)", color=INK_2)
    ax_ok.set_ylim(-5, 105)
    ax_ok.set_ylabel("success (%)", color=INK_2)
    ax_ok.set_title("Success", color=INK, fontsize=10, loc="left")
    ax_shift.set_yscale("log")
    ax_shift.set_ylabel("cube distance from the aimed point\nwhen the gripper closes (mm, median)", color=INK_2,
                        fontsize=9)
    ax_shift.set_title("How far the cube moved since the last look", color=INK, fontsize=10, loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MOTION_STYLE[mo], lw=2, label=mo) for mo in motions]
    handles += [Line2D([], [], color=INK_2, ls=VARIANT_STYLE[va][0], lw=VARIANT_STYLE[va][1], label=VARIANT_STYLE[va][2])
                for va in variants]
    leg = fig.legend(handles=handles, frameon=False, fontsize=9, loc="lower center", ncol=len(handles))
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


PREDICT_STYLE = {"track": ("#8a8984", "tracks only"), "predict": ("#eb6834", "predicts, v1"),
                 "predict2": ("#2a78d6", "predicts, v2")}


def plot_prediction(rows, motions, path):
    """Prediction v1 / v2 vs tracking alone: one column per hand motion, success on top, arm lag below."""
    fig, axes = plt.subplots(2, len(motions), figsize=(12, 6.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Prediction: tracking alone vs aiming 0.2 s ahead (100 trials per point)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    speeds = sorted({r["speed_cm_s"] for r in rows if r["speed_cm_s"] > 0})
    for col, motion in enumerate(motions):
        ax_ok, ax_lag = axes[0, col], axes[1, col]
        for ax in (ax_ok, ax_lag):
            _style_axes(ax)
            ax.set_xscale("log")
        for variant, (color, label) in PREDICT_STYLE.items():
            pts = sorted((r for r in rows if r.get("variant") == variant and r["motion"] == motion
                          and r["speed_cm_s"] > 0), key=lambda r: r["speed_cm_s"])
            if not pts:
                continue
            x = [r["speed_cm_s"] for r in pts]
            ax_ok.plot(x, [100 * r["success"] for r in pts], color=color, lw=2, marker="o", ms=4, label=label, zorder=3)
            ax_ok.fill_between(x, [100 * r["ci_lo"] for r in pts], [100 * r["ci_hi"] for r in pts], color=color,
                               alpha=0.12, lw=0)
            ax_lag.plot(x, [r["lag_median"] for r in pts], color=color, lw=2, marker="o", ms=4, zorder=3)
        ax_ok.set_title(motion, color=INK, fontsize=10, loc="left")
        ax_ok.set_ylim(50, 102)
        ax_lag.set_yscale("log")
        ax_lag.set_xticks(speeds, [f"{v:g}" for v in speeds])
        ax_lag.minorticks_off()
        ax_lag.set_xlabel("hand speed (cm/s)", color=INK_2)
    axes[0, 0].set_ylabel("success (%)", color=INK_2)
    axes[1, 0].set_ylabel("arm behind the cube\nat grasp height (mm, median)", color=INK_2, fontsize=9)
    leg = axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
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
