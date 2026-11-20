"""Controller comparison figures for the README (improved planner, 4 control set-ups).

    python plot_controllers.py

1. results/controllers/comparison.png: reach error and grasp offset (mean ± std) from the
   eval_place.py runs below, without noise and with 10 mm / 10° perception noise.
2. results/controllers/tracking.png: gripper height over time while it comes down to the
   cube and closes, for one pick (cube at the default spot). Shows the gravity sag and the
   motion lag, and when the gripper closes.

The eval runs must exist first (see README, "Controller"):
    eval_place.py [--controller pd] [--no-arrive] [--pos-noise 10 --yaw-noise 10]
"""
import csv

import mujoco
import numpy as np

from demo_reach import Stepper, make_plan, run_plan, wait_until_arrived
from eval_place import ROOT
from plot_results import INK, INK_2, SURFACE, _style_axes, plt
from scene import TCP_SITE, build_model, place_cube

PD, GRAV = "#eb6834", "#2a78d6"  # validated categorical pair (orange, blue)
# (label, controller, waits for arrival, results tag stem, colour, line style)
SETUPS = [
    ("PD, closes on schedule", "pd", False, "fixed_default_pd_noarrive", PD, "--"),
    ("PD + wait to arrive", "pd", True, "fixed_default_pd", PD, "-"),
    ("PD + gravity comp, on schedule", "pd_gravity", False, "fixed_default_noarrive", GRAV, "--"),
    ("PD + gravity comp + wait", "pd_gravity", True, "fixed_default", GRAV, "-"),
]
NOISY = {"fixed_default": "sweep_fixed_10"}  # the 10 mm run of the final set-up is the noise sweep


def load(tag):
    with open(ROOT / "results" / tag / "trials.csv") as f:
        return list(csv.DictReader(f))


def stat(rows, key):
    v = np.array([float(r[key]) for r in rows])
    v = v[~np.isnan(v)]
    return v.mean(), v.std()


def plot_comparison(path):
    panels = [("reach_error_mm", None, "Reach error at grasp height\n(no noise)"),
              ("grasp_offset_mm", None, "Grasp offset\n(no noise)"),
              ("grasp_offset_mm", "noise10mm10deg", "Grasp offset\n(10 mm / 10° perception noise)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle("Control set-ups, improved planner, 100 trials each (mean ± std)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    x = np.arange(len(SETUPS))
    for ax, (key, noise, title) in zip(axes, panels):
        _style_axes(ax)
        for i, (label, _, wait, tag, color, _) in enumerate(SETUPS):
            if noise:
                tag = NOISY.get(tag, f"{tag}_{noise}")
            mean, std = stat(load(tag), key)
            ax.bar(i, mean, 0.7, color=color, alpha=1.0 if wait else 0.45, edgecolor=SURFACE, linewidth=2,
                   yerr=std, error_kw=dict(ecolor=INK_2, lw=1, capsize=3), zorder=3)
            ax.annotate(f"{mean:.1f}", (i, mean + std), (0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xticks(x, ["PD\nschedule", "PD\nwait", "gravity\nschedule", "gravity\nwait"], fontsize=8)
        ax.set_ylabel("mm", color=INK_2)
        ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


class RecordingStepper(Stepper):
    """A Stepper that also records the gripper's height above the cube and when it starts closing."""

    def __init__(self, model, data, controller):
        super().__init__(model, data, controller=controller)
        self.cube_z = data.body("cube").xpos[2]
        self.t, self.z, self.closed_at = [], [], None

    def __call__(self, q_cmd, grip):
        super().__call__(q_cmd, grip)
        self.t.append(self.data.time)
        self.z.append(1000 * (self.data.site(TCP_SITE).xpos[2] - self.cube_z))
        if grip > 0 and self.closed_at is None:
            self.closed_at = self.data.time


def record_approach(controller, wait):
    """Run 'above cube' -> 'down to cube' -> 'close gripper' and record TCP height error vs time."""
    model, data = build_model()
    place_cube(model, data, (0.45, 0.20), 0.0)
    step = RecordingStepper(model, data, controller)
    scratch = mujoco.MjData(model)
    plan, ok, _ = make_plan(model, data, data.body("cube").xpos.copy(), 0.0)

    def arrive(q, grip):
        wait_until_arrived(model, data, step, q, grip, scratch)

    for _ in run_plan(model, data, plan[:3], step, arrive if wait else None):
        pass
    return np.array(step.t), np.array(step.z), step.closed_at


def plot_tracking(path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    for label, controller, wait, _, color, ls in SETUPS:
        t, z, t_close = record_approach(controller, wait)
        keep = t > 2.85  # the descent towards the cube and the close
        ax.plot(t[keep], z[keep], color=color, ls=ls, lw=2, label=label, zorder=3)
        k = np.searchsorted(t, t_close)
        ax.scatter([t_close], [z[k]], marker="v", s=60, color=color, edgecolors=INK, linewidths=0.8, zorder=4)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.annotate("target: cube centre", (ax.get_xlim()[0], 0), (6, 4), textcoords="offset points",
                color=INK_2, fontsize=9)
    ax.set_xlabel("time (s)", color=INK_2)
    ax.set_ylabel("gripper height above cube centre (mm)", color=INK_2)
    ax.set_ylim(-20, 60)
    ax.set_title("Coming down to the cube. ▼ = gripper starts closing", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "results" / "controllers"
    out.mkdir(parents=True, exist_ok=True)
    plot_comparison(out / "comparison.png")
    plot_tracking(out / "tracking.png")
    print(f"saved {out}/comparison.png and tracking.png")
