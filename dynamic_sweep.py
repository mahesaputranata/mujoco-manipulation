"""Dynamic targets: how fast may the cube move while the robot tries to grasp it?

A hand holds the cube and moves it (hand_motion.py: circle, sway, jitter) at a range of speeds,
while the improved system (camera perception, gravity-compensated control) picks it. Planner
variants, each compared on the same trials:
    once    looks once per attempt, then moves blindly (the baseline)
    again   looks again once the gripper is above the cube and re-plans the descent (Option A)

    python dynamic_sweep.py            # run missing settings (resumable), then summarize and plot
    python dynamic_sweep.py --plot     # only summarize and plot existing results

Results: results/dynamic/<setting>/ (trials.csv, run_info.json), summary.csv, dynamic.png
"""
import argparse
import csv
import subprocess
import sys

import numpy as np

from eval_place import ROOT, wilson_interval

OUT = ROOT / "results" / "dynamic"
TRIALS = 100
MOTIONS = ["circle", "sway", "jitter"]
SPEEDS = [0.25, 0.5, 1, 2, 5, 10]  # cm/s
TOLERANCE = 0.95
VARIANTS = {"once": [], "again": ["--look-again"]}  # planner variant -> extra eval_place.py flags


def settings():
    """(name, variant, motion, speed). The look-once runs keep their original names."""
    for variant in VARIANTS:
        prefix = "" if variant == "once" else f"{variant}_"
        yield f"{prefix}static0", variant, "static", 0.0
        for motion in MOTIONS:
            for v in SPEEDS:
                yield f"{prefix}{motion}{v:g}", variant, motion, float(v)


def run_missing(force=False):
    for name, variant, motion, speed in settings():
        if (OUT / name / "trials.csv").exists() and not force:
            continue
        cmd = [sys.executable, "eval_place.py", "--perception", "camera", "--motion", motion, "--speed", str(speed),
               "--trials", str(TRIALS), "--tag", f"dynamic/{name}", *VARIANTS[variant]]
        print("running", name, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def summarize():
    rows = []
    for name, variant, motion, speed in settings():
        path = OUT / name / "trials.csv"
        if not path.exists():
            continue
        with open(path) as f:
            trials = list(csv.DictReader(f))
        ok = sum(t["outcome"] == "OK" for t in trials)
        lo, hi = wilson_interval(ok, len(trials))
        col = lambda k: np.array([float(t[k]) for t in trials if t.get(k) not in (None, "", "nan")])
        shift, age = col("target_shift_mm"), col("look_age_s")
        rows.append(dict(setting=name, variant=variant, motion=motion, speed_cm_s=speed, trials=len(trials), ok=ok,
                         success=ok / len(trials), ci_lo=lo, ci_hi=hi,
                         shift_median=np.median(shift), shift_p90=np.percentile(shift, 90),
                         look_age_s=np.median(age), regrasps=sum(int(t["attempts"]) > 1 for t in trials),
                         regrasp_ok=sum(int(t["attempts"]) > 1 and t["outcome"] == "OK" for t in trials)))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    return rows


def print_table(rows):
    print(f"\n{'setting':16s} {'success':>9s} {'95% CI':>10s} {'shift median/p90 (mm)':>23s} {'look age':>9s} "
          f"{'re-grasps (ok)':>15s}")
    for r in rows:
        print(f"{r['setting']:16s} {r['ok']:>4d}/{r['trials']:<4d} {100 * r['ci_lo']:4.0f}-{100 * r['ci_hi']:3.0f}% "
              f"{r['shift_median']:11.1f} / {r['shift_p90']:5.1f}     {r['look_age_s']:5.2f} s   "
              f"{r['regrasps']:5d} ({r['regrasp_ok']})")
    print(f"\nFastest speed with success >= {TOLERANCE:.0%} (all slower speeds too):")
    for variant in VARIANTS:
        parts = []
        for motion in MOTIONS:
            pts = sorted((r for r in rows if r["motion"] == motion and r["variant"] == variant),
                         key=lambda r: r["speed_cm_s"])
            ok_speed = 0.0
            for r in pts:
                if r["success"] < TOLERANCE:
                    break
                ok_speed = r["speed_cm_s"]
            parts.append(f"{motion} {ok_speed:g} cm/s")
        print(f"  look {variant:5s}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true", help="don't run anything; summarize existing results")
    parser.add_argument("--force", action="store_true", help="re-run settings that already have results")
    args = parser.parse_args()
    if not args.plot:
        run_missing(args.force)
    rows = summarize()
    print_table(rows)
    from plot_results import plot_dynamic  # matplotlib is only needed here
    plot_dynamic(rows, MOTIONS, OUT / "dynamic.png", VARIANTS)
    print(f"\nsaved {OUT / 'summary.csv'} and {OUT / 'dynamic.png'}")


if __name__ == "__main__":
    main()
