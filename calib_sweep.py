"""Camera calibration experiment: how much camera error can the system tolerate, and does a
self-locating camera (ArUco tags on the table) remove the problem?

The real scene camera is bumped (rotated, shifted) and its focal length is believed wrongly,
one factor at a time, while perception believes either
    stale   the camera pose from before the bump (a knocked tripod nobody noticed), or
    marker  the camera pose located from the four ArUco tags in each image.
ideal (perception knows the true camera) runs once, as the control.

    python calib_sweep.py            # run missing settings (resumable), then summarize and plot
    python calib_sweep.py --plot     # only summarize and plot existing results

Results: results/calibration/<setting>/ (trials.csv, run_info.json), summary.csv, calibration.png
"""
import argparse
import csv
import subprocess
import sys

import numpy as np

from eval_place import ROOT, wilson_interval

OUT = ROOT / "results" / "calibration"
TRIALS = 100
SWEEPS = {  # factor -> (eval_place.py flag, magnitudes, unit)
    "rotation": ("--camera-rot", [0.25, 0.5, 1, 2, 3], "deg"),
    "shift": ("--camera-shift", [2, 5, 10, 20], "mm"),
    "focal": ("--focal-err", [1, 2, 5], "%"),
}
METHODS = ["stale", "marker"]
TOLERANCE = 0.95  # a setting is "tolerated" if at least this share of trials succeeds


def settings():
    """(name, method, factor, magnitude, extra eval_place.py args) for every run of the sweep."""
    yield "ideal_none0", "ideal", "none", 0.0, []
    for method in METHODS:
        yield f"{method}_none0", method, "none", 0.0, []
        for factor, (flag, values, _) in SWEEPS.items():
            for v in values:
                yield f"{method}_{factor}{v:g}", method, factor, float(v), [flag, str(v)]
        # A realistic knock: all three at once.
        yield (f"{method}_combined", method, "combined", 1.0,
               ["--camera-rot", "1", "--camera-shift", "5", "--focal-err", "2"])


def run_missing(force=False):
    for name, method, *_, extra in settings():
        if (OUT / name / "trials.csv").exists() and not force:
            continue
        cmd = [sys.executable, "eval_place.py", "--perception", "camera", "--calibration", method,
               "--trials", str(TRIALS), "--tag", f"calibration/{name}", *extra]
        print("running", name, flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def summarize():
    rows = []
    for name, method, factor, magnitude, _ in settings():
        path = OUT / name / "trials.csv"
        if not path.exists():
            continue
        with open(path) as f:
            trials = list(csv.DictReader(f))
        ok = sum(t["outcome"] == "OK" for t in trials)
        lo, hi = wilson_interval(ok, len(trials))
        err = np.array([float(t["perc_pos_err_mm"]) for t in trials])
        err = err[~np.isnan(err)]
        num = lambda k: np.nanmean([float(t[k]) for t in trials]) if trials and k in trials[0] else np.nan
        rows.append(dict(setting=name, method=method, factor=factor, magnitude=magnitude, trials=len(trials),
                         ok=ok, success=ok / len(trials), ci_lo=lo, ci_hi=hi,
                         perc_err_median=np.median(err) if len(err) else np.nan,
                         perc_err_p90=np.percentile(err, 90) if len(err) else np.nan,
                         calib_rot_err_deg=num("calib_rot_err_deg"), calib_pos_err_mm=num("calib_pos_err_mm"),
                         tags_seen=num("tags_seen"),
                         not_seen=sum(t["outcome"] == "NOT_SEEN" for t in trials)))
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    return rows


def print_table(rows):
    print(f"\n{'setting':22s} {'success':>9s} {'95% CI':>10s} {'cube err median/p90 (mm)':>26s} "
          f"{'belief err (deg, mm)':>21s} {'tags':>5s}")
    for r in rows:
        print(f"{r['setting']:22s} {r['ok']:>4d}/{r['trials']:<4d} {100 * r['ci_lo']:4.0f}-{100 * r['ci_hi']:3.0f}% "
              f"{r['perc_err_median']:12.1f} / {r['perc_err_p90']:5.1f}       "
              f"{r['calib_rot_err_deg']:6.2f}, {r['calib_pos_err_mm']:6.1f}   {r['tags_seen']:4.1f}")
    print(f"\nTolerance (largest error with success >= {TOLERANCE:.0%}):")
    for method in METHODS:
        parts = []
        for factor, (_, values, unit) in SWEEPS.items():
            good = [r["magnitude"] for r in rows if r["method"] == method and r["factor"] == factor
                    and r["success"] >= TOLERANCE]
            bad = [r["magnitude"] for r in rows if r["method"] == method and r["factor"] == factor
                   and r["success"] < TOLERANCE]
            first_bad = min(bad) if bad else None
            tol = max([g for g in good if first_bad is None or g < first_bad], default=0.0)
            parts.append(f"{factor} {tol:g} {unit}" + ("" if first_bad is None else f" (fails at {first_bad:g})"))
        print(f"  {method:7s} " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", action="store_true", help="don't run anything; summarize existing results")
    parser.add_argument("--force", action="store_true", help="re-run settings that already have results")
    args = parser.parse_args()
    if not args.plot:
        run_missing(args.force)
    rows = summarize()
    print_table(rows)
    from plot_results import plot_calibration  # matplotlib is only needed here
    plot_calibration(rows, SWEEPS, OUT / "calibration.png")
    print(f"\nsaved {OUT / 'summary.csv'} and {OUT / 'calibration.png'}")


if __name__ == "__main__":
    main()
