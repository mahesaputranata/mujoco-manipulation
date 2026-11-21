"""Step 2b: run many pick-and-place trials with random cube poses and report the success rate.

Every trial i gets its own random generator seeded with (seed, i), so a trial is
reproducible on its own: `--replay 17` shows exactly trial 17 in the viewer.

The simulation is deterministic: the same trial always gives the same outcome. With
perception noise, `--repeats K` re-runs every cube pose with K different noise samples,
which gives each pose its own success rate. Poses in between 0% and 100% are the
fragile ones: whether they succeed depends on luck in the perception error.

    python eval_place.py                                  # 100 trials, all fixes on
    python eval_place.py --baseline                       # original system: planner without the fixes, plain PD
    python eval_place.py --pos-noise 5 --yaw-noise 5      # imperfect "camera" (mm, deg)
    python eval_place.py --rand-physics --workspace wide  # harder: random cube size/mass/friction, wider area
    python eval_place.py --pos-noise 15 --yaw-noise 15 --repeats 10   # per-pose success rate
    python eval_place.py --controller pd                  # improved planner, but plain PD control
    python eval_place.py --yaw-sweep                      # cube at one spot, yaw 0..90 deg, both planners
    python eval_place.py --replay 17                      # watch trial 17 (use the same flags as the eval)
    python render_trials.py ...                           # the same trials as GIFs (see that file)

Results go to results/<tag>/: trials.csv and outcomes.png (+ poses.csv with --repeats).
Besides pass/fail, every trial logs grasp offset, grasp twist and placement error (see
demo_reach.pick_and_place); the summary prints their mean +- std.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import os
import sys
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from controllers import CONTROLLERS
from demo_reach import METRICS, OUTCOMES, ground_truth, make_stepper, pick_and_place, run_in_viewer, wrap_cube_angle
from scene import BOX_POS, CUBE_HALF, CUBE_MASS, SCENE_CAMERA, build_model, cube_yaw, place_cube

ROOT = Path(__file__).parent

# Cube spawn areas (x range, y range) in metres. The robot base sits at the origin.
WORKSPACES = {
    "default": ((0.30, 0.60), (-0.05, 0.35)),
    "wide": ((0.15, 0.70), (-0.25, 0.55)),
}
MIN_RADIUS, MAX_RADIUS = 0.25, 0.75  # stay clear of the base and inside the arm's comfortable reach
BOX_CLEARANCE = 0.13                 # keep the cube out of (and off the walls of) the box


def sample_trial(i, args, repeat=0):
    """Everything random about trial i: start pose, cube physics, and the perception-noise seed.

    Repeats keep the pose and physics and only draw a new perception-noise seed.
    """
    rng = np.random.default_rng([args.seed, i])
    (x_lo, x_hi), (y_lo, y_hi) = WORKSPACES[args.workspace]
    while True:
        x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        near_box = abs(x - BOX_POS[0]) < BOX_CLEARANCE and abs(y - BOX_POS[1]) < BOX_CLEARANCE
        if MIN_RADIUS < np.hypot(x, y) < MAX_RADIUS and not near_box:
            break
    trial = dict(trial=i, x=x, y=y, yaw_deg=rng.uniform(-180, 180),
                 half=CUBE_HALF, mass=CUBE_MASS, friction=1.0, noise_seed=int(rng.integers(2**31)))
    if args.rand_physics:
        trial.update(half=rng.uniform(0.018, 0.030), mass=rng.uniform(0.05, 0.30), friction=rng.uniform(0.4, 1.2))
    if repeat:
        trial["noise_seed"] = int(np.random.default_rng([args.seed, i, repeat]).integers(2**31))
    trial["repeat"] = repeat
    if args.camera_rot or args.camera_shift or args.focal_err:
        # Camera calibration error: magnitudes from the flags, directions from their own seed, so
        # trial i bumps the camera the same way in every method (ideal / stale / marker).
        trial.update(cam_rot_deg=args.camera_rot, cam_shift_mm=args.camera_shift, focal_err_pct=args.focal_err,
                     bump_seed=int(np.random.default_rng([args.seed, i, 1000]).integers(2**31)))
    if args.motion:
        # Dynamic target: a hand holds the cube and moves it (hand_motion.py). Its own seed, so the
        # pattern's direction and phase are the same for trial i at every speed.
        trial.update(motion=args.motion, speed_cm_s=args.speed,
                     motion_seed=int(np.random.default_rng([args.seed, i, 2000]).integers(2**31)))
    return trial


PERCEPTION_METRICS = ["perc_pos_err_mm", "perc_yaw_err_deg", "perc_z_err_mm", "perc_glitches",
                      "calib_pos_err_mm", "calib_rot_err_deg", "tags_seen"]


def bump_camera(model, data, trial):
    """Move the real scene camera as if the tripod was knocked: rotate it about its own position
    (random axis) and shift it (random direction), by the trial's magnitudes. Returns the focal-length
    factor that perception believes (1 + error, random sign): the lens was calibrated imperfectly."""
    rng = np.random.default_rng(trial["bump_seed"])
    axis, direction = rng.normal(size=3), rng.normal(size=3)
    sign = rng.choice([-1.0, 1.0])
    cam = model.camera(SCENE_CAMERA).id
    q_bump, q_new = np.empty(4), np.empty(4)
    mujoco.mju_axisAngle2Quat(q_bump, axis / np.linalg.norm(axis), np.radians(trial["cam_rot_deg"]))
    mujoco.mju_mulQuat(q_new, q_bump, model.cam_quat[cam])  # rotation in the world frame
    model.cam_quat[cam] = q_new
    model.cam_pos[cam] += direction / np.linalg.norm(direction) * trial["cam_shift_mm"] / 1000
    mujoco.mj_forward(model, data)
    return 1.0 + sign * trial["focal_err_pct"] / 100


class Perceiver:
    """The planner's eyes for one trial: the oracle or the camera, plus optional Gaussian noise.

    oracle: the simulator's exact cube pose (what the planner used before perception existed).
    camera: perception.CameraPerception, estimated from the fixed RGB camera's image.
    --pos-noise / --yaw-noise add error on top of either.

    For the trial's first look it also logs how far the estimate is from the truth, so perception
    accuracy is measured on every trial, for the oracle-with-noise and the camera alike.
    """

    def __init__(self, model, args, rng, believed=None, f_belief=None):
        self.model, self.args, self.rng = model, args, rng
        self.camera = None
        if args.perception == "camera":
            from perception import CameraPerception  # needs OpenCV and an OpenGL context
            self.camera = CameraPerception(model, calibration=args.calibration, believed=believed, f_belief=f_belief)
        self.first = None

    def __call__(self, data):
        if self.camera is not None:
            seen = self.camera(data)
            if seen is None:
                return None
            pos, yaw = seen
        else:
            pos, yaw = ground_truth(data)
        pos = pos.copy()
        pos[:2] += self.rng.normal(0, self.args.pos_noise / 1000, 2)
        yaw = yaw + self.rng.normal(0, np.radians(self.args.yaw_noise))
        if self.first is None:
            true_pos, true_yaw = ground_truth(data)
            self.first = dict(perc_pos_err_mm=1000 * np.linalg.norm(pos[:2] - true_pos[:2]),
                              perc_yaw_err_deg=np.degrees(wrap_cube_angle(yaw - true_yaw)),
                              perc_z_err_mm=1000 * (pos[2] - true_pos[2]))
            if self.camera is not None:  # how wrong was the camera belief behind this estimate?
                from perception import PinholeCamera
                true_cam, belief = PinholeCamera(self.model, data), self.camera.belief
                cos = np.clip((np.trace(belief.R.T @ true_cam.R) - 1) / 2, -1, 1)
                self.first.update(calib_pos_err_mm=1000 * np.linalg.norm(belief.pos - true_cam.pos),
                                  calib_rot_err_deg=np.degrees(np.arccos(cos)), tags_seen=self.camera.tags)
        return pos, yaw

    def metrics(self):
        m = {k: np.nan for k in PERCEPTION_METRICS}
        m.update(self.first or {})
        m["perc_glitches"] = self.camera.glitches if self.camera is not None else np.nan
        return m

    def close(self):
        if self.camera is not None:
            self.camera.close()


def setup_trial(trial, args):
    """Build the scene for this trial (a fresh compile is ~0.05 s) and the pick_and_place settings."""
    model, data = build_model(cube_half=trial["half"], cube_mass=trial["mass"], cube_friction=trial["friction"],
                              with_hand="motion" in trial)
    place_cube(model, data, (trial["x"], trial["y"]), np.radians(trial["yaw_deg"]))
    hooks = []
    if "motion" in trial:
        from hand_motion import HandMotion
        hooks.append(HandMotion(model, data, trial["motion"], trial["speed_cm_s"], trial["motion_seed"]))
    rng = np.random.default_rng(trial["noise_seed"])
    believed = f_belief = None
    if args.perception == "camera":
        from perception import PinholeCamera
        believed = PinholeCamera(model, data)  # the camera as calibrated, before any bump
        f_belief = believed.f
        if "bump_seed" in trial:
            f_belief = believed.f * bump_camera(model, data, trial)
            believed.f = f_belief
    baseline = trial.get("baseline", args.baseline)  # the yaw sweep runs both planners in one go
    perceive = Perceiver(model, args, rng, believed, f_belief)
    return model, data, dict(perceive=perceive, hooks=hooks, align=not baseline,
                             retries=0 if baseline else args.retries,
                             ik_restarts=0 if baseline else args.ik_restarts,
                             arrive=not (baseline or args.no_arrive))


def controller_for(trial, args):
    """The original system uses plain PD. The improved one uses --controller (default: pd_gravity)."""
    if trial.get("baseline", args.baseline):
        return "pd"
    return args.controller or "pd_gravity"


_worker = {}


def _init_worker(args):
    _worker["args"] = args


def run_trial(trial):
    model, data, kwargs = setup_trial(trial, _worker["args"])
    t0 = time.perf_counter()
    metrics = {}
    hooks = kwargs.pop("hooks")
    step = make_stepper(model, data, controller=controller_for(trial, _worker["args"]), hooks=hooks)
    try:
        outcome, attempts = pick_and_place(model, data, step, metrics=metrics, **kwargs)
    finally:
        kwargs["perceive"].close()  # free the camera's OpenGL context: one per trial would leak
    metrics.update(kwargs["perceive"].metrics())
    if hooks:  # dynamic target: did the hand hand the cube over?
        metrics["handed_over"] = int(not hooks[0].held)
    return {**trial, "outcome": outcome, "attempts": attempts, **metrics, "seconds": time.perf_counter() - t0,
            "final_x": data.body("cube").xpos[0], "final_y": data.body("cube").xpos[1],
            "final_yaw_deg": np.degrees(cube_yaw(data))}


def run_all(trials, args, rounds=3):
    """Run trials in parallel worker processes and return their rows in the original order.

    A worker process can die in native code (camera rendering has aborted inside the OpenGL
    driver on a laptop under load). multiprocessing.Pool then waits forever for the lost result;
    ProcessPoolExecutor reports it instead. The unfinished trials are re-run in a fresh pool, up
    to `rounds` times, and the number of crashed pools is reported rather than hidden.
    """
    rows, todo, crashes = {}, list(range(len(trials))), 0
    for _ in range(rounds):
        try:
            with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
                futures = {pool.submit(run_trial, trials[i]): i for i in todo}
                for fut in as_completed(futures):
                    rows[futures[fut]] = fut.result()
        except BrokenProcessPool:
            crashes += 1
        todo = [i for i in range(len(trials)) if i not in rows]
        if not todo:
            break
    if todo:
        raise RuntimeError(f"{len(todo)} trials never finished after {crashes} worker crashes")
    if crashes:
        print(f"  note: {crashes} worker crash(es); their unfinished trials were re-run")
    return [rows[i] for i in range(len(trials))]


def default_tag(args):
    parts = ["baseline" if args.baseline else "fixed", args.workspace]
    if not args.baseline and args.controller not in (None, "pd_gravity"):
        parts.append(args.controller)
    if args.no_arrive and not args.baseline:
        parts.append("noarrive")
    if args.pos_noise or args.yaw_noise:
        parts.append(f"noise{args.pos_noise:g}mm{args.yaw_noise:g}deg")
    if args.rand_physics:
        parts.append("randphys")
    if args.motion:
        parts.append(f"{args.motion}{args.speed:g}cms")
    if args.perception == "camera":
        parts.append("camera")
        if args.calibration != "ideal":
            parts.append(args.calibration)
        for flag, unit in (("camera_rot", "deg"), ("camera_shift", "mm"), ("focal_err", "pct")):
            if getattr(args, flag):
                parts.append(f"{flag.replace('camera_', '')}{getattr(args, flag):g}{unit}")
    return "_".join(parts)


def wilson_interval(k, n, z=1.96):
    """95% confidence interval for a success rate of k out of n.

    100/100 does not mean "always works": it means the true rate is very likely above ~96%.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def per_pose(rows):
    """One row per cube pose: how often it succeeded over its noise repeats."""
    poses = {}
    for r in rows:
        p = poses.setdefault(r["trial"], {k: r[k] for k in ("trial", "x", "y", "yaw_deg", "half", "mass", "friction")}
                             | {"runs": 0, "ok": 0, "outcomes": Counter()})
        p["runs"] += 1
        p["ok"] += r["outcome"] == "OK"
        p["outcomes"][r["outcome"]] += 1
    out = []
    for p in poses.values():
        p["ok_rate"] = p["ok"] / p["runs"]
        p["outcomes"] = " ".join(f"{o}:{n}" for o, n in sorted(p["outcomes"].items()))
        out.append(p)
    return out


def write_run_info(path, args, wall):
    """Next to every results CSV: exactly how it was produced, so a baseline stays traceable."""
    import json
    import platform
    import subprocess
    from datetime import datetime, timezone

    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import cv2
    info = dict(command=" ".join(["eval_place.py"] + sys.argv[1:]), args=vars(args),
                git_commit=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--", "*.py")),
                created=datetime.now(timezone.utc).isoformat(timespec="seconds"), wall_seconds=round(wall, 1),
                versions=dict(python=platform.python_version(), mujoco=mujoco.__version__, numpy=np.__version__,
                              opencv=cv2.__version__))
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def print_summary(rows, tag, wall):
    counts = Counter(r["outcome"] for r in rows)
    n = len(rows)
    lo, hi = wilson_interval(counts["OK"], n)
    print(f"\n{tag}: {counts['OK']}/{n} OK ({100 * counts['OK'] / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)   "
          f"wall time {wall:.1f}s, {np.mean([r['seconds'] for r in rows]):.2f}s per trial")
    for o in OUTCOMES:
        if counts[o]:
            ids = [r["trial"] if not r["repeat"] else f'{r["trial"]}.{r["repeat"]}' for r in rows if r["outcome"] == o]
            shown = ", ".join(map(str, ids[:12])) + (" ..." if len(ids) > 12 else "")
            print(f"  {o:<11} {counts[o]:>4}" + ("" if o == "OK" else f"   trials: {shown}"))
    for k in METRICS + PERCEPTION_METRICS:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v):
            print(f"  {k:<16} {np.mean(v):6.1f} ± {np.std(v):4.1f}   (max {np.max(np.abs(v)):5.1f}, n={len(v)})")
    retried = sum(r["attempts"] > 1 for r in rows)
    if retried:
        print(f"  needed a re-grasp: {retried} (of which OK: {sum(r['attempts'] > 1 and r['outcome'] == 'OK' for r in rows)})")


def print_pose_summary(poses):
    rates = np.array([p["ok_rate"] for p in poses])
    fragile = sorted((p for p in poses if 0 < p["ok_rate"] < 1), key=lambda p: abs(p["ok_rate"] - 0.5))
    print(f"  per pose ({poses[0]['runs']} noise samples each): always OK {np.sum(rates == 1)}, "
          f"never OK {np.sum(rates == 0)}, fragile {len(fragile)}")
    for p in fragile[:8]:
        print(f"    trial {p['trial']:>3}: {100 * p['ok_rate']:3.0f}% OK   x={p['x']:.2f} y={p['y']:.2f} "
              f"yaw={p['yaw_deg']:+.0f}   {p['outcomes']}")


def yaw_sweep_trials(args):
    """The cube at one fixed spot, yaw stepped over one 90 deg period, for both planners."""
    trials = []
    for baseline in (True, False):
        for k, yaw in enumerate(np.arange(0, 90 + 1e-9, args.sweep_step)):
            for r in range(args.repeats):
                trials.append(dict(trial=k, x=args.sweep_xy[0], y=args.sweep_xy[1], yaw_deg=float(yaw),
                                   half=CUBE_HALF, mass=CUBE_MASS, friction=1.0, repeat=r, baseline=baseline,
                                   noise_seed=int(np.random.default_rng([args.seed, k, r]).integers(2**31))))
    return trials


def add_scenario_args(parser):
    """The flags that define which trials are run. Shared with render_trials.py."""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", action="store_true", help="original behaviour: no yaw align, re-grasp or IK restarts")
    parser.add_argument("--retries", type=int, default=1, help="re-grasp attempts after a missed grasp")
    parser.add_argument("--ik-restarts", type=int, default=8, help="extra IK starting poses if IK gets stuck")
    parser.add_argument("--workspace", choices=WORKSPACES, default="default")
    parser.add_argument("--pos-noise", type=float, default=0.0, help="perception noise, std dev in mm")
    parser.add_argument("--yaw-noise", type=float, default=0.0, help="perception noise, std dev in deg")
    parser.add_argument("--rand-physics", action="store_true", help="random cube size, mass and friction")
    parser.add_argument("--perception", choices=["oracle", "camera"], default="oracle",
                        help="where the cube pose comes from: the simulator, or the fixed RGB camera")
    parser.add_argument("--calibration", choices=["ideal", "stale", "marker"], default="ideal",
                        help="camera perception's belief about the camera: the truth, the pose before the "
                             "bump, or located from the ArUco tags")
    parser.add_argument("--camera-rot", type=float, default=0.0, help="bump the real camera by this many degrees")
    parser.add_argument("--camera-shift", type=float, default=0.0, help="shift the real camera by this many mm")
    parser.add_argument("--focal-err", type=float, default=0.0, help="believed focal length off by this many %%")
    parser.add_argument("--motion", choices=["static", "circle", "sway", "jitter"],
                        help="dynamic target: a hand holds the cube and moves it (hand_motion.py)")
    parser.add_argument("--speed", type=float, default=0.0, help="hand speed in cm/s (with --motion)")
    parser.add_argument("--controller", choices=CONTROLLERS,
                        help="arm controller for the improved planner (default pd_gravity; --baseline always uses pd)")
    parser.add_argument("--no-arrive", action="store_true",
                        help="close/open the gripper on schedule instead of waiting for the arm to arrive")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(parser)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1, help="noise samples per cube pose (needs noise)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--tag", help="results sub-folder name (default: built from the flags)")
    parser.add_argument("--replay", type=int, metavar="TRIAL", help="watch one trial in the 3D viewer")
    parser.add_argument("--repeat", type=int, default=0, help="with --replay: which noise sample to show")
    parser.add_argument("--yaw-sweep", action="store_true", help="fixed cube position, yaw 0..90 deg, both planners")
    parser.add_argument("--sweep-xy", type=float, nargs=2, default=(0.45, 0.20), metavar=("X", "Y"))
    parser.add_argument("--sweep-step", type=float, default=1.0, help="yaw step in degrees")
    args = parser.parse_args()

    if args.replay is not None:
        trial = sample_trial(args.replay, args, args.repeat)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in trial.items()})
        model, data, kwargs = setup_trial(trial, args)
        run_in_viewer(model, data, controller_for(trial, args), **kwargs, log=print)
        return

    if args.repeats > 1 and not (args.pos_noise or args.yaw_noise):
        parser.error("--repeats only makes sense with noise: without it every repeat is identical")
    if args.yaw_sweep:
        tag = args.tag or "yawsweep" + default_tag(args).removeprefix("baseline").removeprefix("fixed") + \
            (f"_x{args.repeats}" if args.repeats > 1 else "")
        trials = yaw_sweep_trials(args)
    else:
        tag = args.tag or default_tag(args) + (f"_x{args.repeats}" if args.repeats > 1 else "")
        trials = [sample_trial(i, args, r) for i in range(args.trials) for r in range(args.repeats)]
    t0 = time.perf_counter()
    rows = run_all(trials, args)
    wall = time.perf_counter() - t0

    out = ROOT / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "trials.csv")
    write_run_info(out / "run_info.json", args, wall)
    if args.yaw_sweep:
        for baseline, name in ((True, "original system"), (False, "improved system")):
            print_summary([r for r in rows if r["baseline"] == baseline], f"{tag} / {name}", wall)
        from plot_results import plot_yaw_sweep
        plot_yaw_sweep(rows, out / "yaw_sweep.png", tag)
        print(f"  saved results in {out}/")
        return
    print_summary(rows, tag, wall)
    if args.repeats > 1:
        poses = per_pose(rows)
        write_csv(poses, out / "poses.csv")
        print_pose_summary(poses)
    from plot_results import plot_outcomes  # matplotlib is only needed here
    # With repeats the plot shows the first noise sample of each pose, so points don't stack.
    plot_outcomes([r for r in rows if r["repeat"] == 0], out / "outcomes.png", tag, WORKSPACES[args.workspace])
    print(f"  saved results in {out}/")


if __name__ == "__main__":
    main()
