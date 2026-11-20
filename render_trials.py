"""Render eval_place.py trials to GIFs, e.g. for the README.

Trials are rebuilt from the same flags and seed as the eval, so the GIF shows exactly
what happened in that trial. Pass the same scenario flags you used for eval_place.py.

    # trial 7 of the wide-workspace eval, original vs improved system, side by side
    python render_trials.py --workspace wide --trial 7 --compare

    # the 3 most fragile poses from a --repeats run (renders a failing noise sample of each)
    python render_trials.py --pos-noise 15 --yaw-noise 15 --fragile results/fixed_default_noise15mm15deg_x10/poses.csv

    # every GRASP_MISS from an eval (up to --limit)
    python render_trials.py --baseline --pos-noise 5 --yaw-noise 5 \\
        --from-csv results/baseline_default_noise5mm5deg/trials.csv --outcome GRASP_MISS

GIFs go to results/gifs/ (or --out).
"""
import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from demo_reach import Stepper, pick_and_place
from eval_place import ROOT, add_scenario_args, controller_for, sample_trial, setup_trial

FPS = 15
COLORS = 256  # GIF palette size: fewer colours = much smaller files
OK_COLOR, FAIL_COLOR, TEXT_COLOR, BAR_COLOR = (12, 163, 12), (208, 59, 59), (230, 230, 230), (20, 20, 20)


def make_camera():
    """A fixed view from the front-left that shows the arm, the table and the box."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.25, 0.05, 0.22]
    cam.distance, cam.azimuth, cam.elevation = 1.9, 150, -28
    return cam


class FilmingStepper(Stepper):
    """A Stepper that also renders a frame every `every` physics steps."""

    def __init__(self, model, data, controller, renderer, every):
        super().__init__(model, data, controller=controller)
        self.renderer, self.every, self.cam = renderer, every, make_camera()
        self.frames, self.count = [], 0

    def snapshot(self):
        self.renderer.update_scene(self.data, self.cam)
        return self.renderer.render().copy()

    def __call__(self, q_cmd, grip):
        super().__call__(q_cmd, grip)
        self.count += 1
        if self.count % self.every == 0:
            self.frames.append(self.snapshot())


def render_trial(trial, args, width, height, speed):
    """Run one trial and return (frames, outcome, attempts). One frame per 1/FPS s of GIF time."""
    model, data, kwargs = setup_trial(trial, args)
    renderer = mujoco.Renderer(model, height, width)
    every = max(1, round(speed / (FPS * model.opt.timestep)))  # physics steps per GIF frame
    step = FilmingStepper(model, data, controller_for(trial, args), renderer, every)
    outcome, attempts = pick_and_place(model, data, step, **kwargs)
    frames = step.frames or [step.snapshot()] * FPS  # e.g. IK_FAIL before any motion: a still second
    renderer.close()
    return frames, outcome, attempts


def caption(frame, lines, outcome):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    color = OK_COLOR if outcome == "OK" else FAIL_COLOR
    draw.rectangle([0, 0, img.width, 16 * len(lines) + 8], fill=BAR_COLOR)
    for k, text in enumerate(lines):
        draw.text((8, 4 + 16 * k), text, fill=color if k == 0 else TEXT_COLOR)
    return img


def label(trial, args, outcome, attempts):
    mode = "original system" if args.baseline else "improved system"
    rep = f".{trial['repeat']}" if trial["repeat"] else ""
    return [f"{outcome}  ({mode}, {attempts} attempt{'s' * (attempts > 1)})",
            f"trial {trial['trial']}{rep}: x={trial['x']:.2f} y={trial['y']:.2f} yaw={trial['yaw_deg']:+.0f} deg"]


def save_gif(frames, path):
    frames = frames + [frames[-1]] * FPS  # hold the last frame for a second
    # One shared palette for all frames, no dithering: the checker floor dithers into noise otherwise.
    # Build the palette from several frames (one frame can miss colours, e.g. the box's brown),
    # plus solid patches of the caption colours, which are only a few pixels each.
    picks = [frames[int(k * (len(frames) - 1) / 3)] for k in range(4)]
    w, h = picks[0].width, picks[0].height
    sample = Image.new("RGB", (w, 4 * h + 60))
    for k, f in enumerate(picks):
        sample.paste(f, (0, k * h))
    for k, color in enumerate([OK_COLOR, FAIL_COLOR, TEXT_COLOR, BAR_COLOR]):
        sample.paste(color, (k * w // 4, 4 * h, (k + 1) * w // 4, 4 * h + 60))
    palette = sample.quantize(COLORS, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def render(trial, args, opts):
    tag = f"trial{trial['trial']}" + (f".{trial['repeat']}" if trial["repeat"] else "")
    if opts.compare:
        sides = []
        for baseline in (True, False):
            side_args = argparse.Namespace(**{**vars(args), "baseline": baseline})
            frames, outcome, attempts = render_trial(trial, side_args, opts.width, opts.height, opts.speed)
            sides.append([caption(f, label(trial, side_args, outcome, attempts), outcome) for f in frames])
        n = max(map(len, sides))
        sides = [s + [s[-1]] * (n - len(s)) for s in sides]  # the shorter run holds its last frame
        frames = []
        for left, right in zip(*sides):
            both = Image.new("RGB", (2 * opts.width, opts.height))
            both.paste(left, (0, 0))
            both.paste(right, (opts.width, 0))
            frames.append(both)
        name = f"{tag}_compare.gif"
    else:
        frames, outcome, attempts = render_trial(trial, args, opts.width, opts.height, opts.speed)
        frames = [caption(f, label(trial, args, outcome, attempts), outcome) for f in frames]
        name = f"{tag}_{'baseline' if args.baseline else 'fixed'}_{outcome}.gif"
    save_gif(frames, opts.out / name)


def pick_trials(args, opts):
    """Which (trial, repeat) pairs to render, from --trial, --from-csv or --fragile."""
    if opts.trial:
        return [(i, opts.repeat) for i in opts.trial]
    if opts.from_csv:
        with open(opts.from_csv) as f:
            rows = [r for r in csv.DictReader(f) if not opts.outcome or r["outcome"] == opts.outcome]
        return [(int(r["trial"]), int(r.get("repeat") or 0)) for r in rows][:opts.limit]
    if opts.fragile:
        with open(opts.fragile) as f:
            poses = [p for p in csv.DictReader(f) if 0 < float(p["ok_rate"]) < 1]
        poses.sort(key=lambda p: abs(float(p["ok_rate"]) - 0.5))
        picks = []
        trials_csv = Path(opts.fragile).with_name("trials.csv")
        with open(trials_csv) as f:
            runs = list(csv.DictReader(f))
        for p in poses[:opts.limit]:  # show a failing noise sample: that is the interesting one
            fail = next(r for r in runs if r["trial"] == p["trial"] and r["outcome"] != "OK")
            picks.append((int(fail["trial"]), int(fail["repeat"])))
        return picks
    raise SystemExit("choose trials with --trial, --from-csv or --fragile")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(parser)
    parser.add_argument("--trial", type=int, nargs="+", help="trial number(s) to render")
    parser.add_argument("--repeat", type=int, default=0, help="noise sample, as in eval_place.py --repeats")
    parser.add_argument("--from-csv", help="render trials listed in a trials.csv ...")
    parser.add_argument("--outcome", help="... only those with this outcome, e.g. GRASP_MISS")
    parser.add_argument("--fragile", help="render the poses.csv poses closest to 50%% success")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--compare", action="store_true", help="original vs improved system side by side")
    parser.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "gifs")
    opts = parser.parse_args()
    opts.out.mkdir(parents=True, exist_ok=True)

    for i, r in pick_trials(opts, opts):
        trial = sample_trial(i, opts, r)
        print(f"rendering trial {i}" + (f" (noise sample {r})" if r else ""))
        render(trial, opts, opts)


if __name__ == "__main__":
    main()
