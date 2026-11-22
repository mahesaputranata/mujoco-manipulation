"""Where does the fixed camera lose sight of the cube while the gripper comes down, and what blocks it?

In the dynamic-target sweeps, trials where the camera kept seeing the cube never failed (0 of 3024
with under 5 frames lost), while trials that lost 40+ frames failed 59% of the time. This measures
the occlusion directly: a still cube on a grid of positions and rotations, the tracking descent as
usual, and on every frame a segmentation render (which object each pixel belongs to). Per frame:

    visible   share of the cube's silhouette (projected from its true pose) that the camera sees
    blockers  which part of the robot covers the rest: fingers, gripper body, wrist, arm

    python visibility_map.py   ->  results/visibility/frames.csv, visibility.png

The cube's projected silhouette uses the true camera and pose, so this measures what is hidden,
independent of how well perception copes with it.
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import mujoco
import numpy as np

from eval_place import ROOT, build_parser, setup_trial

OUT = ROOT / "results" / "visibility"
XS = [0.30, 0.38, 0.46, 0.54, 0.62]
YS = [-0.05, 0.05, 0.15, 0.25, 0.35]
YAWS = [0, 15, 30, 45]  # degrees; a cube repeats every 90, and the grasp picks the closest faces
GROUPS = [("fingers", ("pad", "follower", "coupler", "driver", "spring")), ("gripper body", ("gripper/",)),
          ("wrist", ("wrist",)), ("arm", ("forearm", "upper_arm", "shoulder", "base"))]


def group_of(name):
    for group, keys in GROUPS:
        if any(k in name for k in keys):
            return group
    return "other"


def measure(case):
    """Run one tracking descent onto a still cube; return per-frame visibility rows."""
    from demo_reach import TCP_SITE, make_stepper, pick_and_place
    from perception import PinholeCamera, cube_corners
    from scene import SCENE_CAMERA

    x, y, yaw_deg = case
    args = build_parser().parse_args(["--perception", "camera", "--track"])
    trial = dict(trial=0, x=x, y=y, yaw_deg=yaw_deg, half=0.025, mass=0.1, friction=1.0, noise_seed=0, repeat=0)
    model, data, kwargs = setup_trial(trial, args)
    hooks = kwargs.pop("hooks")
    seg_renderer = mujoco.Renderer(model, 480, 640)
    seg_renderer.enable_segmentation_rendering()
    cube_geom = model.geom("cube_geom").id
    body_group = {g: group_of(model.body(model.geom_bodyid[g]).name) for g in range(model.ngeom)}
    world_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    rows = []
    perceive = kwargs["perceive"]
    real_quick = perceive.quick

    def quick(d, guess=None):
        result = real_quick(d, guess)
        camera = PinholeCamera(model, d)
        cube = d.body("cube")
        yaw = np.arctan2(cube.xmat[3], cube.xmat[0])
        hull = cv2.convexHull(np.round(camera.project(cube_corners(cube.xpos[0], cube.xpos[1], yaw, 0.025)) - 0.5)
                              .astype(np.int32))
        expected = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(expected, hull, 1)
        expected = expected.astype(bool)
        seg_renderer.update_scene(d, camera=SCENE_CAMERA)
        seg = seg_renderer.render()
        ids, types = seg[..., 0], seg[..., 1]
        is_geom = types == mujoco.mjtObj.mjOBJ_GEOM
        # Pixels of world-fixed geoms (floor, tags) inside the outline are rounding at its edges, not
        # occlusion: nothing fixed can be in front of the cube. They count neither as seen nor hidden.
        movable = is_geom & ~np.isin(ids, world_geoms)
        seen = expected & is_geom & (ids == cube_geom)
        hidden = expected & movable & (ids != cube_geom)
        total = max(seen.sum() + hidden.sum(), 1)
        row = dict(x=x, y=y, yaw_deg=yaw_deg, height_mm=1000 * (d.site(TCP_SITE).xpos[2] - cube.xpos[2]),
                   visible=seen.sum() / total, accepted=int(result is not None))
        for group, _ in GROUPS + [("other", ())]:
            row[group] = 0
        for g in np.unique(ids[hidden]):
            row[body_group[int(g)]] += int(np.count_nonzero(hidden & (ids == g))) / total
        rows.append(row)
        return result

    perceive.quick = quick
    try:
        pick_and_place(model, data, make_stepper(model, data, controller="pd_gravity", hooks=hooks), **kwargs)
    finally:
        perceive.close()
        seg_renderer.close()
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [(x, y, yaw) for x in XS for y in YS for yaw in YAWS]
    rows = []
    with ProcessPoolExecutor(os.cpu_count()) as pool:
        for case_rows in pool.map(measure, cases):
            rows += case_rows
    with open(OUT / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(f"{len(cases)} cases, {len(rows)} frames -> {OUT / 'frames.csv'}")
    summarize(rows)


def summarize(rows):
    a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    print("\nby gripper height above the cube's centre:")
    print(f"{'height (mm)':>12s} {'frames':>7s} {'visible (median)':>17s} {'accepted':>9s}   hidden by: "
          + "  ".join(g for g, _ in GROUPS))
    for lo, hi in ((120, 999), (80, 120), (60, 80), (40, 60), (20, 40), (-99, 20)):
        sel = (a["height_mm"] >= lo) & (a["height_mm"] < hi)
        if sel.sum():
            hidden = "  ".join(f"{100 * a[g][sel].mean():{len(g)}.0f}%" for g, _ in GROUPS)
            print(f"{f'{lo}-{hi}':>12s} {sel.sum():7d} {100 * np.median(a['visible'][sel]):16.0f}% "
                  f"{100 * a['accepted'][sel].mean():8.0f}%   {hidden}")
    # Blind height per case: the highest gripper height at which the cube is less than 40% visible.
    cases = {}
    for r in rows:
        cases.setdefault((r["x"], r["y"], r["yaw_deg"]), []).append(r)
    blind = {k: max([r["height_mm"] for r in v if r["visible"] < 0.4], default=0.0) for k, v in cases.items()}
    print("\nblind height (mm above the cube where < 40% stays visible), mean over yaw; rows y, cols x")
    print("    y\\x " + " ".join(f"{x:6.2f}" for x in XS))
    for y in YS[::-1]:
        print(f"{y:6.2f}  " + " ".join(f"{np.mean([blind[(x, y, w)] for w in YAWS]):6.0f}" for x in XS))
    print("\nby cube yaw (mean blind height):", {w: round(np.mean([blind[(x, y, w)] for x in XS for y in YS]))
                                               for w in YAWS})


if __name__ == "__main__":
    if "--summary" in sys.argv:
        with open(OUT / "frames.csv") as f:
            summarize([{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)])
    else:
        main()
