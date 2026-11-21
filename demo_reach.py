"""Pick the cube and place it in the box, using IK waypoints and checks between segments.

Run with the 3D viewer:   python demo_reach.py
Run without a window:     python demo_reach.py --headless   (prints a report)
Try a hard start pose:    python demo_reach.py --x 0.5 --y 0.1 --yaw 45
Original system:          python demo_reach.py --yaw 45 --no-align --retries 0 --ik-restarts 0 --no-arrive --controller pd

eval_place.py runs this many times with random cube poses and reports the success rate.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from controllers import CONTROLLERS
from ik import DOWN, solve_ik, solve_ik_multistart
from scene import (CUBE_START, GRIPPER_ACTUATOR, TCP_SITE, arm_qpos_ids, build_model, cube_yaw,
                   place_cube, yaw_quat)

OPEN, CLOSED = 0.0, 255.0
IK_POS_TOL = 1e-3            # m: a waypoint farther off than this means the plan can't be trusted
IK_ROT_TOL = np.radians(1.0)
LIFT_MIN = 0.10              # m: after "lift" the cube must have risen at least this much
HOLD_MAX = 0.04              # m: cube centre farther than this from the TCP = no longer in the hand
SETTLE_STEPS = 500
# Before closing or opening the gripper, wait until the arm has actually arrived (see wait_until_arrived).
ARRIVE_BEFORE = ("down to cube", "lower into box")
ARRIVE_TOL = 0.002           # m: TCP this close to its target counts as arrived
STILL_SPEED = 0.005          # m/s: TCP slower than this has stopped; waiting longer won't help
ARRIVE_TIMEOUT = 1.5         # s

OUTCOMES = ["OK", "NOT_SEEN", "IK_FAIL", "GRASP_MISS", "DROPPED", "MISSED_BOX"]


def grasp_quat(psi):
    """Gripper pointing straight down, turned by `psi` rad about the world z-axis."""
    q = np.empty(4)
    mujoco.mju_mulQuat(q, yaw_quat(psi), DOWN)
    return q


def grasp_angles(cube_yaw, align=True):
    """Candidate gripper turns (rad) that put the fingers square to the cube's faces, best first.

    A cube looks the same every 90 deg, so there are two distinct grasps: on one pair of
    faces (turn within +-45 deg) or on the other pair (that turn +-90 deg). The smaller
    wrist turn is tried first; the other is the fallback when the fingers would hit
    something, e.g. the box wall right next to the cube.
    Without `align` the gripper always comes straight down and closes on whatever is there.
    """
    if not align:
        return [0.0]
    psi = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return [psi, psi - np.sign(psi or 1) * np.pi / 2]


def make_plan(model, data, cube_pos, cube_yaw, align=True, ik_restarts=8, log=None):
    """Plan the pick and place. Returns (segments, ok, gripper quat).

    Each segment is (name, arm joint targets, gripper command, seconds). Grasp angles are
    tried in turn until one gives a full plan where every waypoint converges collision-free.
    """
    drop = data.site("box_target").xpos.copy()  # 5 cm above the box walls, centred over the box
    carry_z = cube_pos[2] + 0.20  # travel height: well above the box walls (6 cm)
    above_box = np.array([drop[0], drop[1], carry_z])
    # Only the two big moves at safe height may jump to a different arm configuration (IK restarts).
    # The short moves near the table must continue smoothly from the previous pose; jumping
    # there would sweep the arm through the table on the way.
    waypoints = [
        # --- pick ---         target                   gripper  s    big move
        ("above cube",     cube_pos + [0, 0, 0.15], OPEN,   2.0, True),
        ("down to cube",   cube_pos,                OPEN,   1.5, False),
        ("close gripper",  cube_pos,                CLOSED, 1.0, False),
        ("lift",           cube_pos + [0, 0, 0.20], CLOSED, 2.0, False),
        # --- place ---
        ("carry to box",   above_box,               CLOSED, 2.5, True),   # at a safe height, never sideways near walls
        ("lower into box", drop,                    CLOSED, 1.5, False),  # straight down, so the cube clears the walls
        ("release",        drop,                    OPEN,   1.0, False),  # open while still; the cube drops the last few cm
        ("retreat",        above_box,               OPEN,   1.5, False),  # straight up, so the fingers don't drag the cube
    ]
    check = ik_restarts > 0  # the baseline (no restarts) also skips collision checks, like the original
    for psi in grasp_angles(cube_yaw, align):
        quat = grasp_quat(psi)
        plan, ok, q_prev = [], True, data.qpos[arm_qpos_ids(model)].copy()
        if log:
            log(f"-- planning with gripper turned {np.degrees(psi):+.0f} deg")
        for name, target, grip, dur, big_move in waypoints:
            q, pos_err, rot_err, iters, wp_ok = solve_ik_multistart(
                model, data, TCP_SITE, target, quat, q_init=q_prev, restarts=ik_restarts if big_move else 0,
                pos_tol=IK_POS_TOL, rot_tol=IK_ROT_TOL, check_collisions=check)
            if log:
                log(f"IK '{name}': pos err {pos_err * 1000:.2f} mm, rot err {np.degrees(rot_err):.2f} deg, "
                    f"{iters} iters{'' if wp_ok else '  <- FAILED'}")
            ok &= wp_ok
            plan.append((name, q, grip, dur))
            q_prev = q
        if ok:
            break
    return plan, ok, quat


def run_plan(model, data, plan, step, arrive=None):
    """Send a smoothly blended control target every physics step; yield each segment's name when it ends.

    If given, `arrive(q_target, grip)` runs after the segments in ARRIVE_BEFORE, before the
    name is yielded: the trajectory timing says the arm should be there, `arrive` checks it is.
    """
    q_from = step.q_cmd.copy()  # the last joint target (data.ctrl may include a feedforward offset)
    for name, q_to, grip, dur in plan:
        steps = int(dur / model.opt.timestep)
        for k in range(steps):
            s = (k + 1) / steps
            s = 3 * s**2 - 2 * s**3  # smoothstep: zero velocity at start and end
            step(q_from + s * (q_to - q_from), grip)
        if arrive is not None and name in ARRIVE_BEFORE:
            arrive(q_to, grip)
        q_from = q_to
        yield name


def cube_in_hand(data):
    return np.linalg.norm(data.body("cube").xpos - data.site(TCP_SITE).xpos) < HOLD_MAX


def cube_in_box(data, inner_half=0.07):
    """Success = cube centre inside the box walls and resting low (not stuck on a wall or in the gripper)."""
    rel = data.body("cube").xpos - data.body("box").xpos
    return abs(rel[0]) < inner_half and abs(rel[1]) < inner_half and rel[2] < 0.05


def ground_truth(data):
    """Perfect perception: the cube pose straight from the simulator."""
    return data.body("cube").xpos.copy(), cube_yaw(data)


METRICS = ["grasp_offset_mm", "grasp_twist_deg", "place_error_mm", "target_shift_mm", "look_age_s",
           "reach_error_mm", "track_error_mm",
           "hold_error_mm", "wait_s"]


def tcp_error(model, data, q_target, scratch):
    """Distance (m) between where the TCP is and where it would be at the joint target q_target."""
    scratch.qpos[:] = data.qpos
    scratch.qpos[arm_qpos_ids(model)] = q_target
    mujoco.mj_kinematics(model, scratch)
    return np.linalg.norm(scratch.site(TCP_SITE).xpos - data.site(TCP_SITE).xpos)


def wrap_cube_angle(a):
    """Wrap an angle (rad) to [-45, 45) deg: for a cube, rotations 90 deg apart look the same."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


def wait_until_arrived(model, data, step, q_target, grip, scratch):
    """Keep commanding q_target until the TCP is within ARRIVE_TOL of it. Returns seconds waited.

    The planner's timing assumes the arm keeps up with the trajectory. It doesn't: a PD
    controller lags behind during motion. Closing the gripper on schedule can then pinch
    the cube too high. Stops early if the arm has stopped moving (a controller with a
    static error, like plain PD sagging under gravity, never gets closer) or at the timeout.
    """
    dt, chunk = model.opt.timestep, 10
    prev = data.site(TCP_SITE).xpos.copy()
    n = 0
    while n * dt < ARRIVE_TIMEOUT and tcp_error(model, data, q_target, scratch) > ARRIVE_TOL:
        for _ in range(chunk):
            step(q_target, grip)
        n += chunk
        pos = data.site(TCP_SITE).xpos.copy()
        if np.linalg.norm(pos - prev) / (chunk * dt) < STILL_SPEED:
            break
        prev = pos
    return n * dt


def pick_and_place(model, data, step, perceive=ground_truth, align=True, retries=0, ik_restarts=8, log=None,
                   metrics=None, arrive=True):
    """Run one attempt (plus `retries` re-grasps). Returns (outcome, grasp attempts used).

    `step(q_cmd, grip)` advances the simulation by one physics step.
    `perceive(data)` returns the cube (position, yaw) that the planner sees, or None if it can't see it.

    Pass a dict as `metrics` to also get continuous measurements (NaN when not reached).
    Pass/fail hides how close a run came to failing; these show the margin:
      grasp_offset_mm   horizontal distance cube centre <-> fingertip centre right after closing
      grasp_twist_deg   how far closing the fingers rotated the cube (0 = fingers met it square)
      place_error_mm    final horizontal distance of the cube from the box centre (OK runs)
    And how well the controller follows the planned joint targets (TCP distance, actual vs target):
      reach_error_mm    at the end of "down to cube", i.e. where the grasp really happens
      track_error_mm    mean over all segment ends (the arm is still catching up after big moves)
      hold_error_mm     after the final settle, holding still (static sag)
      wait_s            total time spent waiting for the arm to arrive (with `arrive`)
      target_shift_mm   when the gripper starts closing: how far the cube is from where the planner
                        aimed (perception error plus how far a moving cube travelled since the look)
      look_age_s        how old that look is when the gripper starts closing (s)
    """
    say = log or (lambda msg: None)
    m = metrics if metrics is not None else {}
    m.update({k: np.nan for k in METRICS})
    z0 = data.body("cube").xpos[2]
    scratch = mujoco.MjData(model)
    seg_errors = []
    m["wait_s"] = 0.0

    def wait(q_target, grip):
        m["wait_s"] += wait_until_arrived(model, data, step, q_target, grip, scratch)
    for attempt in range(1, retries + 2):
        seen = perceive(data)
        looked_at = data.time
        if seen is None:  # camera perception: cube occluded or out of view
            say(f"attempt {attempt}: cube not seen")
            return "NOT_SEEN", attempt
        cube_pos, yaw = seen
        plan, ik_ok, quat = make_plan(model, data, cube_pos, yaw, align, ik_restarts, log)
        if not ik_ok:
            return "IK_FAIL", attempt
        targets = {seg[0]: seg[1] for seg in plan}
        for name in run_plan(model, data, plan, step, wait if arrive else None):
            seg_errors.append(1000 * tcp_error(model, data, targets[name], scratch))
            m["track_error_mm"] = np.mean(seg_errors)
            if name == "down to cube":
                yaw_before_close = cube_yaw(data)
                m["reach_error_mm"] = seg_errors[-1]
                m["target_shift_mm"] = 1000 * np.linalg.norm(data.body("cube").xpos[:2] - cube_pos[:2])
                m["look_age_s"] = data.time - looked_at
            if name == "close gripper":
                m["grasp_offset_mm"] = 1000 * np.linalg.norm((data.body("cube").xpos - data.site(TCP_SITE).xpos)[:2])
                m["grasp_twist_deg"] = np.degrees(wrap_cube_angle(cube_yaw(data) - yaw_before_close))
            if name == "lift" and data.body("cube").xpos[2] < z0 + LIFT_MIN:
                say(f"attempt {attempt}: cube did not come up -> grasp missed")
                break
            if name in ("carry to box", "lower into box") and not cube_in_hand(data):
                return "DROPPED", attempt
        else:
            for _ in range(SETTLE_STEPS):
                step(plan[-1][1], OPEN)
            m["hold_error_mm"] = 1000 * tcp_error(model, data, plan[-1][1], scratch)
            if not cube_in_box(data):
                return "MISSED_BOX", attempt
            m["place_error_mm"] = 1000 * np.linalg.norm((data.body("cube").xpos - data.body("box").xpos)[:2])
            return "OK", attempt

        # Grasp missed: open, back straight up, and look again before the next attempt.
        up = data.site(TCP_SITE).xpos + [0, 0, 0.10]
        q_up, *_ = solve_ik(model, data, TCP_SITE, up, quat, q_init=data.qpos[arm_qpos_ids(model)])
        for _ in run_plan(model, data, [("back off", q_up, OPEN, 1.0)], step):
            pass
    return "GRASP_MISS", attempt


class Stepper:
    """step(q_cmd, grip): one physics step towards arm joint target q_cmd.

    `controller` (see controllers.py) turns q_cmd into the arm actuator commands.
    `q_cmd` holds the last joint target. With a viewer, it also draws and runs in real time.
    `hooks` are called as hook(model, data) before every physics step, e.g. a hand moving the cube.

    A class rather than a closure that stores attributes on itself: that closure refers to
    itself, a reference cycle that Python frees only occasionally. Every trial's MuJoCo
    model and data then stayed in memory (about 2 GB per worker after a few hundred trials).
    """

    def __init__(self, model, data, viewer=None, controller="pd", hooks=()):
        self.model, self.data, self.viewer, self.hooks = model, data, viewer, list(hooks)
        self.gripper = model.actuator(GRIPPER_ACTUATOR).id
        self.ctl = CONTROLLERS[controller](model)
        self.q_cmd = data.ctrl[:6].copy()  # reset_home sets ctrl to the home joint angles

    def __call__(self, q_cmd, grip):
        model, data = self.model, self.data
        t0 = time.time()
        self.q_cmd = np.asarray(q_cmd, dtype=float).copy()
        data.ctrl[:6] = self.ctl(model, data, self.q_cmd)
        data.ctrl[self.gripper] = grip
        for hook in self.hooks:
            hook(model, data)
        mujoco.mj_step(model, data)
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit
            self.viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - t0)))  # real-time pace


make_stepper = Stepper


def run_in_viewer(model, data, controller="pd", hooks=(), **kwargs):
    """Run pick_and_place in the 3D viewer, then hold the final pose until the window closes."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = make_stepper(model, data, viewer, controller, hooks)
        result = pick_and_place(model, data, step, **kwargs)
        print(f"Outcome: {result[0]} after {result[1]} attempt(s). Close the window to exit.")
        while viewer.is_running():
            step(step.q_cmd, data.ctrl[-1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--x", type=float, default=CUBE_START[0])
    parser.add_argument("--y", type=float, default=CUBE_START[1])
    parser.add_argument("--yaw", type=float, default=0.0, help="cube yaw in degrees")
    parser.add_argument("--no-align", action="store_true", help="baseline: ignore cube yaw when grasping")
    parser.add_argument("--retries", type=int, default=1, help="re-grasp attempts after a missed grasp")
    parser.add_argument("--ik-restarts", type=int, default=8, help="extra IK starting poses if IK gets stuck")
    parser.add_argument("--controller", choices=CONTROLLERS, default="pd_gravity", help="arm controller, see controllers.py")
    parser.add_argument("--no-arrive", action="store_true", help="close/open the gripper on schedule, don't wait")
    args = parser.parse_args()

    model, data = build_model()
    place_cube(model, data, (args.x, args.y), np.radians(args.yaw))
    kwargs = dict(align=not args.no_align, retries=args.retries, ik_restarts=args.ik_restarts, log=print,
                  arrive=not args.no_arrive)
    if args.headless:
        metrics = {}
        outcome, attempts = pick_and_place(model, data, make_stepper(model, data, controller=args.controller),
                                           metrics=metrics, **kwargs)
        print(f"TCP final: {data.site(TCP_SITE).xpos.round(3)}  cube final: {data.body('cube').xpos.round(3)}")
        print(f"Outcome: {outcome} after {attempts} attempt(s)")
        print("  ".join(f"{k} {v:.1f}" for k, v in metrics.items()))
    else:
        run_in_viewer(model, data, args.controller, **kwargs)


if __name__ == "__main__":
    main()
