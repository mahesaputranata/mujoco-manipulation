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
from scene import (CAMERA_EYE, CUBE_START, GRIPPER_ACTUATOR, TCP_SITE, arm_qpos_ids, build_model, cube_yaw,
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
# Tracking (Option B): during the descent the camera keeps watching and the aim follows the cube.
TRACK_HZ = 20                # camera frames per second while tracking
DESCENT_SPEED = 0.10         # m/s, the same pace as the planned 15 cm descent in 1.5 s
CENTRE_TOL = 0.004           # m: close only once the fingertips are this close above the cube's centre
TRACK_TIMEOUT = 1.0          # s at grasp height without centring: close anyway
CLOSE_TIME = 1.0             # s of closing (as in the plan), still following the cube
MAX_SPEED = 0.5              # m/s: a frame whose estimate jumped faster than this is rejected

OUTCOMES = ["OK", "NOT_SEEN", "IK_FAIL", "GRASP_MISS", "DROPPED", "MISSED_BOX"]


def grasp_quat(psi):
    """Gripper pointing straight down, turned by `psi` rad about the world z-axis."""
    q = np.empty(4)
    mujoco.mju_mulQuat(q, yaw_quat(psi), DOWN)
    return q


def finger_axis(psi):
    """Horizontal direction the fingers close along, for gripper turn psi (measured: 90 deg + psi)."""
    return np.array([-np.sin(psi), np.cos(psi)])


def grasp_angles(cube_yaw, align=True, view=None):
    """Candidate gripper turns (rad) that put the fingers square to the cube's faces, best first.

    A cube looks the same every 90 deg, so there are two distinct grasps: on one pair of
    faces (turn within +-45 deg) or on the other pair (that turn +-90 deg). The smaller
    wrist turn is tried first; the other is the fallback when the fingers would hit
    something, e.g. the box wall right next to the cube.
    Without `align` the gripper always comes straight down and closes on whatever is there.

    With `view` (the camera's horizontal viewing direction towards the cube) the grasp whose fingers
    close *across* the line of sight comes first: the gap between the fingers then stays open towards
    the camera, so it can watch the cube's top face for longer while tracking.
    """
    if not align:
        return [0.0]
    psi = (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    angles = [psi, psi - np.sign(psi or 1) * np.pi / 2]
    if view is not None:
        angles.sort(key=lambda a: abs(finger_axis(a) @ view))
    return angles


def quat_turn(quat):
    """The gripper turn psi of an orientation made by grasp_quat(psi)."""
    q_down_inv, rel = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(q_down_inv, DOWN)
    mujoco.mju_mulQuat(rel, quat, q_down_inv)  # yaw_quat(psi) = quat * DOWN^-1
    return 2 * np.arctan2(rel[3], rel[0])


def make_plan(model, data, cube_pos, cube_yaw, align=True, ik_restarts=8, log=None, view=None, angles=None):
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
    for psi in angles if angles is not None else grasp_angles(cube_yaw, align, view):
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
           "relook_seen", "relook_shift_mm", "track_lag_mm", "frames_lost", "reach_error_mm", "track_error_mm",
           "hold_error_mm", "wait_s", "descent_s"]


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


def wrap_grasp_angle(a):
    """Wrap to [-90, 90) deg: a parallel gripper turned 180 deg grips the same way."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def track_descent(model, data, step, perceive, cube_pos, yaw, align, m, hz=TRACK_HZ, say=None, psi=None):
    """Descend onto the cube while watching it (visual servoing), close the gripper, keep following.

    Every camera frame: estimate the cube's pose, aim the fingertips at its centre at a height that
    shrinks by DESCENT_SPEED, solve IK from the current target, and spread the move over the frame.
    A frame that shows the cube jumping faster than MAX_SPEED, or no cube at all (the fingers can
    block the view close to the grasp), is rejected: the aim stays on the last good estimate.
    Returns the gripper turn used, so the rest of the plan can continue at the same wrist angle.
    """
    say = say or (lambda msg: None)
    quick = getattr(perceive, "quick", None)  # one image per frame, tracking from the last estimate
    steps_per_frame = max(1, round(1 / (hz * model.opt.timestep)))
    est_pos, est_yaw = np.array(cube_pos, float), yaw
    psi = grasp_angles(est_yaw, align)[0] if psi is None else psi  # the plan's wrist angle
    z_grasp, z = est_pos[2], est_pos[2] + 0.15
    q_next = step.q_cmd.copy()
    last_seen, lost, start, closing_since, at_height_since = data.time, 0, data.time, None, None
    lag = []
    while True:
        seen = quick(data, guess=(est_pos, est_yaw)) if quick else perceive(data)
        dt = data.time - last_seen
        if seen is not None and np.linalg.norm(seen[0][:2] - est_pos[:2]) <= MAX_SPEED * dt + 0.01:
            est_pos, est_yaw, last_seen = np.array(seen[0], float), seen[1], data.time
        else:
            lost += 1
        # Of the two grasps (the cube's two pairs of faces), keep the one closest to the current wrist.
        psi = min(grasp_angles(est_yaw, align), key=lambda a: abs(wrap_grasp_angle(a - psi)))
        if closing_since is None:
            z = max(z_grasp, z - DESCENT_SPEED / hz)
        target = np.array([est_pos[0], est_pos[1], z])
        q_prev = q_next
        q_next, *_ = solve_ik(model, data, TCP_SITE, target, grasp_quat(psi), q_init=q_prev, max_iters=30)

        tcp, cube = data.site(TCP_SITE).xpos, data.body("cube").xpos
        if z <= z_grasp:
            lag.append(np.linalg.norm(tcp[:2] - cube[:2]))
            at_height_since = at_height_since if at_height_since is not None else data.time
        centred = np.linalg.norm(tcp[:2] - est_pos[:2]) < CENTRE_TOL and abs(tcp[2] - z_grasp) < 0.005
        timed_out = at_height_since is not None and data.time - at_height_since > TRACK_TIMEOUT
        if closing_since is None and z <= z_grasp and (centred or timed_out):
            closing_since = data.time
            m["target_shift_mm"] = 1000 * np.linalg.norm(cube[:2] - est_pos[:2])
            m["look_age_s"] = data.time - last_seen
            m["reach_error_mm"] = 1000 * np.linalg.norm(tcp - target)
            yaw_before_close = cube_yaw(data)
            if timed_out:
                say("tracking: not centred in time, closing anyway")
        grip = CLOSED if closing_since is not None else OPEN
        for k in range(steps_per_frame):
            step(q_prev + (k + 1) / steps_per_frame * (q_next - q_prev), grip)
        if closing_since is not None and data.time - closing_since >= CLOSE_TIME:
            break
    m["grasp_offset_mm"] = 1000 * np.linalg.norm((data.body("cube").xpos - data.site(TCP_SITE).xpos)[:2])
    m["grasp_twist_deg"] = np.degrees(wrap_cube_angle(cube_yaw(data) - yaw_before_close))
    m["track_lag_mm"] = 1000 * np.mean(lag[-max(1, len(lag) // 2):])  # gripper vs cube, at grasp height
    m["frames_lost"] = lost
    m["descent_s"] = closing_since - start
    return psi


def pick_and_place(model, data, step, perceive=ground_truth, align=True, retries=0, ik_restarts=8, log=None,
                   metrics=None, arrive=True, look_again=False, track=False):
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
    With `look_again`, the planner takes a second look once the gripper is above the cube and
    re-plans the descent from it, so the look is younger when the gripper closes:
      relook_seen       1 if the second look saw the cube (the arm above it can block the view)
      relook_shift_mm   how far the cube moved between the first and the second look
    With `track`, the descent and the grasp follow the cube frame by frame (track_descent):
      track_lag_mm      gripper vs cube, horizontally, while tracking at grasp height (arm lag)
      frames_lost       camera frames rejected or without the cube during the descent
      descent_s         from above the cube until the gripper starts closing
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
        view = None
        if track:  # prefer the grasp that keeps the cube visible to the fixed camera while tracking
            view = cube_pos[:2] - np.asarray(CAMERA_EYE[:2])
            view /= np.linalg.norm(view)
        plan, ik_ok, quat = make_plan(model, data, cube_pos, yaw, align, ik_restarts, log, view=view)
        if not ik_ok:
            return "IK_FAIL", attempt
        # The plan can change mid-motion (look_again), so the loop reads it from `now`.
        now = dict(plan=plan, cube_pos=cube_pos, quat=quat, looked_at=looked_at,
                   targets={seg[0]: seg[1] for seg in plan})

        def segments():
            if track:
                yield from run_plan(model, data, now["plan"][:1], step)  # to above the cube
                psi0 = quat_turn(now["quat"])
                psi = track_descent(model, data, step, perceive, now["cube_pos"], yaw, align, m, say=say, psi=psi0)
                # The cube is in the fingers now (or missed): plan lift, carry and place from here.
                here = data.site(TCP_SITE).xpos.copy()
                new_plan, ok, new_quat = make_plan(model, data, here, psi, align, ik_restarts, log, angles=[psi])
                if ok:
                    now.update(plan=new_plan, cube_pos=here, quat=new_quat,
                               targets={seg[0]: seg[1] for seg in new_plan})
                yield from run_plan(model, data, now["plan"][3:], step, wait if arrive else None)  # lift on
                return
            if not look_again:
                yield from run_plan(model, data, now["plan"], step, wait if arrive else None)
                return
            yield from run_plan(model, data, now["plan"][:1], step)  # to above the cube
            second = perceive(data)
            m["relook_seen"] = int(second is not None)
            if second is not None:
                m["relook_shift_mm"] = 1000 * np.linalg.norm(second[0][:2] - now["cube_pos"][:2])
                new_plan, ok, new_quat = make_plan(model, data, second[0], second[1], align, ik_restarts, log)
                if ok:  # descend to where the cube is now; otherwise keep the first plan
                    now.update(plan=new_plan, cube_pos=second[0], quat=new_quat, looked_at=data.time,
                               targets={seg[0]: seg[1] for seg in new_plan})
            yield from run_plan(model, data, now["plan"][1:], step, wait if arrive else None)

        for name in segments():
            cube_pos, quat, plan, targets = now["cube_pos"], now["quat"], now["plan"], now["targets"]
            seg_errors.append(1000 * tcp_error(model, data, targets[name], scratch))
            m["track_error_mm"] = np.mean(seg_errors)
            if name == "down to cube":
                yaw_before_close = cube_yaw(data)
                m["reach_error_mm"] = seg_errors[-1]
                m["target_shift_mm"] = 1000 * np.linalg.norm(data.body("cube").xpos[:2] - cube_pos[:2])
                m["look_age_s"] = data.time - now["looked_at"]
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
