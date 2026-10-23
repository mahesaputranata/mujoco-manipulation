"""Step 1 demo: use IK to reach the cube, grab it, and lift it.

Run with the 3D viewer:   python demo_reach.py
Run without a window:     python demo_reach.py --headless   (prints a report)

Placing the cube in the box is left as your exercise. See README.md.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from ik import solve_ik
from scene import GRIPPER_ACTUATOR, TCP_SITE, arm_qpos_ids, build_model

OPEN, CLOSED = 0.0, 255.0


def make_plan(model, data):
    """List of (arm joint targets, gripper command, seconds) segments."""
    cube = data.body("cube").xpos.copy()  # ground-truth pose from sim (we'll replace this with a camera later)
    drop = data.site("box_target").xpos.copy()  # 5 cm above the box walls, centred over the box
    carry_z = cube[2] + 0.20  # travel height: well above the box walls (6 cm)
    above_box = np.array([drop[0], drop[1], carry_z])
    waypoints = [
        # --- pick ---
        ("above cube", cube + [0, 0, 0.15], OPEN, 2.0),
        ("down to cube", cube + [0, 0, 0.0], OPEN, 1.5),
        ("close gripper", cube + [0, 0, 0.0], CLOSED, 1.0),
        ("lift", cube + [0, 0, 0.20], CLOSED, 2.0),
        # --- place ---
        ("carry to box", above_box, CLOSED, 2.5),   # move at a safe height, never sideways near walls
        ("lower into box", drop, CLOSED, 1.5),      # straight down, so the cube clears the walls
        ("release", drop, OPEN, 1.0),               # open while still; the cube drops the last few cm
        ("retreat", above_box, OPEN, 1.5),          # straight up, so the fingers don't drag the cube
    ]
    plan, q_prev = [], data.qpos[arm_qpos_ids(model)].copy()
    for name, target, grip, dur in waypoints:
        q, pos_err, rot_err, iters = solve_ik(model, data, TCP_SITE, target, q_init=q_prev)
        print(f"IK '{name}': pos err {pos_err * 1000:.2f} mm, rot err {np.degrees(rot_err):.2f} deg, {iters} iters")
        plan.append((name, q, grip, dur))
        q_prev = q
    return plan


def run(headless):
    model, data = build_model()
    gripper = model.actuator(GRIPPER_ACTUATOR).id
    plan = make_plan(model, data)

    def segments():
        """Yield a control target every physics step, smoothly blending between waypoints."""
        q_from = data.ctrl[:6].copy()
        for name, q_to, grip, dur in plan:
            steps = int(dur / model.opt.timestep)
            for k in range(steps):
                s = (k + 1) / steps
                s = 3 * s**2 - 2 * s**3  # smoothstep: zero velocity at start and end
                yield name, q_from + s * (q_to - q_from), grip
            q_from = q_to

    def step(q_cmd, grip):
        data.ctrl[:6] = q_cmd
        data.ctrl[gripper] = grip
        mujoco.mj_step(model, data)

    if headless:
        for name, q_cmd, grip in segments():
            step(q_cmd, grip)
        for _ in range(500):  # let it settle
            step(q_cmd, grip)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for name, q_cmd, grip in segments():
                if not viewer.is_running():
                    return
                t0 = time.time()
                step(q_cmd, grip)
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.time() - t0)))  # real-time pace
            while viewer.is_running():  # hold the final pose until you close the window
                step(q_cmd, grip)
                viewer.sync()
                time.sleep(model.opt.timestep)

    print(f"TCP final: {data.site(TCP_SITE).xpos.round(3)}  cube final: {data.body('cube').xpos.round(3)}")
    print("Cube in box -> PLACE OK" if cube_in_box(data) else "Cube not in box -> PLACE FAILED")


def cube_in_box(data, inner_half=0.07):
    """Success = cube centre inside the box walls and resting low (not stuck on a wall or in the gripper)."""
    rel = data.body("cube").xpos - data.body("box").xpos
    return abs(rel[0]) < inner_half and abs(rel[1]) < inner_half and rel[2] < 0.05


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    run(parser.parse_args().headless)
