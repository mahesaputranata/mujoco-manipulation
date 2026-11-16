"""See a Denavit-Hartenberg (DH) table in 3D with MuJoCo.

Convention: STANDARD (distal) DH, as in Spong and most robotics courses.
Going from frame i-1 to frame i, apply in this order:

    T_{i-1,i} = Rz(theta_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)

    1. rotate about z_{i-1} by theta  (this is the JOINT: theta = theta_home + q)
    2. slide along z_{i-1} by d
    3. slide along the new x_i by a
    4. twist about x_i by alpha        (tilts the next joint axis)

The pose of frame i in the base frame is just the product
    T_{0,i} = T_{0,1} · T_{1,2} · ... · T_{i-1,i}

What the viewer shows:
    - every DH frame i as a body named "frame i", with RGB axes:
      red = x, green = y, blue = z   (z_i is the axis joint i+1 rotates about)
    - grey cylinders = joint axes, coloured capsules = the d and a offsets
    - right panel -> Control: one slider per joint (q in rad, 0 = home pose)

Run:
    python dh_viewer.py                    # 3D viewer at the home pose
    python dh_viewer.py --q 0 30 0 0 0 0   # start with joint 2 at +30 deg
    python dh_viewer.py --headless         # print the frame table only
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

# ---- Your DH table (edit this to try other robots) --------------------------
L1, L2, L3, L4, L5, L6 = 0.0411, 0.192, 0.3163, 0.0759, 0.306, 0.233

#       theta_home (deg), alpha (deg), a (m), d (m)
DH = [
    (0,   90,  L1, L2),
    (90,  0,   L3, 0),
    (0,   90,  L4, 0),
    (0,   270, 0,  L5),
    (-90, 90,  0,  0),
    (0,   0,   0,  L6),
]
# -----------------------------------------------------------------------------


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def trans(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def dh_transform(theta, d, a, alpha):
    """One row of the table -> 4x4 homogeneous transform from frame i-1 to frame i."""
    return rot_z(theta) @ trans(0, 0, d) @ trans(a, 0, 0) @ rot_x(alpha)


def forward_kinematics(q):
    """Pure DH math, no MuJoCo: returns [T_00, T_01, ..., T_06] in the base frame."""
    frames = [np.eye(4)]
    for (th, al, a, d), qi in zip(DH, q):
        frames.append(frames[-1] @ dh_transform(np.radians(th) + qi, d, a, np.radians(al)))
    return frames


def _quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def build_model():
    """Turn the DH table into a MuJoCo kinematic tree whose body frames ARE the DH frames.

    Body i is placed at T_{i-1,i} (home pose) relative to body i-1. Its hinge joint
    must spin it about z_{i-1}, so we express that axis (and the point it passes
    through, origin_{i-1}) in body i's own coordinates via the inverse transform.
    """
    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, 0]  # zero gravity: the arm stays where the sliders put it
    spec.visual.scale.framelength = 0.1
    spec.visual.scale.framewidth = 0.008
    spec.visual.global_.offwidth = 1280

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.25, 0.28], rgb2=[0.2, 0.2, 0.23], width=300, height=300)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[10, 10])
    world = spec.worldbody
    world.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], material="grid", contype=0, conaffinity=0)

    parent = world.add_body(name="frame 0")  # base frame, fixed
    parent.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.005], rgba=[0.3, 0.3, 0.3, 1])

    d_color, a_color, joint_color = [0.2, 0.5, 0.9, 1], [0.95, 0.6, 0.1, 1], [0.6, 0.6, 0.6, 1]
    for i, (th, al, a, d) in enumerate(DH, start=1):
        A = dh_transform(np.radians(th), d, a, np.radians(al))
        Ainv = np.linalg.inv(A)
        body = parent.add_body(name=f"frame {i}", pos=A[:3, 3], quat=_quat(A[:3, :3]))

        origin_prev = Ainv[:3, 3]                    # origin of frame i-1, seen from frame i
        z_prev = Ainv[:3, :3] @ [0, 0, 1]            # z_{i-1}, seen from frame i
        after_d = (Ainv @ [0, 0, d, 1])[:3]          # point reached after sliding d along z_{i-1}

        body.add_joint(name=f"q{i}", type=mujoco.mjtJoint.mjJNT_HINGE, pos=origin_prev, axis=z_prev,
                       range=[-np.pi, np.pi], damping=2, armature=0.01)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[*(origin_prev - 0.03 * z_prev),
                      *(origin_prev + 0.03 * z_prev)], size=[0.022, 0, 0], rgba=joint_color, contype=0, conaffinity=0)
        if d:  # blue capsule = the d offset (along z_{i-1})
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*origin_prev, *after_d], size=[0.012, 0, 0],
                          rgba=d_color, contype=0, conaffinity=0)
        if a:  # orange capsule = the a offset (along x_i)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*after_d, 0, 0, 0], size=[0.012, 0, 0],
                          rgba=a_color, contype=0, conaffinity=0)

        act = spec.add_actuator(name=f"q{i} (rad, 0 = home)", target=f"q{i}",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-np.pi, np.pi])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0], act.biasprm[1], act.biasprm[2] = 50, -50, -5  # position servo: kp=50, kv=5
        parent = body

    return spec.compile()


def print_frames(frames):
    print(f"{'frame':>5} | {'origin x, y, z (m)':^24} | {'x axis':^14} | {'y axis':^14} | {'z axis':^14}")
    print("-" * 84)
    fmt = lambda v: "[" + " ".join(f"{c:+.0f}" if abs(c - round(c)) < 1e-6 else f"{c:+.2f}" for c in v) + "]"
    for i, T in enumerate(frames):
        p, R = T[:3, 3], T[:3, :3]
        print(f"{i:>5} | {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f} | {fmt(R[:, 0]):^14} | {fmt(R[:, 1]):^14} | {fmt(R[:, 2]):^14}")


def check_against_mujoco(model, data, frames):
    """Proof that MuJoCo and the hand-written DH math agree."""
    worst = 0.0
    for i, T in enumerate(frames):
        b = data.body(f"frame {i}")
        worst = max(worst, np.abs(b.xpos - T[:3, 3]).max(), np.abs(b.xmat.reshape(3, 3) - T[:3, :3]).max())
    print(f"\nMuJoCo vs DH matrix product: max difference = {worst:.1e}  ->  {'MATCH' if worst < 1e-9 else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, nargs=6, default=[0] * 6, metavar="DEG",
                        help="joint angles in degrees, relative to the home pose")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    q = np.radians(args.q)

    model = build_model()
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.ctrl[:] = q
    mujoco.mj_kinematics(model, data)

    frames = forward_kinematics(q)
    print(f"DH frames in the base frame at q = {args.q} deg (offset from home)\n")
    print_frames(frames)
    check_against_mujoco(model, data, frames)
    if args.headless:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY  # draw every DH frame's xyz axes
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY  # write "frame i" next to each
        viewer.cam.lookat[:] = [0.2, 0, 0.3]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 135, -20
        last_print, last_q = 0.0, None
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            if time.time() - last_print > 0.5:  # live end-effector readout when you move a slider
                last_print = time.time()
                q_now = np.round(np.degrees(data.qpos), 1)
                if last_q is None or np.abs(q_now - last_q).max() > 0.5:
                    last_q = q_now
                    p = data.body("frame 6").xpos
                    print(f"q = {q_now} deg  ->  frame 6 origin = [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}] m")
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
