"""Watch the DH table being applied, one sub-step at a time.

For every row i, a big "moving frame" starts at frame i-1 and performs the
4 standard-DH sub-steps, exactly like you do on paper:

    1) rotate about z by theta    2) slide along z by d
    3) slide along x by a         4) twist about x by alpha

When the 4 steps are done it leaves frame i behind (small axes) and starts the
next row. Colours: red = x, green = y, blue = z, yellow line = the axis being
rotated about. Blue links = d slides, orange links = a slides.

Keys (click the window first):
    Space      pause / resume
    Right      jump to the next sub-step
    Left       go back one sub-step
    Backspace  restart from frame 0

Run:  python fundamentals/dh_animate.py [--speed 0.5]
The terminal prints each frame's axes when it is finished, to compare with paper.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from dh_viewer import DH, forward_kinematics, rot_x, rot_z, trans

RED, GREEN, BLUE = [0.9, 0.15, 0.15, 1], [0.15, 0.8, 0.15, 1], [0.2, 0.4, 1.0, 1]
YELLOW, D_COLOR, A_COLOR = [1.0, 0.85, 0.1, 0.8], [0.3, 0.55, 0.95, 1], [0.95, 0.6, 0.1, 1]
MOVE_TIME, HOLD_TIME, SKIP_TIME = 2.0, 0.8, 1.2  # seconds at speed 1


# ---- The sequence of sub-steps ----------------------------------------------
def build_ops():
    """Flatten the DH table into 24 sub-steps: (row, kind, value, symbol)."""
    ops = []
    for i, (th, al, a, d) in enumerate(DH, start=1):
        ops += [(i, "rz", np.radians(th), f"theta{i}"), (i, "tz", d, f"d{i}"),
                (i, "tx", a, f"a{i}"), (i, "rx", np.radians(al), f"alpha{i}")]
    return ops


def op_matrix(kind, value):
    return {"rz": rot_z, "rx": rot_x}[kind](value) if kind in ("rz", "rx") else \
        trans(0, 0, value) if kind == "tz" else trans(value, 0, 0)


def describe(op):
    i, kind, v, sym = op
    text = {
        "rz": f"1) Rotate about z{i-1} by {sym} = {np.degrees(v):.0f} deg   (joint {i} spins about this axis)",
        "tz": f"2) Slide along z{i-1} by {sym} = {v:.4f} m",
        "tx": f"3) Slide along the NEW x{i} by {sym} = {v:.4f} m",
        "rx": f"4) Twist about x{i} by {sym} = {np.degrees(v):.0f} deg   (tilts the next joint axis z{i})",
    }[kind]
    if v == 0:
        text += "   ->  zero, nothing happens"
    elif kind == "rx" and np.isclose(np.degrees(v), 270):
        text += "   (270 deg ends where -90 deg does)"
    return text


def state_at(ops, k, s):
    """Return (frames finished so far, link segments drawn so far, moving frame T)."""
    T = np.eye(4)
    frames, links = [T], []
    for j, (i, kind, v, _) in enumerate(ops[:k + 1]):
        frac = s if j == k else 1.0
        T_next = T @ op_matrix(kind, frac * v)
        if kind in ("tz", "tx") and v != 0 and frac > 0:
            links.append((T[:3, 3].copy(), T_next[:3, 3].copy(), D_COLOR if kind == "tz" else A_COLOR))
        T = T_next
        if kind == "rx" and frac == 1.0:
            frames.append(T)
    return frames, links, T


# ---- Drawing into a MuJoCo scene -------------------------------------------
def _add(scn, gtype, p_from, p_to, rgba, width, label=""):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(p_from, float), np.asarray(p_to, float))
    g.label = label
    scn.ngeom += 1


def draw_axes(scn, T, length, width, label=""):
    o = T[:3, 3]
    for col, rgba in zip(range(3), (RED, GREEN, BLUE)):
        _add(scn, mujoco.mjtGeom.mjGEOM_ARROW, o, o + length * T[:3, col], rgba, width,
             label if col == 2 else "")


def draw_state(scn, ops, k, s):
    frames, links, T = state_at(ops, k, s)
    for p0, p1, rgba in links:
        _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, p0, p1, rgba, 0.008)
    for n, F in enumerate(frames):
        # Frames sharing an origin (e.g. 4 and 5 at the wrist) get one combined label.
        same = [m for m, G in enumerate(frames) if np.allclose(G[:3, 3], F[:3, 3])]
        label = ("frame " + ", ".join(map(str, same))) if n == same[-1] else ""
        draw_axes(scn, F, 0.1, 0.006, label)
    if k < len(ops):
        _, kind, v, _ = ops[k]
        if kind in ("rz", "rx") and v != 0:  # show the axis we're rotating about
            start = state_at(ops, k, 0.0)[2]
            o, axis = start[:3, 3], start[:3, 2 if kind == "rz" else 0]
            _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, o - 0.2 * axis, o + 0.2 * axis, YELLOW, 0.003)
        draw_axes(scn, T, 0.17, 0.011, "moving")
    return frames, T


def fmt_vec(v):
    return "[" + " ".join(f"{0.0 if abs(c) < 1e-9 else c:+.0f}" if abs(c - round(c)) < 1e-6 else f"{c:+.2f}"
                          for c in v) + "]"


def axes_text(T):
    return (f"origin [{T[0, 3]:+.4f} {T[1, 3]:+.4f} {T[2, 3]:+.4f}]\n"
            f"x {fmt_vec(T[:3, 0])}   y {fmt_vec(T[:3, 1])}   z {fmt_vec(T[:3, 2])}")


def build_world():
    spec = mujoco.MjSpec()
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.25, 0.28], rgb2=[0.2, 0.2, 0.23], width=300, height=300)
    spec.add_material(name="grid", textures=["", "grid"], texrepeat=[10, 10])
    spec.worldbody.add_light(pos=[0, 0, 2], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    # Floor sits slightly below z=0 so frame 0's x and y arrows aren't buried in it.
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], pos=[0, 0, -0.015], material="grid")
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.003], pos=[0, 0, -0.012],
                            rgba=[0.35, 0.35, 0.35, 1])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    model = spec.compile()
    return model, mujoco.MjData(model)


# ---- Interactive player ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0, help="0.5 = half speed")
    args = parser.parse_args()

    ops = build_ops()
    model, data = build_world()
    mujoco.mj_forward(model, data)
    player = {"k": 0, "t": 0.0, "paused": False, "printed": -1}

    def on_key(key):
        if key == 32:     # Space
            player["paused"] = not player["paused"]
        elif key == 262:  # Right
            player["k"], player["t"] = min(player["k"] + 1, len(ops)), 0.0
        elif key == 263:  # Left
            player["k"], player["t"] = max(player["k"] - 1, 0), 0.0
        elif key == 259:  # Backspace
            player.update(k=0, t=0.0, printed=-1)

    print("Frame-by-frame result (compare each one with your paper):\n")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SELECTION  # keeps MuJoCo's own labels off
        viewer.cam.lookat[:] = [0.17, 0, 0.32]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.15, 140, -15
        last = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()
            dt, last = (now - last) * args.speed, now
            k = player["k"]

            if k < len(ops):
                moving = ops[k][2] != 0
                total = MOVE_TIME + HOLD_TIME if moving else SKIP_TIME
                if not player["paused"]:
                    player["t"] += dt
                s = min(player["t"] / MOVE_TIME, 1.0) if moving else 1.0
                s = 3 * s**2 - 2 * s**3  # smooth start and stop
                if player["t"] >= total:
                    player["k"], player["t"] = k + 1, 0.0
                headline = f"Row {ops[k][0]}  ->  building frame {ops[k][0]}  from frame {ops[k][0] - 1}"
                step_text = describe(ops[k])
            else:
                s, headline, step_text = 1.0, "Done: all 6 frames built", "Left = step back,  Backspace = restart"

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                frames, T = draw_state(viewer.user_scn, ops, k, s)
                viewer.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, headline, ""),
                    (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, step_text + "\n\nmoving frame:\n" + axes_text(T), ""),
                    (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     "Space  pause" + (" (PAUSED)" if player["paused"] else "") +
                     "\nRight  next step\nLeft   back\nBksp   restart", ""),
                ])
            viewer.sync()

            # Print each newly finished frame once (again if you step back past it).
            n_done = len(frames) - 1
            for n in range(player["printed"] + 1, n_done + 1):
                print(f"frame {n}: {axes_text(frames[n]).replace(chr(10), '   ')}")
            player["printed"] = n_done
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
