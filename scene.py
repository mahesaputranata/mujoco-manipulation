"""Builds the simulation scene: UR5e arm + Robotiq 2F-85 gripper, a cube, and an open box.

The robot models come from MuJoCo Menagerie (third_party/menagerie). Instead of
hand-merging XML files, we load each model as an MjSpec and attach the gripper
to the arm's flange site in code.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).parent
MENAGERIE = ROOT / "third_party" / "menagerie"

# Joint angles (rad) for a comfortable "ready" pose, gripper pointing down.
HOME_QPOS = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_ACTUATOR = "gripper/fingers_actuator"  # ctrl 0 = open, 255 = closed
TCP_SITE = "gripper/pinch"  # tool centre point, between the fingertips

CUBE_START = (0.45, 0.20, 0.025)
BOX_POS = (0.40, -0.30, 0.0)


def _add_open_box(world, name, pos, inner=0.14, wall=0.01, height=0.06):
    """An open-top box made of 5 thin slabs, so objects can be dropped into it."""
    box = world.add_body(name=name, pos=pos)
    half, h = inner / 2 + wall, height / 2
    rgba = [0.55, 0.35, 0.2, 1]
    slabs = [
        ([0, 0, wall / 2], [half, half, wall / 2]),          # floor
        ([half - wall / 2, 0, h], [wall / 2, half, h]),       # +x wall
        ([-half + wall / 2, 0, h], [wall / 2, half, h]),      # -x wall
        ([0, half - wall / 2, h], [half, wall / 2, h]),       # +y wall
        ([0, -half + wall / 2, h], [half, wall / 2, h]),      # -y wall
    ]
    for p, size in slabs:
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=size, rgba=rgba)
    box.add_site(name=f"{name}_target", pos=[0, 0, height + 0.05], size=[0.01] * 3, rgba=[0, 1, 0, 0.3])


def build_spec() -> mujoco.MjSpec:
    arm = mujoco.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur5e" / "ur5e.xml"))
    gripper = mujoco.MjSpec.from_file(str(MENAGERIE / "robotiq_2f85" / "2f85.xml"))

    # Keyframes would no longer match the combined model's size; we set the pose in code.
    for key in list(arm.keys):
        arm.delete(key)
    for key in list(gripper.keys):
        gripper.delete(key)

    # The gripper model relies on these contact settings for a stable grip.
    arm.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10

    arm.attach(gripper, site=arm.site("attachment_site"), prefix="gripper/")

    # Environment: sky, floor, light.
    arm.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                    rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    arm.add_material(name="grid", textures=["", "grid"], texrepeat=[5, 5], reflectance=0.2)
    world = arm.worldbody
    world.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")

    # Object to pick: a 5 cm red cube with a free joint (6-DOF, falls under gravity).
    cube = world.add_body(name="cube", pos=CUBE_START)
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025] * 3, rgba=[0.85, 0.15, 0.15, 1], mass=0.1)

    _add_open_box(world, "box", BOX_POS)
    return arm


def build_model():
    model = build_spec().compile()
    data = mujoco.MjData(model)
    reset_home(model, data)
    return model, data


def arm_qpos_ids(model):
    return np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])


def arm_dof_ids(model):
    return np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])


def reset_home(model, data):
    mujoco.mj_resetData(model, data)
    data.qpos[arm_qpos_ids(model)] = HOME_QPOS
    data.ctrl[:6] = HOME_QPOS  # the 6 arm actuators come first
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    m, d = build_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu}")
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])
    print("TCP at home:", d.site(TCP_SITE).xpos.round(3))
