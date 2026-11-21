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

CUBE_HALF = 0.025  # half the cube's edge length (m)
CUBE_MASS = 0.1
CUBE_START = (0.45, 0.20, CUBE_HALF)
BOX_POS = (0.40, -0.30, 0.0)

# Fixed "tripod" camera in front of the table, looking down at the workspace at about 45 deg,
# like a webcam on a stand at a desk (eye-to-hand). Used by perception.py.
SCENE_CAMERA = "scene_cam"
CAMERA_EYE = (1.02, 0.15, 0.80)
CAMERA_TARGET = (0.46, 0.15, 0.0)
CAMERA_FOVY = 55.0  # vertical field of view, degrees

# Printed ArUco tags (DICT_4X4_50) lying flat on the table at known positions, like a mat with a
# tag in each corner: two near the robot base, two beyond the far edge of the workspace. All are in
# the camera's view and outside the area where the cube can be. The camera can locate itself from
# them (perception.py, --calibration marker). Each entry: id -> (x, y) of the tag's centre.
# MARKER_SIZE is the black square's side; the printed sheet adds a white border of 0.2 x that.
MARKERS = {0: (0.18, 0.10), 1: (0.17, -0.12), 2: (0.79, -0.22), 3: (0.79, 0.48)}
MARKER_SIZE = 0.08
MARKER_Z = 0.0005  # just above the floor, so the two surfaces don't flicker


def look_at_xyaxes(eye, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera orientation (x and y axes in world frame) for a camera at `eye` looking at `target`.

    A MuJoCo camera looks along its -z axis, with x to the right of the image and y up.
    """
    forward = np.asarray(target, float) - np.asarray(eye, float)
    z = -forward / np.linalg.norm(forward)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return [*x, *y]


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


def build_spec(cube_half=CUBE_HALF, cube_mass=CUBE_MASS, cube_friction=1.0) -> mujoco.MjSpec:
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
    cube = world.add_body(name="cube", pos=[CUBE_START[0], CUBE_START[1], cube_half])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cube_half] * 3,
                  rgba=[0.85, 0.15, 0.15, 1], mass=cube_mass, friction=[cube_friction, 0.005, 0.0001])

    _add_open_box(world, "box", BOX_POS)
    sheet = MARKER_SIZE * 1.4 / 2  # half side of the printed sheet, white border included
    for marker_id, (mx, my) in MARKERS.items():
        name = f"aruco{marker_id}"
        arm.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                        file=str(ROOT / "assets" / f"aruco_4x4_id{marker_id}.png"))
        arm.add_material(name=name, textures=["", name], texrepeat=[1, 1], specular=0, shininess=0)
        world.add_geom(name=f"marker{marker_id}", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[sheet, sheet, 0.01],
                       pos=[mx, my, MARKER_Z], material=name, contype=0, conaffinity=0)
    world.add_camera(name=SCENE_CAMERA, pos=CAMERA_EYE, fovy=CAMERA_FOVY,
                     xyaxes=look_at_xyaxes(CAMERA_EYE, CAMERA_TARGET))
    return arm


def build_model(**cube):
    """Compile the scene. Pass cube_half / cube_mass / cube_friction to change the cube."""
    model = build_spec(**cube).compile()
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


def yaw_quat(yaw):
    """Quaternion (w, x, y, z) for a rotation of `yaw` rad about the world z-axis."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def cube_yaw(data):
    """Cube rotation about world z (rad), read from its orientation matrix."""
    R = data.body("cube").xmat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def place_cube(model, data, xy, yaw):
    """Put the cube flat on the floor at `xy`, rotated by `yaw` rad, at rest."""
    jnt = model.joint("cube_joint")
    half = model.geom("cube_geom").size[0]
    data.qpos[jnt.qposadr[0]:jnt.qposadr[0] + 7] = [xy[0], xy[1], half, *yaw_quat(yaw)]
    data.qvel[jnt.dofadr[0]:jnt.dofadr[0] + 6] = 0
    mujoco.mj_forward(model, data)


if __name__ == "__main__":
    m, d = build_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu}")
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])
    print("TCP at home:", d.site(TCP_SITE).xpos.round(3))
