"""Unit tests for the inverse kinematics (ik.py)."""
import mujoco
import numpy as np
import pytest

from ik import DOWN, robot_hits_scene, solve_ik, solve_ik_multistart
from scene import ARM_JOINTS, HOME_QPOS, TCP_SITE, arm_qpos_ids


def tcp_pose(model, data, q):
    """Forward kinematics: TCP position and orientation (quaternion) at arm joint angles q."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[arm_qpos_ids(model)] = q
    mujoco.mj_kinematics(model, scratch)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, scratch.site(TCP_SITE).xmat)
    return scratch.site(TCP_SITE).xpos.copy(), quat


@pytest.mark.parametrize("seed", range(5))
def test_ik_round_trip(model_data, seed):
    """FK of a random pose gives a target; IK must find a pose that reaches it (position and orientation)."""
    model, data = model_data
    q_true = HOME_QPOS + np.random.default_rng(seed).uniform(-0.5, 0.5, 6)
    pos, quat = tcp_pose(model, data, q_true)
    q, pos_err, rot_err, _, ok = solve_ik_multistart(model, data, TCP_SITE, pos, quat, q_init=HOME_QPOS,
                                                     check_collisions=False)
    assert ok
    reached_pos, _ = tcp_pose(model, data, q)
    assert np.linalg.norm(reached_pos - pos) < 1e-3
    assert rot_err < np.radians(1.0)


def test_ik_unreachable_target_stays_finite_and_within_limits(model_data):
    """Damped least squares must not blow up when the target is out of reach (near-singular arm)."""
    model, data = model_data
    q, pos_err, rot_err, _ = solve_ik(model, data, TCP_SITE, [2.0, 0.0, 0.3], DOWN)
    lo, hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS]].T
    assert np.all(np.isfinite(q))
    assert np.all((q >= lo - 1e-9) & (q <= hi + 1e-9))
    assert pos_err > 0.5  # it reports the miss instead of pretending to converge


def test_collision_check_home_is_free(model_data):
    model, data = model_data
    assert not robot_hits_scene(model, data, HOME_QPOS)


def test_collision_check_detects_floor(model_data):
    """A pose that pushes the gripper below the floor must count as a collision."""
    model, data = model_data
    q, *_ = solve_ik(model, data, TCP_SITE, [0.45, 0.2, -0.10], DOWN)
    assert robot_hits_scene(model, data, q)


def test_multistart_rejects_colliding_solutions(model_data):
    """With collision checks on, a target inside the floor has no acceptable solution."""
    model, data = model_data
    *_, ok = solve_ik_multistart(model, data, TCP_SITE, [0.45, 0.2, -0.10], DOWN, restarts=4)
    assert not ok
