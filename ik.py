"""Damped least-squares (DLS) inverse kinematics, written from scratch.

Goal: find arm joint angles q so the tool site reaches a target position and
orientation.

Idea (one iteration):
    1. Forward kinematics: where is the tool now?            -> x(q)
    2. Error to the target (3 position + 3 rotation numbers) -> e  (6,)
    3. Jacobian: how the tool moves per joint velocity       -> J  (6 x 6)
       e ≈ J dq   for a small joint change dq
    4. Solve for dq. Plain inverse J^-1 blows up near singularities (e.g. arm
       fully stretched), so we use the damped version:
           dq = J^T (J J^T + λ² I)^-1 e
       λ trades a little accuracy for a lot of stability.
    5. q <- q + dq, clamp to joint limits, repeat until the error is tiny.
"""
import mujoco
import numpy as np

from scene import arm_dof_ids, arm_qpos_ids

# Quaternion (w, x, y, z) for "tool z-axis points straight down": 180° about world x.
DOWN = np.array([0.0, 1.0, 0.0, 0.0])


def pose_error(data, site_id, target_pos, target_quat):
    """6-vector error [position (m); rotation (rad, axis*angle)] from current site pose to target."""
    pos_err = target_pos - data.site_xpos[site_id]

    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
    rot_err = np.empty(3)
    mujoco.mju_subQuat(rot_err, target_quat, site_quat)  # rotation taking current -> target
    # subQuat expresses the rotation in the *site's local frame*, but the Jacobian
    # is in *world* coordinates, so rotate it into the world frame. Skipping this
    # makes the solver push the wrist the wrong way and never converge.
    rot_err = data.site_xmat[site_id].reshape(3, 3) @ rot_err
    return np.concatenate([pos_err, rot_err])


def solve_ik(model, data, site_name, target_pos, target_quat=DOWN, q_init=None,
             damping=1e-2, max_iters=200, tol=1e-4, step_limit=0.2):
    """Return 6 arm joint angles that place `site_name` at the target pose.

    Uses a scratch copy of `data`, so the running simulation is not touched.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    qadr, dofs = arm_qpos_ids(model), arm_dof_ids(model)
    if q_init is not None:
        scratch.qpos[qadr] = q_init

    site_id = model.site(site_name).id
    lo, hi = model.jnt_range[[model.dof_jntid[d] for d in dofs]].T
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for i in range(max_iters):
        mujoco.mj_kinematics(model, scratch)  # step 1: forward kinematics
        mujoco.mj_comPos(model, scratch)      # needed by mj_jacSite
        err = pose_error(scratch, site_id, np.asarray(target_pos), target_quat)  # step 2
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)  # step 3
        J = np.vstack([jacp, jacr])[:, dofs]  # keep only the 6 arm columns

        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)  # step 4
        dq = np.clip(dq, -step_limit, step_limit)  # small steps converge more reliably

        q = scratch.qpos[qadr] + dq  # step 5
        scratch.qpos[qadr] = np.clip(q, lo, hi)

    return scratch.qpos[qadr].copy(), np.linalg.norm(err[:3]), np.linalg.norm(err[3:]), i
