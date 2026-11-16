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


def robot_hits_scene(model, data, q):
    """True if the arm at joint angles `q` touches the floor or the box (touching the cube is fine)."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[arm_qpos_ids(model)] = q
    mujoco.mj_forward(model, scratch)
    robot_root, cube = model.body("base").id, model.body("cube").id
    for c in scratch.contact[:scratch.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        robot1, robot2 = model.body_rootid[b1] == robot_root, model.body_rootid[b2] == robot_root
        if robot1 != robot2 and cube not in (b1, b2):
            return True
    return False


def solve_ik_multistart(model, data, site_name, target_pos, target_quat=DOWN, q_init=None, restarts=8,
                        pos_tol=1e-3, rot_tol=np.radians(1.0), check_collisions=True, seed=0):
    """solve_ik, but if it gets stuck in a local minimum, retry from other starting poses.

    DLS only follows the local gradient. On a big move (e.g. swinging the base ~100 deg) it
    can wind a joint the long way round into its limit and stall. Tries, in order: q_init,
    the home pose, then `restarts` random poses near home. A solution only counts if it
    converges AND (with check_collisions) the arm doesn't touch the floor or box: IK alone
    happily puts the elbow through the table. Among those, returns the one closest to q_init,
    so the arm doesn't make needlessly large moves.

    Returns (q, pos_err, rot_err, iters, ok). If nothing is ok, q is the closest miss.
    """
    from scene import HOME_QPOS

    rng = np.random.default_rng(seed)  # fixed seed: the same target always gives the same answer
    q_ref = data.qpos[arm_qpos_ids(model)] if q_init is None else np.asarray(q_init)
    starts = [q_ref]
    if restarts:
        starts += [HOME_QPOS] + [HOME_QPOS + rng.uniform(-np.pi, np.pi, 6) * [1, 0.5, 0.5, 1, 1, 1]
                                 for _ in range(restarts)]
    good, fallback = [], None
    for q0 in starts:
        q, pos_err, rot_err, iters = solve_ik(model, data, site_name, target_pos, target_quat, q_init=q0)
        converged = pos_err < pos_tol and rot_err < rot_tol
        if converged and not (check_collisions and robot_hits_scene(model, data, q)):
            if q0 is q_ref:
                return q, pos_err, rot_err, iters, True  # the plain solve already worked: keep it
            good.append((q, pos_err, rot_err, iters, True))
        elif fallback is None or pos_err + rot_err < fallback[1] + fallback[2]:
            fallback = (q, pos_err, rot_err, iters, False)
    if good:
        return min(good, key=lambda r: np.linalg.norm(r[0] - q_ref))
    return fallback
