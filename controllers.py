"""Arm controllers: turn the planner's joint-angle target into the 6 arm actuator commands.

The arm's actuators are MuJoCo position servos (from Menagerie):

    force = kp * (ctrl - q) - kd * qdot,   clipped to +-forcerange

A plain PD only pushes when there is an error. To hold the arm up against gravity it has
to push with the gravity torque g(q), so it must sag until kp * error = g(q).

Gravity compensation adds g(q), computed from the robot model, as a feedforward torque.
It is sent through the same servo by offsetting the command:

    ctrl = q_target + g(q) / kp   ->   force = kp * (q_target - q) - kd * qdot + g(q)

Why not compute the whole torque in Python and use torque actuators? The model uses
MuJoCo's implicitfast integrator, which treats the servo's damping term implicitly. Damping
computed in Python is explicit, and with these gains the wrist 3 joint sits at
kd * dt / inertia = 1.99, right at the explicit stability limit of 2. Feedforward through
the servo keeps the stable damping and the servo's torque limits.
"""
import mujoco
import numpy as np

from scene import arm_dof_ids


class ServoPD:
    """The Menagerie position servo as it is: command = joint target."""

    def __init__(self, model):
        pass

    def __call__(self, model, data, q_target):
        return q_target


class GravityCompPD:
    """PD servo + gravity compensation: adds the model's gravity torque g(q) as feedforward."""

    def __init__(self, model):
        self.dofs = arm_dof_ids(model)
        self.kp = model.actuator_gainprm[:6, 0].copy()
        self.scratch = mujoco.MjData(model)  # for g(q): same pose, zero velocity
        self.g = np.zeros(model.nv)

    def gravity(self, model, data):
        """Generalized gravity forces on the arm joints at the current pose.

        Recursive Newton-Euler with zero velocity and zero acceleration leaves only gravity.
        It includes the gripper (part of the model) but not a grasped object: payload is unmodelled.
        """
        s = self.scratch
        s.qpos[:] = data.qpos
        s.qvel[:] = 0
        mujoco.mj_kinematics(model, s)
        mujoco.mj_comPos(model, s)
        mujoco.mj_rne(model, s, 0, self.g)  # flg_acc=0: no inertial term
        return self.g[self.dofs]

    def __call__(self, model, data, q_target):
        return q_target + self.gravity(model, data) / self.kp


CONTROLLERS = {"pd": ServoPD, "pd_gravity": GravityCompPD}
