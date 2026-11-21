"""A "hand" that holds the cube and moves it while the robot tries to grasp it (dynamic targets).

The hand is a mocap body (scene.build_model(with_hand=True)): the script sets its position every
physics step, and a weld constraint ties the cube to it while held. When the gripper has closed
around the cube, the hand lets go: a handover. If the gripper misses, the hand keeps holding and
moving, like a person waiting for the robot to try again.

Motion patterns, all around the cube's start position and at a chosen speed (cm/s):
    static   held still (the weld is on, nothing moves)
    circle   a circle of 4 cm radius
    sway     back and forth along a line, +-5 cm (constant speed, turning at the ends)
    jitter   smooth, irregular motion: a sum of slow sine waves on each axis (rms speed = speed)

The cube stays on the table: camera perception assumes the cube rests on the table plane. A cube
held up in the air needs depth sensing (a later stage).
"""
import numpy as np

from scene import GRIPPER_ACTUATOR, HAND_WELD, TCP_SITE

MOTIONS = ["static", "circle", "sway", "jitter"]
CIRCLE_RADIUS = 0.04
SWAY_AMPLITUDE = 0.05
JITTER_FREQS = (0.4, 0.9, 1.7)  # Hz, one sine per frequency and axis (hand-like, stays within ~10 cm)
RELEASE_AFTER = 0.3  # s the gripper must have been closing, with the cube between its fingers
IN_FINGERS = 0.02  # m: cube centre this close to the fingertip centre counts as between the fingers


class HandMotion:
    """Moves the hand each physics step: call it as a Stepper hook, hand(model, data)."""

    def __init__(self, model, data, kind, speed_cm_s, seed):
        if kind not in MOTIONS:
            raise ValueError(f"unknown motion {kind!r}; choose from {MOTIONS}")
        self.kind, self.speed = kind, speed_cm_s / 100
        rng = np.random.default_rng(seed)
        self.start = data.body("cube").xpos.copy()
        self.weld = model.equality(HAND_WELD).id
        self.gripper = model.actuator(GRIPPER_ACTUATOR).id
        self.phase = rng.uniform(0, 2 * np.pi)  # circle: where on the circle the cube starts
        self.turn = rng.choice([-1.0, 1.0])  # circle: direction of travel
        self.direction = np.array([np.cos(self.phase), np.sin(self.phase)])  # sway: the line
        # jitter: random phases; amplitudes so that each sine contributes equally to the rms speed
        self.jitter_phase = rng.uniform(0, 2 * np.pi, (2, len(JITTER_FREQS)))
        omega = 2 * np.pi * np.asarray(JITTER_FREQS)
        self.jitter_amp = self.speed / (omega * np.sqrt(len(JITTER_FREQS)))  # per axis; rms speed over both axes
        self.held, self.closing_since, self.released_at = True, None, None
        data.mocap_pos[0] = self.start
        data.mocap_quat[0] = data.body("cube").xquat
        data.eq_active[self.weld] = 1

    def offset(self, t):
        """Hand position relative to the cube's start, at time t (s)."""
        v = self.speed
        if self.kind == "static" or v == 0:
            return np.zeros(2)
        if self.kind == "circle":
            a = self.phase + self.turn * v * t / CIRCLE_RADIUS
            return CIRCLE_RADIUS * (np.array([np.cos(a), np.sin(a)]) - [np.cos(self.phase), np.sin(self.phase)])
        if self.kind == "sway":
            period = 4 * SWAY_AMPLITUDE / v
            u = (t / period) % 1.0  # triangle wave: 0 -> +A -> 0 -> -A -> 0 at constant speed
            s = 4 * u if u < 0.25 else 2 - 4 * u if u < 0.75 else 4 * u - 4
            return SWAY_AMPLITUDE * s * self.direction
        omega = 2 * np.pi * np.asarray(JITTER_FREQS)
        waves = np.sin(omega * t + self.jitter_phase) - np.sin(self.jitter_phase)  # starts at 0
        return (self.jitter_amp * waves).sum(axis=1)

    def __call__(self, model, data):
        if not self.held:
            return
        data.mocap_pos[0, :2] = self.start[:2] + self.offset(data.time)
        # Let go once the gripper has been closing for a moment with the cube between its fingers.
        closing = data.ctrl[self.gripper] > 0
        between = np.linalg.norm(data.body("cube").xpos - data.site(TCP_SITE).xpos) < IN_FINGERS
        if not (closing and between):
            self.closing_since = None
            return
        if self.closing_since is None:
            self.closing_since = data.time
        if data.time - self.closing_since >= RELEASE_AFTER:
            data.eq_active[self.weld] = 0
            self.held, self.released_at = False, data.time
