"""Motion prediction for tracking (Option C1): remember how the cube has been moving, aim ahead.

A constant-velocity Kalman filter on the cube's horizontal position. Each accepted camera frame
updates it; between frames (or when a frame is rejected) it keeps predicting. Tracking then aims at
where the cube will be `lead` seconds from now: position + velocity x lead. The lead compensates the
arm's lag behind a moving cube, measured in Option B as about speed x 0.2 s.

State x = [px, py, vx, vy]. The motion model says velocity stays constant, up to a random
acceleration (process noise ACCEL_STD): a hand changes direction, but not instantly.
"""
import numpy as np

ACCEL_STD = 0.5  # m/s^2: how hard the cube may accelerate. The test motions reach 0.25-0.6 (a 10 cm/s
# circle of 4 cm: 0.25). Larger trusts each frame more (noisier velocity), smaller lags in turns.
MEAS_STD = 0.0005  # m: camera noise per frame. Measured in sim: ~0.1 mm (plus a steady ~1 mm bias, which
# does not disturb velocity). 0.5 mm leaves room for pixel steps on a moving cube. A real webcam is
# much noisier: re-measure on hardware, since this sets how much the filter trusts each frame.


class ConstantVelocityKF:
    def __init__(self, accel_std=ACCEL_STD, meas_std=MEAS_STD):
        self.q, self.r = accel_std ** 2, meas_std ** 2
        self.x = None  # state, set by the first measurement
        self.P = None
        self.t = None

    def _transition(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        g = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])  # acceleration -> state
        return F, self.q * g @ g.T

    def predict_to(self, t):
        """Advance the filter to time t (no measurement)."""
        if self.x is None or t <= self.t:
            return
        F, Q = self._transition(t - self.t)
        self.x, self.P, self.t = F @ self.x, F @ self.P @ F.T + Q, t

    def update(self, t, xy):
        """Fuse a measured position at time t."""
        xy = np.asarray(xy, float)
        if self.x is None:
            self.x, self.t = np.array([xy[0], xy[1], 0.0, 0.0]), t
            self.P = np.diag([self.r, self.r, 0.1 ** 2, 0.1 ** 2])  # velocity unknown: up to ~10 cm/s
            return
        self.predict_to(t)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + self.r * np.eye(2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (xy - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P

    def position_at(self, t):
        """Predicted position at time t (>= the filter's time), without changing the filter."""
        dt = max(0.0, t - self.t)
        return self.x[:2] + self.x[2:] * dt

    @property
    def velocity(self):
        return self.x[2:].copy()
