"""Tests for motion prediction (predict.py) and its use while tracking (Option C1)."""
import numpy as np
import pytest

from predict import ConstantVelocityKF


def test_learns_a_constant_velocity_exactly():
    kf = ConstantVelocityKF()
    for k in range(40):
        kf.update(k / 20, [0.05 * k / 20, -0.02 * k / 20])
    assert np.allclose(kf.velocity, [0.05, -0.02], atol=1e-4)
    assert np.allclose(kf.position_at(2.15), [0.05 * 2.15, -0.02 * 2.15], atol=1e-4)


def test_keeps_predicting_without_frames():
    kf = ConstantVelocityKF()
    for k in range(20):
        kf.update(k / 20, [0.1 * k / 20, 0.0])
    kf.predict_to(1.5)  # half a second without a measurement
    assert kf.x[0] == pytest.approx(0.15, abs=1e-3)


@pytest.mark.parametrize("speed", [0.05, 0.10])
def test_aiming_ahead_beats_aiming_at_the_old_spot(speed):
    """On a circling cube with camera-like noise, predicting 0.2 s ahead lands much closer to where
    the cube will be than the current estimate does."""
    rng, kf, r = np.random.default_rng(0), ConstantVelocityKF(), 0.04
    path = lambda t: r * np.array([np.cos(speed / r * t), np.sin(speed / r * t)])
    ahead, stay = [], []
    for k in range(200):
        t = k / 20
        kf.update(t, path(t) + rng.normal(0, 0.0003, 2))
        if k > 20:
            ahead.append(np.linalg.norm(kf.position_at(t + 0.2) - path(t + 0.2)))
            stay.append(np.linalg.norm(path(t) - path(t + 0.2)))
    assert np.mean(ahead) < 0.5 * np.mean(stay)


def test_prediction_cuts_the_lag_on_a_swaying_cube():
    """Tracking alone trails a cube swaying at 5 cm/s; aiming 0.2 s ahead removes most of the lag."""
    from test_dynamic import run

    args = ("--motion", "sway", "--speed", "5", "--perception", "camera", "--track", "--seed", "1")
    plain, ahead = run(2, *args), run(2, *args, "--predict", "0.2")
    assert ahead["outcome"] == "OK"
    assert ahead["track_lag_mm"] < 0.6 * plain["track_lag_mm"]
