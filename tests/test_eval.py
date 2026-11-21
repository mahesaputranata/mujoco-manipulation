"""Unit tests for the evaluation harness (eval_place.py)."""
import numpy as np
import pytest

from conftest import scenario_args
from eval_place import BOX_CLEARANCE, MAX_RADIUS, MIN_RADIUS, per_pose, sample_trial, wilson_interval
from scene import BOX_POS


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(100, 100)
    assert lo == pytest.approx(0.963, abs=1e-3)  # 100/100 does not mean "always": true rate likely > 96%
    assert hi == pytest.approx(1.0)
    lo, hi = wilson_interval(50, 100)
    assert lo + hi == pytest.approx(1.0)  # symmetric around 50%
    assert wilson_interval(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (80, 100), (499, 500)])
def test_wilson_interval_contains_estimate(k, n):
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


@pytest.mark.parametrize("workspace", ["default", "wide"])
def test_sample_trial_is_deterministic_and_valid(workspace):
    args = scenario_args("--workspace", workspace, "--rand-physics")
    for i in range(50):
        t = sample_trial(i, args)
        assert t == sample_trial(i, args)  # reproducible from (seed, i) alone
        assert MIN_RADIUS < np.hypot(t["x"], t["y"]) < MAX_RADIUS
        assert not (abs(t["x"] - BOX_POS[0]) < BOX_CLEARANCE and abs(t["y"] - BOX_POS[1]) < BOX_CLEARANCE)
        assert -180 <= t["yaw_deg"] <= 180
        assert 0.018 <= t["half"] <= 0.030


def test_repeats_change_only_the_noise():
    args = scenario_args("--pos-noise", "5", "--rand-physics")
    a, b = sample_trial(3, args, 0), sample_trial(3, args, 1)
    assert a["noise_seed"] != b["noise_seed"]
    for key in ("x", "y", "yaw_deg", "half", "mass", "friction"):
        assert a[key] == b[key]


def test_per_pose_aggregates_repeats():
    base = dict(x=0.4, y=0.1, yaw_deg=10.0, half=0.025, mass=0.1, friction=1.0)
    rows = [dict(base, trial=0, outcome=o) for o in ("OK", "OK", "GRASP_MISS", "OK")]
    rows += [dict(base, trial=1, outcome="OK")]
    poses = {p["trial"]: p for p in per_pose(rows)}
    assert poses[0]["runs"] == 4 and poses[0]["ok_rate"] == 0.75
    assert poses[0]["outcomes"] == "GRASP_MISS:1 OK:3"
    assert poses[1]["ok_rate"] == 1.0
