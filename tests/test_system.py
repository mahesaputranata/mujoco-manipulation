"""Integration and regression tests: full pick-and-place runs in the simulator.

These are the guard rails for future milestones (new controllers, perception, ...):
- every controller must complete a basic pick-and-place,
- failures that were fixed must stay fixed,
- the improved system must not drop below its success floor,
- runs must stay reproducible from their seed.
"""
import numpy as np
import pytest

from conftest import scenario_args
from controllers import CONTROLLERS
from demo_reach import ARRIVE_TIMEOUT, Stepper, pick_and_place
from eval_place import controller_for, run_trial, sample_trial, setup_trial, _init_worker
from scene import build_model, place_cube


def run(trial_index, *argv):
    """Run one eval_place.py trial in-process; returns the result row."""
    args = scenario_args(*argv)
    _init_worker(args)
    return run_trial(sample_trial(trial_index, args))


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_pick_and_place_with_every_controller(controller):
    model, data = build_model()
    place_cube(model, data, (0.45, 0.20), 0.0)
    metrics = {}
    outcome, attempts = pick_and_place(model, data, Stepper(model, data, controller=controller), metrics=metrics)
    assert outcome == "OK" and attempts == 1
    assert metrics["place_error_mm"] < 15
    assert metrics["wait_s"] <= 2 * ARRIVE_TIMEOUT


@pytest.mark.parametrize("yaw_deg", [10, 20, 30])
def test_gravity_comp_with_wait_grasps_precisely(yaw_deg):
    """The headline result: gravity compensation + wait-to-arrive grasps within ~1 mm, square to the cube.

    Twist only shows a misaligned grasp at some yaws: around 30-40 deg an unaligned but well-centred
    gripper holds the cube by its corners without turning it. 10 and 20 deg catch it.
    """
    model, data = build_model()
    place_cube(model, data, (0.45, 0.20), np.radians(yaw_deg))
    metrics = {}
    outcome, _ = pick_and_place(model, data, Stepper(model, data, controller="pd_gravity"), metrics=metrics)
    assert outcome == "OK"
    assert metrics["grasp_offset_mm"] < 1.5
    assert abs(metrics["grasp_twist_deg"]) < 2.0  # fingers met the cube square, didn't drag it round
    assert metrics["reach_error_mm"] < 3.0


# --- fixed failures stay fixed ------------------------------------------------------------------

def test_ik_local_minimum_trial_7():
    """Wide workspace, trial 7: the original planner's IK gets stuck; multi-start IK solves it."""
    assert run(7, "--workspace", "wide", "--baseline")["outcome"] == "IK_FAIL"
    assert run(7, "--workspace", "wide")["outcome"] == "OK"


def test_cube_next_to_box_wall_trial_85():
    """Wide workspace, trial 85: the aligned grasp would put a finger in the box wall; the 90 deg grasp works."""
    row = run(85, "--workspace", "wide")
    assert row["outcome"] == "OK" and row["attempts"] == 1


# --- success floor and reproducibility ----------------------------------------------------------

@pytest.mark.slow
def test_success_floor_hardest_setting():
    """Improved system, hardest setting: at least 19 of the first 20 trials must succeed."""
    argv = ("--workspace", "wide", "--pos-noise", "5", "--yaw-noise", "5", "--rand-physics")
    ok = sum(run(i, *argv)["outcome"] == "OK" for i in range(20))
    assert ok >= 19


def test_trials_are_reproducible():
    argv = ("--workspace", "wide", "--pos-noise", "5", "--yaw-noise", "5", "--rand-physics")
    a, b = run(3, *argv), run(3, *argv)
    for key in ("outcome", "attempts", "grasp_offset_mm", "place_error_mm", "final_x", "final_y"):
        assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))


def test_baseline_uses_original_control():
    args = scenario_args("--baseline")
    assert controller_for(sample_trial(0, args), args) == "pd"
    assert controller_for(sample_trial(0, scenario_args()), scenario_args()) == "pd_gravity"
    _, _, kwargs = setup_trial(sample_trial(0, args), args)
    assert kwargs["arrive"] is False and kwargs["align"] is False and kwargs["ik_restarts"] == 0
