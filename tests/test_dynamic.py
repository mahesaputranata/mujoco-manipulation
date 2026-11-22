"""Tests for dynamic targets: the hand that holds and moves the cube (hand_motion.py)."""
import mujoco
import numpy as np
import pytest

from conftest import scenario_args
from hand_motion import MOTIONS, HandMotion
from scene import HAND_WELD, build_model, place_cube


def hand_at(motion, speed, seed=3):
    model, data = build_model(with_hand=True)
    place_cube(model, data, (0.45, 0.20), 0.3)
    return model, data, HandMotion(model, data, motion, speed, seed)


@pytest.mark.parametrize("motion", [m for m in MOTIONS if m != "static"])
@pytest.mark.parametrize("speed", [1.0, 10.0])
def test_motion_has_the_requested_speed(motion, speed):
    _, _, hand = hand_at(motion, speed)
    t = np.arange(0, 20, 0.002)
    p = np.array([hand.offset(x) for x in t])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / 0.002
    assert np.allclose(p[0], 0)  # starts where the cube is
    assert 100 * np.sqrt(np.mean(v ** 2)) == pytest.approx(speed, rel=0.02)  # rms speed in cm/s
    assert np.linalg.norm(p, axis=1).max() < 0.15  # stays near the start


def test_static_hand_does_not_move():
    _, _, hand = hand_at("static", 5.0)
    assert np.allclose(hand.offset(3.7), 0)


def test_hand_carries_the_cube():
    model, data, hand = hand_at("circle", 5.0)
    for _ in range(1000):  # 2 s
        hand(model, data)
        mujoco.mj_step(model, data)
    assert hand.held and data.eq_active[model.equality(HAND_WELD).id]
    lag = np.linalg.norm(data.mocap_pos[0, :2] - data.body("cube").xpos[:2])
    assert lag < 0.005  # the weld keeps the cube within a few mm of the hand
    assert np.linalg.norm(data.body("cube").xpos[:2] - hand.start[:2]) > 0.02  # and it really moved


def test_scene_without_hand_is_unchanged():
    """Only scenes built for dynamic targets get the hand, so all other results stay identical."""
    model, _ = build_model()
    welds = [model.equality(i).name for i in range(model.neq)]  # the gripper has its own linkage constraints
    assert model.nmocap == 0 and HAND_WELD not in welds
    assert len(welds) == build_model(with_hand=True)[0].neq - 1


def run(trial_index, *argv):
    import eval_place

    args = scenario_args(*argv)
    eval_place._init_worker(args)
    return eval_place.run_trial(eval_place.sample_trial(trial_index, args))


def test_handover_of_a_held_cube():
    """A cube held still: the robot grasps it, the hand lets go, the cube ends up in the box."""
    row = run(0, "--motion", "static")
    assert row["outcome"] == "OK" and row["handed_over"] == 1


def test_baseline_look_once_misses_a_moving_cube():
    """Baseline (dynamic_sweep.py): the planner looks once, so a cube moving 1 cm/s has travelled
    ~35 mm by the time the gripper closes, and the grasp misses. Update this test when closed-loop
    tracking is added: it documents the limitation the next stage has to remove."""
    row = run(0, "--motion", "circle", "--speed", "1")
    assert row["outcome"] == "GRASP_MISS"
    assert row["target_shift_mm"] > 25
    assert row["look_age_s"] > 3


def test_motion_seed_is_the_same_at_every_speed():
    import eval_place

    a = eval_place.sample_trial(5, scenario_args("--motion", "sway", "--speed", "1"))
    b = eval_place.sample_trial(5, scenario_args("--motion", "sway", "--speed", "10"))
    assert a["motion_seed"] == b["motion_seed"] and (a["x"], a["y"]) == (b["x"], b["y"])


def test_look_again_catches_what_look_once_misses():
    """Option A: a second look above the cube halves the look's age (3.8 s -> 1.8 s), so the cube
    moving 1 cm/s that the look-once planner misses (see the baseline test) is caught."""
    row = run(0, "--motion", "circle", "--speed", "1", "--perception", "camera", "--look-again")
    assert row["outcome"] == "OK"
    assert row["relook_seen"] == 1  # the gripper above the cube does not block the camera's view
    assert row["look_age_s"] < 2.5
    assert row["relook_shift_mm"] > 15  # moved ~2 s x 1 cm/s = 20 mm between the two looks


def test_look_again_with_a_still_cube():
    row = run(3, "--perception", "camera", "--look-again")
    assert row["outcome"] == "OK" and row["relook_seen"] == 1 and row["relook_shift_mm"] < 3


def test_tracking_catches_a_cube_that_look_again_misses():
    """Option B: watching the cube every frame of the descent catches a cube circling at 2 cm/s,
    which both look-once and look-again miss (0/100 in dynamic_sweep.py)."""
    args = ("--motion", "circle", "--speed", "2", "--perception", "camera")
    assert run(0, *args, "--look-again")["outcome"] == "GRASP_MISS"
    row = run(0, *args, "--track")
    assert row["outcome"] == "OK"
    assert row["target_shift_mm"] < 12  # aimed at where the cube is, not where it was
    assert row["frames_lost"] <= 5  # the silhouette fit keeps seeing it under the gripper's shadow


def test_tracking_recovers_instead_of_cascading():
    """Trial 8 at 10 cm/s once lost 56 frames in a row: the silhouette fit started from a stale
    guess 4-9 cm away and every later frame started further off, although the cube stayed 80-90%
    visible. Starting each fit also from the red blob's centre lets tracking recover."""
    row = run(8, "--motion", "circle", "--speed", "10", "--perception", "camera", "--track")
    assert row["outcome"] == "OK"
    assert row["frames_lost"] <= 10
