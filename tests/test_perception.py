"""Tests for camera perception (perception.py).

Geometry tests need no rendering: they draw a synthetic image of a cube whose pose is known.
Rendering tests need an OpenGL context. They are skipped if none is available, unless the
environment variable REQUIRE_GL=1 is set (CI sets it, so they cannot silently disappear there).
"""
import os

import cv2
import numpy as np
import pytest

from conftest import scenario_args
from demo_reach import wrap_cube_angle
from perception import CameraPerception, PinholeCamera, estimate_from_image, largest_blob, red_mask
from scene import CAMERA_TARGET, build_model, look_at_xyaxes, place_cube

BRIGHT_RED, DARK_RED, FLOOR = (220, 40, 40), (120, 20, 20), (40, 60, 90)


@pytest.fixture
def camera():
    model, data = build_model()
    return PinholeCamera(model, data)


def cube_corners(x, y, yaw, half):
    """The 8 corners of a cube resting on the table: 4 bottom, then 4 top."""
    c, s = np.cos(yaw), np.sin(yaw)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    xy = square @ np.array([[c, s], [-s, c]]) + [x, y]
    return np.vstack([np.c_[xy, np.zeros(4)], np.c_[xy, np.full(4, 2 * half)]])


def synthetic_image(camera, x, y, yaw, half, ss=4):
    """Draw the cube as the camera would see it: dark side faces, bright top face (lit from above).

    Drawn at ss times the resolution and shrunk by area averaging, so edges are anti-aliased like
    MuJoCo's renders (perception.EDGE_PUSH is tuned for that).
    """
    big = np.full((camera.height * ss, camera.width * ss, 3), FLOOR, np.uint8)
    px = camera.project(cube_corners(x, y, yaw, half)) * ss  # pixel-centre coordinates, scaled
    px = np.round(px - 0.5).astype(np.int32)
    cv2.fillConvexPoly(big, cv2.convexHull(px), DARK_RED)  # whole silhouette
    cv2.fillConvexPoly(big, px[4:], BRIGHT_RED)  # top face on top
    return cv2.resize(big, (camera.width, camera.height), interpolation=cv2.INTER_AREA)


# --- geometry, no rendering ---------------------------------------------------------------------

def test_look_at_is_orthonormal_and_aims_at_target(camera):
    x, y = np.reshape(look_at_xyaxes((1, 0.2, 0.7), (0.3, 0.1, 0.0)), (2, 3))
    assert np.isclose(np.linalg.norm(x), 1) and np.isclose(np.linalg.norm(y), 1) and np.isclose(x @ y, 0)
    assert np.isclose(x[2], 0)  # no camera roll: the image's horizontal axis stays level
    u, v = camera.project(np.array(CAMERA_TARGET))[0]
    assert abs(u - camera.width / 2) < 1 and abs(v - camera.height / 2) < 1


def test_pixel_plane_round_trip(camera):
    uv = np.array([[0, 0], [320, 240], [639, 479], [100, 400]], float)
    for z in (0.0, 0.05):
        assert np.allclose(camera.project(camera.pixels_to_plane(uv, z)), uv, atol=1e-6)


def test_red_mask_and_largest_blob():
    img = np.full((50, 50, 3), FLOOR, np.uint8)
    img[5:15, 5:15] = BRIGHT_RED  # big red square
    img[30:33, 30:33] = BRIGHT_RED  # small red speck
    img[20:25, 20:25] = (140, 90, 50)  # brown, like the box: must not count as red
    mask = red_mask(img)
    assert mask.sum() == 100 + 9
    assert largest_blob(mask).sum() == 100


@pytest.mark.parametrize("x,y,yaw_deg,half", [(0.45, 0.20, 30, 0.025), (0.30, 0.00, -10, 0.020),
                                              (0.60, 0.35, 44, 0.030), (0.40, -0.10, 0, 0.025)])
def test_estimate_from_synthetic_image(camera, x, y, yaw_deg, half):
    """The geometry recovers a known cube pose, including its size, from the top face only."""
    est = estimate_from_image(synthetic_image(camera, x, y, np.radians(yaw_deg), half), camera)
    assert est is not None
    assert np.linalg.norm(est["pos"][:2] - [x, y]) < 0.003
    assert abs(np.degrees(wrap_cube_angle(est["yaw"] - np.radians(yaw_deg)))) < 2.0
    assert abs(est["half"] - half) < 0.002


def test_whole_blob_centre_is_biased(camera):
    """Why the top face is used: the centre of the whole red blob (top + visible sides) is biased.

    Projected onto the table or the top-face plane it is off by about 2 cm. Projected onto the
    cube's mid-height it is a decent shortcut (2-4 mm), but the top-face fit is better, and it
    also gives the yaw and the size.
    """
    x, y, half = 0.45, 0.20, 0.025
    img = synthetic_image(camera, x, y, 0.0, half)
    v, u = np.nonzero(red_mask(img))
    centre = np.array([[u.mean(), v.mean()]])
    off = {z: np.linalg.norm(camera.pixels_to_plane(centre, z)[0, :2] - [x, y]) for z in (0.0, half, 2 * half)}
    assert off[0.0] > 0.015 and off[2 * half] > 0.015
    est = estimate_from_image(img, camera)
    assert np.linalg.norm(est["pos"][:2] - [x, y]) < off[half]


def test_nothing_red_means_not_seen(camera):
    img = np.full((camera.height, camera.width, 3), FLOOR, np.uint8)
    assert estimate_from_image(img, camera) is None


# --- rendering ----------------------------------------------------------------------------------

@pytest.fixture
def gl():
    try:
        model, data = build_model()
        CameraPerception(model).close()
    except Exception as exc:  # no OpenGL context available
        if os.environ.get("REQUIRE_GL") == "1":
            raise
        pytest.skip(f"no OpenGL context: {exc}")


@pytest.mark.parametrize("x,y,yaw_deg", [(0.45, 0.20, 30), (0.30, -0.05, 0), (0.65, 0.45, -40), (0.20, 0.30, 15)])
def test_camera_estimate_rendered(gl, x, y, yaw_deg):
    model, data = build_model()
    place_cube(model, data, (x, y), np.radians(yaw_deg))
    cam = CameraPerception(model)
    try:
        pos, yaw = cam(data)
    finally:
        cam.close()
    assert np.linalg.norm(pos[:2] - [x, y]) < 0.004
    assert abs(np.degrees(wrap_cube_angle(yaw - np.radians(yaw_deg)))) < 2.0
    assert cam.glitches == 0


def test_cube_out_of_view_is_not_seen(gl):
    model, data = build_model()
    place_cube(model, data, (-1.5, 0.0), 0.0)  # behind the robot, outside the camera's view
    cam = CameraPerception(model)
    try:
        assert cam(data) is None
    finally:
        cam.close()


def test_bad_frame_is_rejected(gl, monkeypatch):
    """A corrupted frame (implausible or inconsistent estimate) is thrown away and counted."""
    model, data = build_model()
    place_cube(model, data, (0.45, 0.20), 0.0)
    cam = CameraPerception(model)
    real_look = cam.look
    looks = iter([dict(pos=np.array([1.2, -0.5, 0.06]), yaw=0.3, half=0.03)])  # a wrong frame first
    monkeypatch.setattr(cam, "look", lambda d: next(looks, None) or real_look(d))
    try:
        pos, _ = cam(data)
    finally:
        cam.close()
    assert np.linalg.norm(pos[:2] - [0.45, 0.20]) < 0.004
    assert cam.glitches == 1


def test_pick_and_place_with_camera(gl):
    """End to end: the planner works from the camera's estimate instead of the simulator's truth."""
    from test_system import run
    row = run(5, "--perception", "camera")
    assert row["outcome"] == "OK"
    assert row["perc_pos_err_mm"] < 4
    assert row["perc_glitches"] == 0


# --- camera calibration error (calib_sweep.py) ----------------------------------------------------

def rotation_angle_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def bumped(rot=0.0, shift=0.0, focal=0.0, trial_index=0, calibration="stale"):
    """Set up a camera trial with a bumped camera; returns (model, data, kwargs, believed camera)."""
    import eval_place

    args = scenario_args("--perception", "camera", "--calibration", calibration, "--camera-rot", str(rot),
                         "--camera-shift", str(shift), "--focal-err", str(focal))
    return eval_place.setup_trial(eval_place.sample_trial(trial_index, args), args)


@pytest.mark.parametrize("rot,shift", [(1.0, 0.0), (0.0, 10.0), (2.0, 5.0)])
def test_bump_moves_the_real_camera_by_the_requested_amount(rot, shift):
    model, data, kwargs = bumped(rot, shift)
    try:
        nominal = PinholeCamera(*build_model())
        true = PinholeCamera(model, data)
        assert rotation_angle_deg(true.R, nominal.R) == pytest.approx(rot, abs=1e-4)  # arccos rounding
        assert 1000 * np.linalg.norm(true.pos - nominal.pos) == pytest.approx(shift, abs=1e-6)
    finally:
        kwargs["perceive"].close()


def test_every_method_gets_the_same_bump():
    """Trial i must bump the camera identically for stale and marker, or the comparison is unfair."""
    cams = []
    for calibration in ("stale", "marker"):
        model, data, kwargs = bumped(2.0, 10.0, 2.0, trial_index=4, calibration=calibration)
        cams.append(PinholeCamera(model, data))
        kwargs["perceive"].close()
    assert np.allclose(cams[0].R, cams[1].R) and np.allclose(cams[0].pos, cams[1].pos)


def test_stale_without_bump_is_the_true_camera():
    model, data, kwargs = bumped()
    believed = kwargs["perceive"].camera.believed
    true = PinholeCamera(model, data)
    kwargs["perceive"].close()
    assert np.allclose(believed.R, true.R) and np.allclose(believed.pos, true.pos) and believed.f == true.f


def test_camera_locates_itself_from_the_tags(gl):
    from perception import locate_camera

    model, data = build_model()
    true = PinholeCamera(model, data)
    cam = CameraPerception(model)
    try:
        located, tags = locate_camera(cam.image(data), true.f)
    finally:
        cam.close()
    assert tags == 4
    assert 1000 * np.linalg.norm(located.pos - true.pos) < 3
    assert rotation_angle_deg(located.R, true.R) < 0.15


@pytest.mark.parametrize("calibration,max_err_mm,min_err_mm", [("marker", 3, 0), ("stale", 1000, 15)])
def test_bumped_camera_stale_vs_marker(gl, calibration, max_err_mm, min_err_mm):
    """After a 2 deg knock, the stale belief misses the cube by centimetres; the tags fix it."""
    model, data, kwargs = bumped(rot=2.0, trial_index=1, calibration=calibration)
    try:
        seen = kwargs["perceive"](data)
    finally:
        kwargs["perceive"].close()
    err = 1000 * np.linalg.norm(seen[0][:2] - data.body("cube").xpos[:2])
    assert min_err_mm <= err < max_err_mm
