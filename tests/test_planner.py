"""Unit tests for grasp selection and planning helpers (demo_reach.py)."""
import numpy as np
import pytest

from demo_reach import ground_truth, grasp_angles, make_plan, wrap_cube_angle

YAWS = np.radians(np.arange(-180, 180, 7.5))


@pytest.mark.parametrize("yaw", YAWS)
def test_wrap_cube_angle_range(yaw):
    w = wrap_cube_angle(yaw)
    assert -np.pi / 4 <= w < np.pi / 4
    assert np.isclose(np.sin(4 * w), np.sin(4 * yaw))  # same angle modulo 90 deg


@pytest.mark.parametrize("yaw", YAWS)
def test_grasp_angles(yaw):
    first, second = grasp_angles(yaw)
    assert abs(first) <= np.pi / 4 + 1e-12  # the smaller wrist turn comes first
    assert np.isclose(abs(first - second), np.pi / 2)  # the other pair of faces
    assert abs(second) <= np.pi / 2 + 1e-12


@pytest.mark.parametrize("yaw", YAWS)
def test_grasp_lines_up_with_cube_faces(yaw):
    """Both grasp options must turn the fingers square to the cube's faces (equal modulo 90 deg)."""
    for psi in grasp_angles(yaw):
        assert np.isclose(wrap_cube_angle(psi - yaw), 0.0, atol=1e-9)


@pytest.mark.parametrize("yaw", YAWS)
def test_grasp_angles_respect_cube_symmetry(yaw):
    """A cube rotated by 90 deg looks the same, so the preferred grasp must be the same."""
    assert np.isclose(grasp_angles(yaw)[0], grasp_angles(yaw + np.pi / 2)[0])


def test_grasp_angles_without_align():
    assert grasp_angles(0.7, align=False) == [0.0]


def test_make_plan_default_pose(model_data):
    model, data = model_data
    pos, yaw = ground_truth(data)
    plan, ok, quat = make_plan(model, data, pos, yaw)
    assert ok
    assert [seg[0] for seg in plan] == ["above cube", "down to cube", "close gripper", "lift", "carry to box",
                                        "lower into box", "release", "retreat"]
    assert np.isclose(np.linalg.norm(quat), 1.0)
