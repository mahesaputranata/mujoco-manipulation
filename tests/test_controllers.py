"""Controller tests. Parametrized over CONTROLLERS, so every new controller is tested automatically."""
import numpy as np
import pytest

from controllers import CONTROLLERS
from demo_reach import Stepper, tcp_error
from scene import HOME_QPOS

import mujoco


def hold_still(model, data, controller, seconds=1.5):
    """Command the home pose and return the TCP error (m) afterwards."""
    step = Stepper(model, data, controller=controller)
    for _ in range(int(seconds / model.opt.timestep)):
        step(HOME_QPOS, 0.0)
    return tcp_error(model, data, HOME_QPOS, mujoco.MjData(model))


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_controller_is_stable(model_data, controller):
    """Every controller must hold the arm without diverging (no NaNs, error within a few cm)."""
    model, data = model_data
    err = hold_still(model, data, controller)
    assert np.all(np.isfinite(data.qpos))
    assert err < 0.02


def test_pd_sags_under_gravity(model_data):
    """Characterization: plain PD needs a steady error to hold the arm up (about 11 mm at home)."""
    model, data = model_data
    assert hold_still(model, data, "pd") > 0.005


def test_gravity_compensation_removes_sag(model_data):
    model, data = model_data
    assert hold_still(model, data, "pd_gravity") < 0.001
