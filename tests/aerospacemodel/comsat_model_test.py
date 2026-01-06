import matplotlib
import numpy as np
import pytest

from tensoraerospace.aerospacemodel.comsat import ComSat


matplotlib.use("Agg")


def _make_env():
    return ComSat(x0=np.zeros(3), number_time_steps=3, dt=0.1)


def test_comsat_run_step_clips_and_updates():
    env = _make_env()
    # Very large command should be clipped by magnitude/rate limits
    out = env.run_step(np.array([1000.0]))
    assert out.shape == (3, 1)
    assert env.time_step == 1
    assert np.all(np.abs(env.store_input[:, 0]) <= env.input_magnitude_limits[0] + 1e-6)


def test_get_state_and_control_and_errors():
    env = _make_env()
    env.run_step(np.array([0.0]))
    env.run_step(np.array([0.0]))

    rho_hist = env.get_state("rho")
    assert rho_hist.shape[0] == env.number_time_steps - 1

    u_hist = env.get_control("u2")
    assert u_hist.shape[0] == env.number_time_steps - 1

    with pytest.raises(Exception):
        env.get_state("bad")
    with pytest.raises(Exception):
        env.get_control("bad")


def test_get_output_smoke():
    env = _make_env()
    env.run_step(np.array([0.0]))
    env.run_step(np.array([0.0]))
    out = env.get_output("rho_dot")
    assert out.shape[0] == env.time_step - 1
    # plot_output skipped: ComSat states not in state_to_latex_* dicts
import numpy as np

from tensoraerospace.aerospacemodel.comsat import ComSat


def test_comsat_initialization_and_run_step():
    # Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
    x0 = np.array([6371.0, 0.0, 0.001])
    steps = 4
    model = ComSat(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    # Apply tangential thrust u2
    x1 = model.run_step(np.array([100.0]))
    assert x1.shape[0] == 3

    assert model.get_state("rho").shape[0] == steps - 1
    assert model.get_state("rho_dot").shape[0] == steps - 1
    assert model.get_output("theta_dot").shape[0] == model.time_step - 1
