import numpy as np
import pytest

from tensoraerospace.aerospacemodel.ultrastick import Ultrastick


def test_ultrastick_initialization_run_and_histories():
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    steps = 5
    dt = 0.01
    model = Ultrastick(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=dt
    )

    x1 = model.run_step(np.array([5.0, 2.0]))
    assert x1.shape[0] == 5

    assert model.get_state("theta").shape[0] == steps
    assert model.get_control("ele").shape[0] == steps
    assert model.get_output("h").shape[0] == model.time_step - 1


def test_ultrastick_multiple_steps():
    """Cover time_step != 0 branch."""
    model = Ultrastick(x0=np.zeros(5), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0, 0.5]))
    x2 = model.run_step(np.array([10.0, 5.0]))
    assert x2.shape[0] == 5
    assert model.time_step == 2


def test_ultrastick_get_state_conversions():
    """Cover to_deg/to_rad branches and aliases."""
    model = Ultrastick(x0=np.zeros(5), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0, 0.5]))
    model.run_step(np.array([1.0, 0.5]))
    state_deg = model.get_state("theta", to_deg=True)
    assert state_deg is not None
    state_rad = model.get_state("q", to_rad=True)
    assert state_rad is not None


def test_ultrastick_get_control_conversions():
    """Cover to_deg/to_rad and alias branches."""
    model = Ultrastick(x0=np.zeros(5), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0, 0.5]))
    model.run_step(np.array([1.0, 0.5]))
    ctrl_deg = model.get_control("ele", to_deg=True)
    assert ctrl_deg is not None
    ctrl_rad = model.get_control("stab", to_rad=True)  # alias for ele
    assert ctrl_rad is not None


def test_ultrastick_get_output_conversions():
    """Cover to_deg/to_rad branches in get_output."""
    model = Ultrastick(x0=np.zeros(5), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0, 0.5]))
    model.run_step(np.array([1.0, 0.5]))
    out_deg = model.get_output("h", to_deg=True)
    assert out_deg is not None
    out_rad = model.get_output("theta", to_rad=True)
    assert out_rad is not None
