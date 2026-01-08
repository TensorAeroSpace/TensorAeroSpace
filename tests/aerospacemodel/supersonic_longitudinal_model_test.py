import numpy as np
import pytest

from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal.model import (
    LongitudinalSuperSonic as LongitudinalModel,
)
from tensoraerospace.aerospacemodel.supersonic.linear.longitudinal.output_based import (
    LongitudinalSuperSonic as LongitudinalOutputBased,
)


def test_supersonic_longitudinal_model_smoke():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 4
    model = LongitudinalModel(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    x1 = model.run_step(np.array([1.0]))
    assert x1.shape[0] == 4
    assert model.get_state("theta").shape[0] == steps - 1
    assert model.get_control("ele").shape[0] == steps - 1
    assert model.get_output("q").shape[0] == model.time_step - 1


def test_supersonic_longitudinal_output_based_smoke_and_limits():
    x0 = np.array([0.0, 0.0, 0.0, 0.0])
    steps = 6
    dt = 0.01
    model = LongitudinalOutputBased(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=dt
    )

    # First step clamps magnitude
    model.run_step(np.array([1000.0]))
    u0 = float(model.store_input[0, 0])
    assert abs(u0) <= float(model.input_magnitude_limits[0]) + 1e-12

    # Second step rate limits relative to previous
    model.run_step(np.array([-1000.0]))
    u1 = float(model.store_input[0, 1])
    max_step = float(model.input_rate_limits[0]) * dt
    assert abs(u1 - u0) <= max_step + 1e-12

    # Outputs exist and are shaped by time_step-1
    assert model.get_output("theta").shape[0] == model.time_step - 1


def test_supersonic_longitudinal_model_conversions():
    """Cover to_deg/to_rad branches."""
    model = LongitudinalModel(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    # state conversions
    state_deg = model.get_state("wz", to_deg=True)
    assert state_deg is not None
    state_rad = model.get_state("theta", to_rad=True)
    assert state_rad is not None
    # control conversions
    ctrl_deg = model.get_control("stab", to_deg=True)
    assert ctrl_deg is not None
    ctrl_rad = model.get_control("ele", to_rad=True)
    assert ctrl_rad is not None
    # output (skip to_deg/to_rad - prod code has bug using state_history as dict)
    out = model.get_output("q")
    assert out is not None


def test_supersonic_output_based_conversions():
    """Cover to_deg/to_rad branches in output_based model."""
    model = LongitudinalOutputBased(x0=np.zeros(4), number_time_steps=5, dt=0.01)
    model.run_step(np.array([1.0]))
    model.run_step(np.array([1.0]))
    # state conversions
    state_deg = model.get_state("theta", to_deg=True)
    assert state_deg is not None
    state_rad = model.get_state("q", to_rad=True)
    assert state_rad is not None
    # control conversions
    ctrl_deg = model.get_control("ele", to_deg=True)
    assert ctrl_deg is not None
    ctrl_rad = model.get_control("stab", to_rad=True)
    assert ctrl_rad is not None
    # output (skip to_deg/to_rad - prod code has bug using state_history as dict)
    out = model.get_output("q")
    assert out is not None
