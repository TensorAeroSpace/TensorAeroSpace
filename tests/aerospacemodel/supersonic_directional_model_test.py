import numpy as np
import pytest

from tensoraerospace.aerospacemodel.supersonic.linear.directional.model import (
    DirectionalSuperSonic,
)


def test_supersonic_directional_model_smoke_and_getters():
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    steps = 6
    model = DirectionalSuperSonic(
        x0=x0, number_time_steps=steps, selected_state_output=None, t0=0, dt=0.01
    )

    # Run a few steps
    for _ in range(3):
        x1 = model.run_step(np.array([1.0, -1.0]))
        assert x1.shape[0] == 5

    # Getters should return arrays of expected lengths
    assert model.get_state("phi").shape[0] == steps - 1
    assert model.get_state("wx").shape == model.get_state("p").shape  # alias
    assert model.get_state("wy").shape == model.get_state("r").shape  # alias
    assert model.get_control("ail").shape[0] == steps - 1
    assert model.get_control("dir").shape[0] == steps - 1  # alias to rud
    # Outputs are sized by time_step-1
    assert model.get_output("beta").shape[0] == model.time_step - 1


def test_supersonic_directional_input_limits_rate_and_magnitude():
    x0 = np.zeros(5, dtype=float)
    steps = 5
    dt = 0.01
    model = DirectionalSuperSonic(x0=x0, number_time_steps=steps, dt=dt)

    # First step: magnitude limits apply
    model.run_step(np.array([1000.0, -1000.0]))
    u0 = model.store_input[:, 0]
    assert abs(u0[0]) <= model.input_magnitude_limits[0] + 1e-12
    assert abs(u0[1]) <= model.input_magnitude_limits[1] + 1e-12

    # Second step: rate limits apply relative to previous
    model.run_step(np.array([-1000.0, 1000.0]))
    u1 = model.store_input[:, 1]
    max_step0 = model.input_rate_limits[0] * dt
    max_step1 = model.input_rate_limits[1] * dt
    assert abs(u1[0] - u0[0]) <= max_step0 + 1e-12
    assert abs(u1[1] - u0[1]) <= max_step1 + 1e-12
