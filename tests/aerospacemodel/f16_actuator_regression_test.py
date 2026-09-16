"""Physical regressions for the retained elevator actuator in linear F-16."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tensoraerospace.aerospacemodel.f16.linear.longitudinal.model import LongitudinalF16


@pytest.mark.parametrize("dt", [0.01, 0.1])
@pytest.mark.parametrize("initial_elevator", [0.0, 2.0])
def test_elevator_follows_first_order_actuator_response(dt, initial_elevator):
    model = LongitudinalF16([0.0, 0.0, 0.0, initial_elevator], 5, dt=dt)
    command = 1.0  # degrees, matching the supplied MATLAB actuator matrices
    for step in range(1, 5):
        state = model.run_step(np.array([command])).reshape(-1)
        expected = command + (initial_elevator - command) * np.exp(-20.2 * step * dt)
        assert state[3] == pytest.approx(expected, abs=1e-12)


def test_airframe_response_matches_continuous_system_with_retained_actuator():
    model = LongitudinalF16([0.0, 0.0, 0.0, 0.0], 5, dt=0.02)
    states = model.create_dictionary("states")
    rows = [states[name] for name in ["theta", "alpha", "q", "ele"]]
    elevator_input = model.create_dictionary("input")["ele"]
    a = model.A[np.ix_(rows, rows)]
    b = model.B[rows, elevator_input]
    times = np.arange(1, 6) * model.dt
    reference = solve_ivp(
        lambda t, x: a @ x + b,
        (0, times[-1]),
        np.zeros(4),
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
    )
    assert reference.success
    for expected in reference.y.T:
        actual = model.run_step(np.array([1.0])).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-11)


def test_elevator_can_be_selected_as_a_state():
    model = LongitudinalF16(np.zeros(4), 2, selected_state_output=["ele", "theta"])
    actual = model.run_step(np.array([1.0]))
    assert actual.shape == (2, 1)
    np.testing.assert_allclose(actual, model.xt[[3, 0]])
    assert actual[0, 0] == pytest.approx(1.0 - np.exp(-20.2 * model.dt))


def test_load_factor_is_not_silently_returned_as_elevator_state():
    with pytest.raises(ValueError, match="nz"):
        LongitudinalF16(np.zeros(4), 2, selected_state_output=["nz"])
