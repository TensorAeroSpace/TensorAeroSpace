"""Validate retained F-16 actuators against continuous-time dynamics."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tensoraerospace.aerospacemodel.f16.linear.angular.model import AngularF16


@pytest.mark.parametrize("channel", range(3))
@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_retained_actuators_follow_their_first_order_lag(channel, dt):
    initial = np.zeros(11)
    initial[8 + channel] = 1.5
    model = AngularF16(initial, 4, dt=dt)
    command = np.zeros((3, 1))
    command[channel] = -0.5
    for step in range(1, 5):
        actual = model.run_step(command).reshape(-1)
        expected = -0.5 + 2.0 * np.exp(-20.2 * step * dt)
        assert actual[8 + channel] == pytest.approx(expected, abs=1e-12)
        np.testing.assert_array_equal(np.delete(actual[8:], channel), np.zeros(2))


def test_discrete_airframe_matches_continuous_model_with_retained_actuators():
    model = AngularF16(np.zeros(11), 5, dt=0.02)
    state_rows = model.create_dictionary("states")
    input_rows = model.create_dictionary("input")
    rows = [state_rows[name] for name in model.selected_states]
    columns = [input_rows[name] for name in model.selected_input]
    a = model.A[np.ix_(rows, rows)]
    b = model.B[np.ix_(rows, columns)]
    u = np.array([1.0, -0.5, 0.25])
    times = np.arange(1, 6) * model.dt
    reference = solve_ivp(
        lambda t, x: a @ x + b @ u,
        (0, times[-1]),
        np.zeros(11),
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
    )
    assert reference.success
    np.testing.assert_array_equal(model.filt_D, np.zeros((11, 3)))
    for expected in reference.y.T:
        np.testing.assert_allclose(
            model.run_step(u).reshape(-1), expected, rtol=1e-8, atol=1e-11
        )


def test_actuator_states_are_selectable_and_load_factors_are_not_states():
    model = AngularF16(np.zeros(11), 2, selected_state_output=["rud", "ele", "phi"])
    actual = model.run_step(np.array([1.0, 0.0, 0.5]))
    np.testing.assert_array_equal(actual, model.xt[[10, 8, 0]])
    with pytest.raises(ValueError, match="nz"):
        AngularF16(np.zeros(11), 2, selected_state_output=["nz"])


def test_control_radian_conversion_uses_commands_not_aircraft_angles():
    model = AngularF16(np.full(11, 0.02), 2)
    command = np.array([2.0, -1.0, 0.5])
    model.run_step(command)
    for i, name in enumerate(model.selected_input):
        assert model.get_control(name, to_rad=True)[0] == pytest.approx(
            np.deg2rad(command[i])
        )


def test_output_equation_includes_direct_feedthrough():
    model = AngularF16(np.zeros(11), 2)
    model.filt_D = model.filt_D.astype(float)
    model.filt_D[0, 0] = 0.25
    model.run_step(np.array([2.0, 0.0, 0.0]))
    assert model.store_outputs[0, 0] == pytest.approx(0.5)
