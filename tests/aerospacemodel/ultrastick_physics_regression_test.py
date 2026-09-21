"""Ultrastick names and observations must follow the state/output equations."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.ultrastick import Ultrastick
from tensoraerospace.envs.ultrastick import (
    ImprovedUltrastickEnv,
    LinearLongitudinalUltrastick,
)


def test_pitch_state_names_follow_dtheta_dt_equals_q():
    initial = np.array([0.0, 0.0, 0.1, -0.2, 0.0])
    model = Ultrastick(initial, 3)
    assert model.get_state("theta")[0] == 0.1
    assert model.get_state("q")[0] == -0.2
    theta_row = model.selected_states.index("theta")
    assert (model.A @ initial)[theta_row] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "names,indices",
    [
        (["theta", "q"], [2, 3]),
        (["q", "theta"], [3, 2]),
        (["Va", "alpha"], [0, 1]),
        (["u", "w"], [0, 1]),
    ],
)
def test_selected_outputs_follow_their_physical_names(names, indices):
    model = Ultrastick(
        np.array([0.2, 0.1, 0.02, 0.03, 0.1]), 3, selected_state_output=names
    )
    actual = model.run_step(np.array([0.01, 0.1]))
    expected = (model.filt_C @ model.xt).reshape(-1)
    np.testing.assert_allclose(actual, expected[indices])


def test_step_returns_observation_after_the_transition():
    model = Ultrastick(np.array([0.2, 0.1, 0.02, 0.03, 0.1]), 3)
    actual = model.run_step(np.array([0.01, 0.1]))
    np.testing.assert_allclose(actual, (model.filt_C @ model.xt).reshape(-1))
    model.run_step(np.zeros(2))
    assert model.get_output("theta")[0] == pytest.approx(0.02)
    assert model.get_output("q")[0] == pytest.approx(0.03)
    np.testing.assert_array_equal(model.get_output("Va"), model.get_output("u"))
    np.testing.assert_array_equal(model.get_output("alpha"), model.get_output("w"))


def test_legacy_environment_returns_selected_physical_states_after_step():
    initial = np.array([0.2, 0.1, 0.02, 0.03, 0.1])
    names = ["u", "w", "theta", "q", "h"]
    env = LinearLongitudinalUltrastick(initial, np.zeros((1, 5)), 5, state_space=names)
    try:
        observation, _ = env.reset()
        np.testing.assert_allclose(observation, initial)
        observation, *_ = env.step(np.array([1.0]))
        np.testing.assert_allclose(observation, env.model.xt.reshape(-1), rtol=1e-6)
    finally:
        env.close()


def test_improved_environment_has_no_one_step_observation_delay():
    env = ImprovedUltrastickEnv(
        np.array([0.2, 0.1, 0.02, 0.03, 0.1]), np.zeros((1, 5)), 5
    )
    try:
        env.reset()
        observation, *_ = env.step(np.array([0.5, -1.0]))
        state = env.model.xt.reshape(-1)
        assert observation[1] == pytest.approx(
            state[3] / env.max_pitch_rate_rad_s, rel=1e-6
        )
        assert observation[2] == pytest.approx(state[2] / env.max_pitch_rad, rel=1e-6)
    finally:
        env.close()


def test_plotting_airspeed_does_not_modify_recorded_outputs():
    import matplotlib.pyplot as plt

    model = Ultrastick(np.array([0.2, 0.1, 0.02, 0.03, 0.1]), 3)
    for _ in range(2):
        model.run_step(np.zeros(2))
    original = model.store_outputs.copy()
    figure = model.plot_output("u", np.arange(3) * model.dt)
    plt.close(figure)
    np.testing.assert_array_equal(model.store_outputs, original)


# Ahmed et al. (2015), DOI 10.4172/2168-9695.1000126, p. 6.
# The published state coordinates are [u, w, theta, q, -h].
PUBLISHED_A = np.array(
    [
        [-0.5944, 0.8008, -9.791, -0.8747, 5.077e-5],
        [-0.744, -7.56, -0.5294, 15.72, -0.000939],
        [0, 0, 0, 1, 0],
        [1.041, -7.406, 0, -15.81, -7.284e-18],
        [-0.05399, 0.9985, -17, 0, 0],
    ]
)
PUBLISHED_ELEVATOR = np.array([0.4669, -2.703, 0, -133.7, 0])


def test_trajectory_matches_published_model_in_positive_altitude_coordinates():
    from scipy.integrate import solve_ivp

    initial = np.array([0.2, -0.1, 0.01, -0.02, 0.3])
    transform = np.diag([1, 1, 1, 1, -1])
    model = Ultrastick(initial, 20, dt=0.02)
    elevator = 0.01
    times = np.arange(1, 21) * model.dt
    reference = solve_ivp(
        lambda t, x: PUBLISHED_A @ x + PUBLISHED_ELEVATOR * elevator,
        (0, times[-1]),
        transform @ initial,
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
    )
    assert reference.success
    for expected in reference.y.T:
        model.run_step(np.array([elevator, 0.0]))
        np.testing.assert_allclose(
            model.xt.reshape(-1), transform @ expected, rtol=1e-8, atol=1e-11
        )


def test_longitudinal_modes_have_no_spurious_unstable_pole():
    model = Ultrastick(np.zeros(5), 2)
    poles = np.linalg.eigvals(model.A)
    assert np.all(poles.real < 0), poles
    # The source reports the short-period pair as -11.7 +/- 10.0j.
    short_period = poles[np.argmax(poles.imag)]
    assert short_period.real == pytest.approx(-11.7, abs=0.05)
    assert short_period.imag == pytest.approx(10.0, abs=0.05)


def test_pitch_rate_perturbation_has_aerodynamic_damping():
    model = Ultrastick(np.zeros(5), 2)
    initial = np.array([0.0, 0.0, 0.0, 0.1, 0.0])
    derivative = model.A @ initial
    assert derivative[2] == pytest.approx(0.1)
    assert derivative[3] == pytest.approx(-1.581)


def test_altitude_row_matches_linearized_vertical_kinematics():
    model = Ultrastick(np.zeros(5), 2)
    # h_dot = u*sin(theta) - w*cos(theta), trim speed 17 m/s.
    theta_trim = np.arctan2(0.05399, 0.9985)
    expected = np.array([np.sin(theta_trim), -np.cos(theta_trim), 17.0, 0.0])
    np.testing.assert_allclose(model.A[4, :4], expected, rtol=1e-4, atol=1e-12)


def test_airdata_output_matches_derivatives_at_trim():
    model = Ultrastick(np.zeros(5), 2)
    trim = 17.0 * np.array([0.9985, 0.05399])
    epsilon = 1e-4

    def airdata(velocity):
        return np.array(
            [np.linalg.norm(velocity), np.arctan2(velocity[1], velocity[0])]
        )

    jacobian = np.column_stack(
        [
            (airdata(trim + direction) - airdata(trim - direction)) / (2 * epsilon)
            for direction in epsilon * np.eye(2)
        ]
    )
    np.testing.assert_allclose(model.C[:2, :2], jacobian, rtol=3e-4, atol=1e-7)


def test_reduced_model_does_not_invent_a_throttle_to_altitude_channel():
    # This published reduction has a zero throttle column; a propulsion model
    # is required before the second input can affect the physical trajectory.
    model = Ultrastick(np.zeros(5), 3)
    for _ in range(3):
        np.testing.assert_array_equal(model.run_step(np.array([0.0, 0.7])), np.zeros(5))
