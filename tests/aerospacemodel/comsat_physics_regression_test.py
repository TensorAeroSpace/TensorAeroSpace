"""ComSat uses normalized perturbations, consistent dynamics, and actual thrust."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from tensoraerospace.aerospacemodel.comsat import ComSat
from tensoraerospace.envs.comsat import ComSatEnv, ImprovedComSatEnv

PAPER_A = np.array([[0, 1, 0], [0.01036, 0, 0.7757], [0, -0.01775, 0]])
PAPER_B = np.array([[0], [0], [0.1513]])


def test_model_matches_published_linearization():
    model = ComSat(np.zeros(3), 2)
    np.testing.assert_array_equal(model.A, PAPER_A)
    np.testing.assert_array_equal(model.B, PAPER_B)


def test_matlab_reference_uses_same_decimal_coefficient():
    path = Path(inspect.getfile(ComSat)).parent / "simulinkModel/comsat/comsat_data.m"
    assert "0 -0.01775 0" in path.read_text()
    assert "0 -0.1775 0" not in path.read_text()


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_first_step_and_reversal_obey_thrust_slew_limit(dt):
    model = ComSat(np.zeros(3), 4, dt=dt)
    previous = 0.0
    for request in [25.0, -25.0, 25.0, -25.0]:
        model.run_step(np.array([request]))
        applied = model.store_input[0, model.time_step - 1]
        assert abs(applied - previous) <= 60 * dt + 1e-13
        previous = applied


def test_zero_order_hold_matches_independent_matrix_exponential():
    model = ComSat([0.01, -0.02, 0.001], 100, dt=0.1)
    augmented = np.zeros((4, 4))
    augmented[:3, :3], augmented[:3, 3:] = PAPER_A, PAPER_B
    state = np.array([0.01, -0.02, 0.001])
    for request in [0.001] * 50 + [-0.002] * 50:
        state = (expm(augmented * 0.1) @ np.append(state, request))[:3]
        np.testing.assert_allclose(
            model.run_step([request]).reshape(-1), state, atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("nominal", [0.0, 6.6108, 6371.0])
def test_nominal_offset_does_not_create_unforced_acceleration(nominal):
    env = ImprovedComSatEnv(
        [nominal, 0, 0], np.zeros((1, 30)), 30, nominal_rho=nominal, dt=0.1
    )
    env.reset()
    for _ in range(20):
        obs, reward, terminated, _, _ = env.step(np.zeros(1))
        np.testing.assert_array_equal(env.state, [nominal, 0, 0])
        np.testing.assert_array_equal(obs, np.zeros(4))
        assert reward == 0.1
        assert not terminated


def test_offset_environments_follow_same_perturbation_trajectory():
    low = ImprovedComSatEnv(
        [0.01, -0.01, 0.001],
        np.zeros((1, 30)),
        30,
        nominal_rho=0,
        use_initial_action_on_first_step=False,
    )
    high = ImprovedComSatEnv(
        [6371.01, -0.01, 0.001],
        np.zeros((1, 30)),
        30,
        nominal_rho=6371,
        use_initial_action_on_first_step=False,
    )
    for _ in range(20):
        a, ra, *_ = low.step(np.array([0.001]))
        b, rb, *_ = high.step(np.array([0.001]))
        np.testing.assert_allclose(a, b, atol=1e-10, rtol=0)
        assert ra == pytest.approx(rb, abs=1e-10)


def test_reward_and_observation_use_slew_limited_thrust():
    env = ImprovedComSatEnv(
        [0, 0, 0],
        np.zeros((1, 5)),
        5,
        nominal_rho=0,
        use_initial_action_on_first_step=False,
        dt=0.01,
    )
    obs, reward, *_ = env.step(np.ones(1))
    applied = env.model.store_input[0, 0] / env.max_thrust
    assert env.previous_action == pytest.approx(applied)
    assert obs[-1] == pytest.approx(applied)
    state = env.state
    expected = 0.1 - env.reward_scale * (
        env.w_theta_dot * (state[2] / env.max_angular_velocity) ** 2
        + env.w_rho * (state[0] / env.max_radial_position_deviation) ** 2
        + env.w_rho_dot * (state[1] / env.max_radial_velocity) ** 2
        + (env.w_action + env.w_smooth + env.w_jerk) * applied**2
    )
    assert reward == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["model", "env"])
@pytest.mark.parametrize("command", [[np.nan], [np.inf], [0, 1]])
def test_invalid_command_leaves_state_and_time_unchanged(kind, command):
    obj = (
        ComSat(np.zeros(3), 5)
        if kind == "model"
        else ImprovedComSatEnv([0, 0, 0], np.zeros((1, 5)), 5, nominal_rho=0)
    )
    model = obj if kind == "model" else obj.model
    before = np.array(model.xt, copy=True)
    with pytest.raises(ValueError):
        (obj.run_step if kind == "model" else obj.step)(command)
    np.testing.assert_array_equal(model.xt, before)
    assert model.time_step == 0
    if kind == "env":
        assert obj.current_step == 0


def test_legacy_time_limit_bootstraps_and_action_box_matches_thrust():
    env = ComSatEnv(np.zeros(3), np.zeros((1, 3)), 3)
    assert env.action_space.high[0] == env.model.input_magnitude_limits[0]
    env.reset()
    env.step(np.zeros(1))
    _, _, terminated, truncated, _ = env.step(np.zeros(1))
    assert not terminated and truncated


def test_initial_state_is_not_aliased_from_caller():
    initial = np.zeros(3)
    model = ComSat(initial, 2)
    initial[:] = 100
    np.testing.assert_array_equal(model.run_step(np.zeros(1)), np.zeros((3, 1)))


@pytest.mark.parametrize(
    "configuration",
    [
        {"initial_state": [0]},
        {"initial_state": [0, np.nan, 0]},
        {"nominal_rho": np.inf},
        {"reference_signal": np.zeros(5)},
        {"reference_signal": np.empty((1, 0))},
        {"reference_signal": np.array([[np.nan]])},
        {"dt": 0},
        {"dt": np.inf},
    ],
)
def test_invalid_configuration_is_rejected(configuration):
    kwargs = {
        "initial_state": np.zeros(3),
        "reference_signal": np.zeros((1, 5)),
        "number_time_steps": 5,
        "nominal_rho": 0,
        **configuration,
    }
    with pytest.raises(ValueError):
        ImprovedComSatEnv(**kwargs)
