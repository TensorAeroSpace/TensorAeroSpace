"""GeoSat reduced orbital equations, command limits and environment coordinates."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from tensoraerospace.aerospacemodel.geosat import GeoSat
from tensoraerospace.envs.geosat import GeoSatEnv

A = np.array([[0, 1, 0], [0.01036, 0, 0.7753], [0, -0.01774, 0]])
B = np.array([[0], [0], [0.1512]])


def test_reduced_model_matches_equation_61_and_matlab():
    model = GeoSat(np.zeros(3), 2)
    np.testing.assert_array_equal(model.A, A)
    np.testing.assert_array_equal(model.B, B)
    source = Path(inspect.getfile(GeoSat)).parent / "simulinkModel/geosat/geosat_data.m"
    assert "0 -0.01774 0" in source.read_text()


@pytest.mark.parametrize("dt", [0.01, 0.1])
def test_input_limit_applies_from_first_step_and_matches_held_input_oracle(dt):
    model = GeoSat([0.02, -0.004, 0.002], 20, dt=dt)
    augmented = np.zeros((4, 4))
    augmented[:3, :3], augmented[:3, 3:] = A, B
    expected = np.array([0.02, -0.004, 0.002])
    previous = 0.0
    for command in [1.0] * 10 + [-1.0] * 10:
        applied = np.clip(
            command, previous - np.deg2rad(60) * dt, previous + np.deg2rad(60) * dt
        )
        applied = np.clip(applied, -np.deg2rad(25), np.deg2rad(25))
        expected = (expm(augmented * dt) @ np.append(expected, applied))[:3]
        actual = model.run_step([command]).reshape(-1)
        assert model.store_input[0, model.time_step - 1] == pytest.approx(applied)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        previous = applied


def test_linearized_angular_momentum_is_preserved():
    model = GeoSat([0.02, -0.004, 0.002], 100, dt=0.1)
    initial = model.xt.reshape(-1)
    h0 = initial[2] + 0.01774 * initial[0]
    for _ in range(100):
        state = model.run_step([0]).reshape(-1)
        assert state[2] + 0.01774 * state[0] == pytest.approx(h0, abs=1e-13)


def test_env_subset_shape_and_tracking_use_full_physical_state():
    env = GeoSatEnv(
        [0.01, 0.02, 0.03],
        np.zeros((1, 5)),
        5,
        state_space=["rho", "theta", "omega"],
        output_space=["omega", "rho"],
        tracking_states=["theta"],
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs, [0.03, 0.01])
    obs, reward, *_ = env.step([0])
    np.testing.assert_allclose(obs, env.model.xt.reshape(-1)[[2, 0]])
    assert reward == pytest.approx(-abs(env.model.xt.reshape(-1)[1]))


def test_default_outputs_follow_selected_state_space():
    env = GeoSatEnv(
        [0.01, 0.02, 0.03], np.zeros((1, 5)), 5, state_space=["omega", "theta"]
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs, [0.03, 0.02])


def test_time_limit_and_action_space_match_model():
    env = GeoSatEnv(np.zeros(3), np.zeros((1, 3)), 3)
    np.testing.assert_allclose(env.action_space.high, env.model.input_magnitude_limits)
    env.step([0])
    _, _, term, trunc, _ = env.step([0])
    assert not term and trunc


@pytest.mark.parametrize("command", [[np.nan], [np.inf], [0, 1]])
def test_invalid_command_does_not_mutate_env(command):
    env = GeoSatEnv(np.zeros(3), np.zeros((1, 3)), 3)
    with pytest.raises(ValueError):
        env.step(command)
    assert env.current_step == env.model.time_step == 0
    np.testing.assert_array_equal(env.model.xt.reshape(-1), np.zeros(3))


def test_initial_state_is_copied():
    initial = np.zeros(3)
    env = GeoSatEnv(initial, np.zeros((1, 3)), 3)
    initial[:] = 100
    np.testing.assert_array_equal(env.reset()[0], np.zeros(3))


def test_multichannel_reward_uses_each_tracking_error():
    state = np.array([[0.2], [-0.3]])
    ref = np.array([[0.1, 0.1], [0.1, 0.1]])
    assert GeoSatEnv.reward(state, ref, 1) == pytest.approx(-0.25)
