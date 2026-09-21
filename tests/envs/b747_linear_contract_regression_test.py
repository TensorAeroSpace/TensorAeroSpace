"""Keep tracking rewards, observations and the physical model in consistent units."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel import LongitudinalB747
from tensoraerospace.envs.b747 import LinearLongitudinalB747


def make_env(**kwargs):
    args = dict(
        initial_state=[0, 0, 0, 0],
        reference_signal=np.zeros((1, 5)),
        number_time_steps=5,
    )
    args.update(kwargs)
    return LinearLongitudinalB747(**args)


@pytest.mark.parametrize(
    "outputs", [["theta", "q"], ["q", "theta"], ["u"], ["w", "u", "theta"]]
)
def test_reward_tracks_pitch_in_radians_independently_of_observation(outputs):
    target = np.deg2rad(5)
    env = make_env(
        initial_state=[0.2, -0.1, 0.01, target],
        reference_signal=np.full((1, 5), target),
        output_space=outputs,
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    obs, reward, _, _, _ = env.step([0])
    x = env.model.xt.ravel()
    assert reward == pytest.approx(-((x[3] - target) ** 2), abs=1e-14)
    expected = [
        (
            np.rad2deg(x[env.model.list_state.index(name)])
            if name in ("q", "theta")
            else x[env.model.list_state.index(name)]
        )
        for name in outputs
    ]
    np.testing.assert_allclose(obs, expected, rtol=1e-6)
    assert env.observation_space.contains(obs)


def test_multichannel_tracking_uses_requested_order_and_si_units():
    ref = np.array([[0.02], [0.3]])
    env = make_env(
        initial_state=[0.2, 0.1, 0.03, 0.04],
        tracking_states=["q", "u"],
        output_space=["theta"],
        reference_signal=ref,
    )
    _, reward, _, _, _ = env.step([1])
    expected = -np.mean((env.model.xt.ravel()[[2, 0]] - ref[:, 0]) ** 2)
    assert reward == pytest.approx(expected)


def test_reference_holds_last_sample_instead_of_flattening_time_axis():
    state = np.array([0.2])
    assert LinearLongitudinalB747.reward(
        state, np.array([[0.0, 0.1]]), 4
    ) == pytest.approx(-0.01)


def test_time_limit_truncates_without_terminal_state():
    env = make_env(number_time_steps=2)
    _, _, terminated, truncated, _ = env.step([0])
    assert not terminated and truncated and env.done


def test_reset_does_not_alias_initial_state_or_reference():
    initial, reference = np.zeros(4), np.zeros((1, 5))
    env = make_env(initial_state=initial, reference_signal=reference)
    initial[:] = 10
    reference[:] = 20
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, 0)
    assert env.step([0])[1] == 0


@pytest.mark.parametrize("action", [[np.nan], [np.inf], [], [0, 1]])
def test_invalid_command_does_not_advance_environment_or_model(action):
    env = make_env()
    with pytest.raises(ValueError, match="one finite"):
        env.step(action)
    assert env.current_step == env.model.time_step == 0
    np.testing.assert_array_equal(env.model.xt, np.zeros(4))


@pytest.mark.parametrize("reference", [np.empty((1, 0)), [[np.nan]], np.zeros((3, 2))])
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError, match="reference"):
        make_env(reference_signal=reference)


def test_custom_reward_gets_tracked_si_state_and_degree_command():
    calls = []

    def reward(state, reference, step, action=None):
        calls.append((state.copy(), action.copy()))
        return -4.0

    env = make_env(
        initial_state=[0, 0, 0.01, 0.1], output_space=["u"], reward_func=reward
    )
    assert env.step([2])[1] == -4
    np.testing.assert_array_equal(calls[0][0].ravel(), env.model.xt.ravel()[[3]])
    np.testing.assert_array_equal(calls[0][1], [2])


@pytest.mark.parametrize("dt", [0, -0.01, np.nan, np.inf])
def test_model_rejects_invalid_clock(dt):
    with pytest.raises(ValueError, match="dt"):
        LongitudinalB747(np.zeros(4), 5, dt=dt)


@pytest.mark.parametrize("initial", [[0, 0, 0], [0, 0, 0, np.nan]])
def test_model_rejects_invalid_initial_state(initial):
    with pytest.raises(ValueError, match="four finite"):
        LongitudinalB747(np.array(initial), 5)


def test_model_copies_initial_state_and_rejects_commands_atomically():
    initial = np.zeros(4)
    model = LongitudinalB747(initial, 1)
    initial[:] = 1
    np.testing.assert_array_equal(model.xt, 0)
    for command in [[np.nan], [np.inf], [], [0, 1]]:
        with pytest.raises(ValueError, match="one finite"):
            model.run_step(command)
        assert model.time_step == 0
    model.run_step([0])
    before = model.xt.copy()
    with pytest.raises(RuntimeError, match="history"):
        model.run_step([1])
    np.testing.assert_array_equal(model.xt, before)
    assert model.time_step == 1


@pytest.mark.parametrize("steps", [0, -1, 2.5])
def test_model_rejects_invalid_history_capacity(steps):
    with pytest.raises(ValueError, match="number_time_steps"):
        LongitudinalB747(np.zeros(4), steps)
