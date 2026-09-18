"""Actuator timing, physical state ownership and environment clock regressions."""

import numpy as np
import pytest
from scipy.linalg import expm

from tensoraerospace.aerospacemodel.ultrastick import Ultrastick
from tensoraerospace.envs.ultrastick import (
    ImprovedUltrastickEnv,
    LinearLongitudinalUltrastick,
)


def make_env(**kwargs):
    params = dict(
        initial_state=np.zeros(5),
        reference_signal=np.zeros((1, 20)),
        number_time_steps=20,
        use_initial_action_on_first_step=False,
    )
    params.update(kwargs)
    return ImprovedUltrastickEnv(**params)


def test_initial_step_respects_elevator_speed_from_rest():
    model = Ultrastick(np.zeros(5), 4, dt=0.01)
    model.run_step(np.array([np.deg2rad(30.0), 1.0]))
    assert np.rad2deg(model.store_input[0, 0]) == pytest.approx(3.0)


def test_explicit_initial_actuator_position_survives_reset():
    model = Ultrastick(np.zeros(5), 4, initial_control=[np.deg2rad(10.0), 0.4])
    for _ in range(2):
        model.run_step(np.array([-1.0, 0.4]))
        assert np.rad2deg(model.store_input[0, 0]) == pytest.approx(7.0)
        model.initialise_system(np.zeros(5), 4)


def test_initial_state_does_not_alias_callers_array():
    state = np.zeros(5)
    model = Ultrastick(state, 4)
    state[2] = 0.2
    np.testing.assert_array_equal(model.run_step(np.zeros(2)), np.zeros(5))


@pytest.mark.parametrize("dt", [0.0, -0.01, np.nan, np.inf])
def test_invalid_clock_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        Ultrastick(np.zeros(5), 4, dt=dt)


@pytest.mark.parametrize("action", [[np.nan, 0.0], [0.0], [0.0, 0.0, 0.0]])
def test_invalid_control_does_not_advance_or_poison_the_plant(action):
    model = Ultrastick(np.zeros(5), 4)
    with pytest.raises(ValueError, match="control"):
        model.run_step(np.array(action))
    assert model.time_step == 0
    np.testing.assert_array_equal(model.store_states, np.zeros((5, 5)))


@pytest.mark.parametrize("dt", [0.002, 0.02, 0.05])
def test_environment_and_plant_share_the_same_physical_clock(dt):
    initial = np.array([0.2, -0.1, 0.02, -0.03, 0.1])
    env = make_env(initial_state=initial, dt=dt)
    env.reset()
    env.step([0.0, -1.0])
    expected = expm(env.model.A * dt) @ initial
    np.testing.assert_allclose(env.model.xt.reshape(-1), expected, atol=1e-12)
    assert env.model.dt == env.dt


def test_observation_and_reward_report_the_actually_applied_elevator():
    env = make_env(dt=0.02)
    env.reset()
    env.model.input_rate_limits[0] = np.deg2rad(25.0)
    env.step([0.0, -1.0])
    obs, _, _, _, info = env.step([1.0, -1.0])
    applied = np.rad2deg(env.model.store_input[0, 1])
    assert info["elevator_deg"] == pytest.approx(applied)
    assert obs[3] == pytest.approx(applied / env.max_elevator_deg)
    assert info["du_elev2"] == pytest.approx((applied / env.max_elevator_deg) ** 2)


def test_initial_position_matches_observation_and_first_applied_command():
    env = make_env(
        initial_elevator_deg=40.0,
        initial_throttle=2.0,
        use_initial_action_on_first_step=True,
    )
    obs, _ = env.reset()
    assert obs[3:].tolist() == [1.0, 1.0]
    _, _, _, _, info = env.step([-1.0, -1.0])
    assert info["elevator_deg"] == pytest.approx(env.max_elevator_deg)
    assert np.rad2deg(env.model.store_input[0, 0]) == pytest.approx(
        env.max_elevator_deg
    )
    assert info["throttle"] == pytest.approx(1.0)


def test_legacy_time_limit_preserves_value_bootstrapping():
    env = LinearLongitudinalUltrastick(np.zeros(5), np.zeros((1, 3)), 3)
    env.reset()
    env.step([0.0])
    assert env.step([0.0])[2:4] == (False, True)


def test_legacy_invalid_action_keeps_both_clocks_unchanged():
    env = LinearLongitudinalUltrastick(np.zeros(5), np.zeros((1, 3)), 3)
    env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan])
    assert env.current_step == env.model.time_step == 0
