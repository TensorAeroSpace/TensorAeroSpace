"""Physical event timing, continuous thrust decay and ownership regressions."""

from unittest.mock import Mock

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.quadrotor.damage import (
    DamageProfile,
    MotorEfficiencyDecay,
    RotorDamageEvent,
    RotorDamageManager,
    RotorDamageState,
    RotorLossEvent,
)
from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv


def make_env(events, dt=0.02, **kwargs):
    env = NonlinearQuadrotorEnv(
        np.zeros(12), 100, dt=dt, damage_profile=DamageProfile(events), **kwargs
    )
    env.reset()
    env.model.param.kdz = 0.0
    return env


@pytest.mark.parametrize("time", [0.0, 0.015, 0.02, 0.03])
@pytest.mark.parametrize("mode", ["virtual", "rotor"])
def test_free_fall_starts_at_actual_rotor_loss(time, mode):
    env = make_env([RotorLossEvent(time, i) for i in range(4)], action_space=mode)
    action = np.array([env.model.hover_thrust, 0, 0, 0])
    if mode == "rotor":
        action = env.allocator.unmix(action)
    for index in range(2):
        obs, *_ = env.step(action)
        elapsed = max(0.0, (index + 1) * env.dt - time)
        assert obs[5] == pytest.approx(9.81 * elapsed, abs=1e-12)
        assert obs[2] == pytest.approx(0.5 * 9.81 * elapsed**2, abs=1e-12)
    assert len(env.model.x_history) == 3
    assert env.model.time_step == 3
    assert all(r["time"] == time for r in env.damage_events_log)


def test_decay_only_applies_after_trigger_and_events_are_sorted():
    manager = RotorDamageManager(
        DamageProfile(
            [RotorDamageEvent(0.75, 0, mu=0.5), MotorEfficiencyDecay(0.5, 0, tau=0.2)]
        )
    )
    manager.inject_event(RotorDamageEvent(0.25, 0, mu=0.8))
    fired = manager.update(1.0, 0.0, 1.0)
    assert [ev.trigger_time for ev in fired] == [0.25, 0.5, 0.75]
    assert manager.state.mu[0] == pytest.approx(0.5 * np.exp(-0.25 / 0.2))
    assert manager.update(1.0, 1.0, 0.0) == []


@pytest.mark.parametrize("trigger", [0.0, 0.015, 0.02])
def test_continuous_decay_matches_analytic_vertical_impulse(trigger):
    tau = 0.1
    floor = 0.2
    duration = 0.2
    env = make_env(
        [MotorEfficiencyDecay(trigger, i, tau=tau, mu_floor=floor) for i in range(4)]
    )
    for _ in range(round(duration / env.dt)):
        env.step([env.model.hover_thrust, 0, 0, 0])
    elapsed = duration - trigger
    expected_v = 9.81 * (1 - floor) * (elapsed - tau * (1 - np.exp(-elapsed / tau)))
    expected_z = (
        9.81
        * (1 - floor)
        * (elapsed**2 / 2 - tau * elapsed + tau**2 * (1 - np.exp(-elapsed / tau)))
    )
    assert env.model.current_state[5] == pytest.approx(expected_v, abs=1e-6)
    assert env.model.current_state[2] == pytest.approx(expected_z, abs=4e-7)
    np.testing.assert_allclose(
        env.damage_manager.state.mu,
        floor + (1 - floor) * np.exp(-elapsed / tau),
        atol=1e-14,
    )


def test_injected_loss_uses_actual_event_time_and_callback_once():
    callback = Mock()
    env = make_env([], damage_event_callback=callback)
    for i in range(4):
        env.damage_manager.inject_event(RotorLossEvent(0.015, i))
    env.step([env.model.hover_thrust, 0, 0, 0])
    assert env.model.current_state[5] == pytest.approx(0.04905, abs=1e-12)
    assert callback.call_count == 4
    env.step([env.model.hover_thrust, 0, 0, 0])
    assert callback.call_count == 4


def test_quadrotor_env_owns_initial_state():
    initial = np.zeros(12)
    initial[2] = -10
    env = NonlinearQuadrotorEnv(initial, 10)
    initial[:] = 99
    obs, _ = env.reset()
    assert obs[2] == -10
    assert obs[0] == 0


def test_bad_action_does_not_advance_damage_or_dynamics():
    env = make_env([RotorLossEvent(0.0, 0)])
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan, 0, 0, 0])
    assert env._step_index == 0
    assert env.damage_events_log == []
    np.testing.assert_array_equal(env.damage_manager.state.mu, np.ones(4))
    assert len(env.model.x_history) == 1


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0.0}, {"dt": np.nan}, {"omega_max": np.inf}, {"omega_min": -1.0}]
)
def test_invalid_env_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NonlinearQuadrotorEnv(np.zeros(12), 10, **kwargs)


def test_damage_state_arrays_are_independent():
    mu = np.ones(4)
    state = RotorDamageState(mu=mu)
    mu[:] = 0.0
    np.testing.assert_array_equal(state.mu, np.ones(4))


@pytest.mark.parametrize(
    "event",
    [
        lambda: RotorLossEvent(np.nan, 0),
        lambda: MotorEfficiencyDecay(0.0, 0, tau=np.inf),
    ],
)
def test_invalid_event_values_are_rejected(event):
    with pytest.raises(ValueError):
        event()
