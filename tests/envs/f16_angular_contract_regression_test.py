"""F-16 altitude observations and thrust actions must match the physical model."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular import AngularF16
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.trim import find_trim
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


@pytest.mark.parametrize("damage", [False, True])
def test_altitude_reset_matches_declared_observation(damage):
    env = NonlinearAngularF16(
        np.zeros(14), 10, track_altitude=True, damage_observable=damage
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    np.testing.assert_array_equal(obs[:16], env.model.current_state)
    assert obs[14] == env.model.param.Oy
    assert obs[15] == env.model.param.V
    assert env.observation_space.contains(env.step(np.zeros(3))[0])


def test_full_trim_state_can_be_used_by_environment():
    trim = find_trim(V_target=150, h_target=2500)
    assert trim.converged
    env = NonlinearAngularF16(trim.x0, 200, track_altitude=True, thrust_mode="control")
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, trim.x0)
    action = np.array([np.rad2deg(trim.stab_rad), 0, 0, trim.T_thrust])
    assert env.action_space.contains(action)
    for _ in range(100):
        obs, *_ = env.step(action)
    np.testing.assert_allclose(obs, trim.x0, rtol=0, atol=1e-6)


@pytest.mark.parametrize("split", [False, True])
def test_thrust_box_uses_newtons_and_model_limit(split):
    env = NonlinearAngularF16(np.zeros(14), 10, thrust_mode="control", split_stab=split)
    env.reset()
    assert env.action_space.low[-1] == 0
    assert env.action_space.high[-1] == default_parameters().T_max_thrust
    action = np.zeros(env.action_space.shape)
    action[-1] = 50000
    assert env.action_space.contains(action)
    env.step(action)
    assert env.model.param.T_active == 50000
    assert env.model.u_history[-1].shape == (action.size, 1)
    assert env.model.u_history[-1][-1, 0] == 50000


@pytest.mark.parametrize("split", [False, True])
def test_model_history_retains_applied_thrust(split):
    model = AngularF16(np.zeros(14), split_stab=split, thrust_mode="control")
    action = np.zeros(model.action_space_length)
    action[-1] = 1e9
    model.run_step(action)
    assert model.u_history[-1].shape == (action.size, 1)
    assert model.u_history[-1][-1, 0] == model.param.T_max_thrust
    assert action[-1] == 1e9


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_leaves_model_unchanged(value):
    env = NonlinearAngularF16(np.zeros(14), 10, thrust_mode="control")
    env.reset()
    before = env.model.current_state
    with pytest.raises(ValueError, match="finite"):
        env.step([0, 0, 0, value])
    np.testing.assert_array_equal(env.model.current_state, before)
    assert len(env.model.x_history) == 1
    assert env._step_index == 0


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("kind", ["env", "model"])
def test_invalid_time_step_is_rejected(dt, kind):
    with pytest.raises(ValueError, match="dt"):
        if kind == "env":
            NonlinearAngularF16(np.zeros(14), 10, dt=dt)
        else:
            AngularF16(np.zeros(14), dt=dt)


@pytest.mark.parametrize("kind", ["env", "model"])
@pytest.mark.parametrize("invalid", ["state", "speed", "thrust_mode"])
def test_invalid_flight_configuration_is_rejected(kind, invalid):
    state = np.zeros(16)
    state[14:] = [3000, 120]
    kwargs = {"track_altitude": True}
    if invalid == "state":
        state[0] = np.nan
    elif invalid == "speed":
        state[15] = 0
    else:
        kwargs["thrust_mode"] = "unknown"
    with pytest.raises(ValueError):
        if kind == "env":
            NonlinearAngularF16(state, 10, **kwargs)
        else:
            AngularF16(state, **kwargs)


def test_model_rejects_nonfinite_control_before_changing_thrust():
    model = AngularF16(np.zeros(14), thrust_mode="control")
    with pytest.raises(ValueError, match="finite"):
        model.run_step([np.nan, 0, 0, 50000])
    assert np.isnan(model.param.T_active)
    assert len(model.x_history) == 1


@pytest.mark.parametrize(
    "kwargs", [{"number_time_steps": 0}, {"airspeed": 0}, {"airspeed": np.nan}]
)
def test_env_rejects_invalid_episode_parameters(kwargs):
    config = {"number_time_steps": 10, **kwargs}
    with pytest.raises(ValueError):
        NonlinearAngularF16(np.zeros(14), **config)


def test_step_before_reset_has_clear_error():
    env = NonlinearAngularF16(np.zeros(14), 10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3))
