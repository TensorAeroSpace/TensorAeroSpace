"""A steady-flight initial condition must balance all six dynamic equations."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import (
    NonlinearSkywalkerX8,
    default_parameters,
    trim,
    x8_ode_6dof,
)
from tensoraerospace.envs.skywalker_x8_nonlinear import NonlinearSkywalkerX8Env


@pytest.mark.parametrize("speed", [17.1, 18.0, 18.9])
def test_trim_balances_full_rigid_body_and_holds_altitude(speed):
    result = trim(178.0, speed)
    assert result.converged
    control = result.to_control()
    state = result.to_state()
    derivative = x8_ode_6dof(state, control, 0.0, default_parameters())
    np.testing.assert_allclose(derivative[:9], 0.0, atol=1e-7)
    assert abs(derivative[11]) < 1e-10
    model = NonlinearSkywalkerX8(state, dt=0.01)
    for _ in range(500):
        model.run_step(control)
    np.testing.assert_allclose(model.current_state[:9], state[:9], atol=1e-6)
    assert abs(model.altitude_m - 178.0) < 1e-6


@pytest.mark.parametrize("speed", [0.0, -1.0, np.nan, np.inf])
def test_trim_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="V_m_s"):
        trim(178.0, speed)


def test_trim_does_not_accept_infeasible_elevons():
    params = default_parameters()
    params.elevon_max_rad = np.deg2rad(0.1)
    assert not trim(178.0, 18.0, params=params).converged


def test_env_preserves_initial_state_and_limits_individual_elevons():
    state = trim(178.0, 18.0).to_state()
    env = NonlinearSkywalkerX8Env(initial_state=state, action_space="normalized")
    expected = state.copy()
    state[:] = 0.0
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, expected)
    env.step([2.0, 2.0, 2.0])
    de, da, throttle = env.model.u_history[-1].ravel()
    assert max(abs(de + da), abs(de - da)) <= np.deg2rad(20.0) + 1e-12
    assert throttle == 1.0


def test_env_rejects_nonfinite_action_without_advancing():
    env = NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0))
    state, _ = env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step([np.nan, 0.0, 0.5])
    np.testing.assert_array_equal(env.model.current_state, state)


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0}, {"dt": np.nan}, {"number_time_steps": 0}]
)
def test_env_rejects_invalid_time_settings(kwargs):
    with pytest.raises(ValueError):
        NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"damage_profile": object()}, {"damage_event_callback": lambda *a: None}]
)
def test_env_does_not_silently_ignore_damage(kwargs):
    with pytest.raises(NotImplementedError, match="damage"):
        NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0), **kwargs)


def test_x8_newton_euler_energy_balance():
    from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import x8_aero, x8_thrust
    from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear.aero import AeroState

    params = default_parameters()
    inertia = np.array(
        [
            [params.Ix, 0.0, -params.Ixz],
            [0.0, params.Iy, 0.0],
            [-params.Ixz, 0.0, params.Iz],
        ]
    )
    rng = np.random.default_rng(20260916)
    for _ in range(100):
        speed = rng.uniform(17, 19)
        alpha, beta = rng.uniform(0.05, 0.18), rng.uniform(-0.1, 0.1)
        state = np.zeros(12)
        state[:3] = speed * np.array(
            [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)]
        )
        state[3:9] = rng.uniform(-0.2, 0.2, 6)
        state[11] = -rng.uniform(0, 1000)
        control = np.array(
            [rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0.4, 0.7)]
        )
        thrust, ct = x8_thrust(control[2], speed, -state[11], params)
        aero = x8_aero(
            AeroState(alpha, beta, speed, *state[3:6], -state[11], *control[:2], ct),
            params,
        )
        dx = x8_ode_6dof(state, control, 0.0, params)
        moments = np.array([aero.l, aero.m, aero.n])
        np.testing.assert_allclose(
            inertia @ dx[3:6] + np.cross(state[3:6], inertia @ state[3:6]),
            moments,
            atol=1e-12,
        )
        energy_rate = (
            params.mass_kg * state[:3] @ dx[:3]
            + state[3:6] @ inertia @ dx[3:6]
            - params.mass_kg * params.g_m_s2 * dx[11]
        )
        power = (
            thrust * state[0]
            - aero.D * speed * np.cos(beta)
            + aero.Y * state[1]
            + moments @ state[3:6]
        )
        assert energy_rate == pytest.approx(power, abs=1e-10)
