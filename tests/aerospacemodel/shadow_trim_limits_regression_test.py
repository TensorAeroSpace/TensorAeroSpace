"""Check physically feasible Shadow trim and the environment input contract."""

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
    NonlinearAAIShadow,
    default_parameters,
    shadow_ode_6dof,
    trim,
)
from tensoraerospace.envs.aai_shadow_nonlinear import NonlinearAAIShadowEnv


def test_feasible_fast_cruise_does_not_stall_at_throttle_clipping():
    result = trim(1000.0, 37.8)
    assert result.converged
    assert 0.97 < result.throttle < 1.0
    control = np.array([result.elevator_rad, 0.0, 0.0, result.throttle])
    state = result.to_state()
    derivative = shadow_ode_6dof(state, control, 0.0, default_parameters())
    np.testing.assert_allclose(derivative[:9], 0.0, atol=1e-7)
    assert abs(derivative[11]) < 1e-10
    model = NonlinearAAIShadow(state, dt=0.01)
    for _ in range(500):
        model.run_step(control)
    np.testing.assert_allclose(model.current_state[:9], state[:9], atol=1e-6)
    assert abs(model.current_state[11] - state[11]) < 1e-6


def test_trim_rejects_unavailable_elevator_authority():
    params = default_parameters()
    params.elevator_max_rad = np.deg2rad(0.1)
    assert not trim(1000.0, 36.0, params=params).converged


@pytest.mark.parametrize("speed", [0.0, -10.0, np.nan, np.inf])
def test_trim_validates_airspeed(speed):
    with pytest.raises(ValueError, match="V_m_s"):
        trim(1000.0, speed)


@pytest.mark.parametrize("mode", ["virtual", "normalized"])
def test_env_bounds_actions_and_owns_initial_state(mode):
    state = trim(1000.0, 36.0).to_state()
    expected = state.copy()
    env = NonlinearAAIShadowEnv(initial_state=state, action_space=mode)
    state[:] = 0.0
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, expected)
    env.step([10.0, -10.0, 10.0, 10.0])
    applied = env.model.u_history[-1].ravel()
    np.testing.assert_allclose(
        applied, [np.deg2rad(20), -np.deg2rad(20), np.deg2rad(15), 1.0]
    )


def test_env_nonfinite_action_cannot_corrupt_model():
    env = NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0))
    state, _ = env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step([0.0, np.nan, 0.0, 0.5])
    np.testing.assert_array_equal(env.model.current_state, state)


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0}, {"dt": np.nan}, {"number_time_steps": 0}]
)
def test_env_rejects_invalid_time_settings(kwargs):
    with pytest.raises(ValueError):
        NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), **kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"damage_profile": object()}, {"damage_event_callback": lambda *a: None}]
)
def test_env_does_not_silently_ignore_damage(kwargs):
    with pytest.raises(NotImplementedError, match="damage"):
        NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), **kwargs)


def test_shadow_newton_euler_energy_balance():
    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear.aero import (
        AeroState,
        shadow_aero,
    )
    from tensoraerospace.aerospacemodel.aai_shadow.nonlinear.engine import shadow_thrust

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
        speed = rng.uniform(34, 38)
        alpha, beta = rng.uniform(0.02, 0.12), rng.uniform(-0.1, 0.1)
        state = np.zeros(12)
        state[:3] = speed * np.array(
            [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)]
        )
        state[3:9] = rng.uniform(-0.2, 0.2, 6)
        state[11] = -rng.uniform(0, 1000)
        control = np.r_[rng.uniform(-0.05, 0.05, 3), rng.uniform(0.5, 1.0)]
        thrust, _ = shadow_thrust(control[3], speed, -state[11], params)
        aero = shadow_aero(
            AeroState(alpha, beta, speed, *state[3:6], -state[11], *control[:3]), params
        )
        dx = shadow_ode_6dof(state, control, 0.0, params)
        moments = np.array([aero.l, aero.m, aero.n])
        np.testing.assert_allclose(
            inertia @ dx[3:6] + np.cross(state[3:6], inertia @ state[3:6]),
            moments,
            atol=1e-10,
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
        assert energy_rate == pytest.approx(power, abs=1e-8, rel=0)
