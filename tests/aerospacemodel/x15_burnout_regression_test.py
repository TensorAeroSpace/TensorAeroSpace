"""Fuel exhaustion must cut thrust at the physical burnout time."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.x15.nonlinear import (
    NonlinearX15,
    default_parameters,
)
from tensoraerospace.aerospacemodel.x15.nonlinear.engine import xlr99_thrust
from tensoraerospace.envs.x15_nonlinear import NonlinearX15Env


@pytest.mark.parametrize("integrator", ["rk4", "euler"])
@pytest.mark.parametrize("burn_time", [0.1, 0.5, 1.0])
def test_burnout_impulse_matches_rocket_equation(monkeypatch, integrator, burn_time):
    dynamics = importlib.import_module(
        "tensoraerospace.aerospacemodel.x15.nonlinear.dynamics"
    )
    # No aerodynamic forces/moments: horizontal motion obeys the rocket equation;
    # gravity acts only on the vertical channel with zero attitude/body rates.
    monkeypatch.setattr(
        dynamics,
        "x15_aero",
        lambda *a: SimpleNamespace(L=0.0, D=0.0, Y=0.0, l=0.0, m=0.0, n=0.0),
    )
    p = default_parameters()
    _, mdot = xlr99_thrust(1.0, 1000.0, p)
    fuel = mdot * burn_time
    x = np.zeros(13)
    x[0] = 1000.0
    x[11] = -70000.0
    x[12] = fuel
    model = NonlinearX15(x, dt=1.0, integrator=integrator)
    model.run_step([0, 0, 0, 1])
    expected = p.engine_isp_s * p.g_ft_s2 * np.log1p(fuel / p.empty_weight_lb)
    # Euler retains its first-order quadrature error; it must not burn for
    # an entire step when only a fraction of that step has propellant.
    tol = 2e-5 if integrator == "rk4" else 1.1
    assert model.current_state[0] - x[0] == pytest.approx(expected, abs=tol)
    assert model.propellant_lb == 0.0
    speed = model.current_state[0]
    model.run_step([0, 0, 0, 1])
    assert model.current_state[0] == pytest.approx(speed, abs=1e-10)


@pytest.mark.parametrize("mode", ["virtual", "normalized"])
def test_x15_env_copies_and_bounds_controls(mode):
    x = np.zeros(13)
    x[0] = 2000.0
    x[11] = -70000.0
    x[12] = 100.0
    expected = x.copy()
    env = NonlinearX15Env(initial_state=x, action_space=mode)
    x[:] = 0
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, expected)
    env.step([5, -5, 5, 5])
    np.testing.assert_allclose(
        env.model.u_history[-1].ravel(), np.r_[np.deg2rad([15, -15, 8.5]), 1.0]
    )


@pytest.mark.parametrize(
    "kwargs", [{"damage_profile": object()}, {"damage_event_callback": lambda *a: None}]
)
def test_x15_env_rejects_unimplemented_damage(kwargs):
    x = np.zeros(13)
    x[0] = 1000
    with pytest.raises(NotImplementedError, match="damage"):
        NonlinearX15Env(initial_state=x, **kwargs)


@pytest.mark.parametrize("config", ["BASIC", "A2"])
def test_default_state_uses_configuration_full_load(config):
    from tensoraerospace.aerospacemodel.x15.nonlinear import (
        X15Configuration,
        default_state,
    )

    selected = getattr(X15Configuration, config)
    assert (
        default_state(config=selected)[12]
        == default_parameters(selected).propellant_full_lb
    )


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_x15_rejects_invalid_time_steps(dt):
    x = np.zeros(13)
    with pytest.raises(ValueError, match="dt"):
        NonlinearX15(x, dt=dt)
    with pytest.raises(ValueError, match="dt"):
        NonlinearX15Env(initial_state=x, dt=dt)


def test_nonfinite_action_does_not_change_x15_state():
    x = np.zeros(13)
    x[0] = 1000.0
    env = NonlinearX15Env(initial_state=x)
    env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step([0.0, 0.0, 0.0, np.nan])
    with pytest.raises(ValueError, match="finite"):
        env.model.run_step([0.0, 0.0, 0.0, np.nan])
    np.testing.assert_array_equal(env.model.current_state, x)
    assert len(env.model.x_history) == 1
