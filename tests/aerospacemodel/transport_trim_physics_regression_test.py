"""Check transport trim feasibility and thrust continuity at layer boundaries."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("aircraft,speed", [("b747", 700.0), ("b737", 738.0)])
def test_trim_recovers_from_initial_throttle_above_limit(aircraft, speed):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear"
    )
    altitude = 30000.0 if aircraft == "b747" else 25000.0
    result = module.trim(altitude, speed, initial_guess=(0.05, 0.0, 1.2))
    assert result.converged
    assert 0.0 <= result.throttle <= 1.0
    assert abs(result.elevator_rad) <= module.default_parameters().elevator_max_rad


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
def test_trim_does_not_accept_unbalanced_engine_out(aircraft):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear"
    )
    params = module.default_parameters()
    params.damage_state = SimpleNamespace(engines_mu={1: 0.0}, flap_jam_config=None)
    result = module.trim(0.0, 500.0, params=params)
    dx = getattr(
        importlib.import_module(
            f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear.dynamics"
        ),
        f"{aircraft}_ode_6dof",
    )(
        result.to_state(),
        np.array([result.elevator_rad, 0, 0, result.throttle]),
        0,
        params,
    )
    assert abs(dx[5]) > 1e-4  # This longitudinal solver cannot cancel yaw.
    assert not result.converged
    assert result.residual == pytest.approx(np.linalg.norm(dx[:6]))


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
def test_trim_rejects_elevator_outside_physical_authority(aircraft):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear"
    )
    params = module.default_parameters()
    params.elevator_max_rad = 1e-6
    result = module.trim(0.0, 500.0, params=params)
    assert not result.converged
    assert abs(result.elevator_rad) <= params.elevator_max_rad


@pytest.mark.parametrize(
    "aircraft,engine_name", [("b747", "JT9DEngine"), ("b737", "B737Engine")]
)
def test_engine_thrust_is_continuous_and_monotonic_at_tropopause(aircraft, engine_name):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear.engine"
    )
    engine = getattr(module, engine_name)()
    hs = [36089.0 - 1e-3, 36089.0, 36089.0 + 1e-3]
    thrust = [engine.installed_thrust(0.8, h, 0.7) for h in hs]
    assert thrust[0] >= thrust[1] >= thrust[2]
    assert thrust[2] == pytest.approx(thrust[0], rel=1e-6)
    # The exponent above the boundary stays one; only its reference changes.
    t40 = engine.installed_thrust(0.8, 40000.0, 0.7)
    assert t40 / thrust[1] == pytest.approx(
        module.isa_density_slug_ft3(40000.0) / module.isa_density_slug_ft3(36089.0)
    )


@pytest.mark.parametrize("aircraft,speed", [("b747", 700.0), ("b737", 738.0)])
@pytest.mark.parametrize("mode", ["virtual", "normalized"])
def test_env_owns_state_and_applies_declared_action_bounds(aircraft, speed, mode):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear"
    )
    cls = getattr(
        importlib.import_module(f"tensoraerospace.envs.{aircraft}_nonlinear"),
        f"Nonlinear{aircraft.upper()}Env",
    )
    state = module.trim(20000.0, speed).to_state()
    expected = state.copy()
    env = cls(initial_state=state, action_space=mode)
    state[:] = 0
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs, expected)
    env.step([2.0, -2.0, 2.0, 2.0])
    p = module.default_parameters()
    np.testing.assert_allclose(
        env.model.u_history[-1].ravel(),
        [p.elevator_max_rad, -p.aileron_max_rad, p.rudder_max_rad, 1.0],
    )


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
def test_env_rejects_nan_before_advancing(aircraft):
    cls = getattr(
        importlib.import_module(f"tensoraerospace.envs.{aircraft}_nonlinear"),
        f"Nonlinear{aircraft.upper()}Env",
    )
    state = np.zeros(12)
    state[0] = 500.0
    env = cls(initial_state=state)
    env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step([0, np.nan, 0, 0.5])
    np.testing.assert_array_equal(env.model.current_state, state)


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
@pytest.mark.parametrize("speed", [0.0, -1.0, np.nan, np.inf])
def test_trim_rejects_invalid_speed(aircraft, speed):
    module = importlib.import_module(
        f"tensoraerospace.aerospacemodel.{aircraft}.nonlinear"
    )
    with pytest.raises(ValueError, match="V_ft_s"):
        module.trim(20000.0, speed)


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
@pytest.mark.parametrize(
    "kwargs", [{"dt": 0}, {"dt": np.nan}, {"number_time_steps": 0}]
)
def test_env_rejects_invalid_time(aircraft, kwargs):
    cls = getattr(
        importlib.import_module(f"tensoraerospace.envs.{aircraft}_nonlinear"),
        f"Nonlinear{aircraft.upper()}Env",
    )
    with pytest.raises(ValueError):
        cls(initial_state=np.zeros(12), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"damage_profile": object()}, {"damage_event_callback": lambda *args: None}],
)
def test_b737_does_not_ignore_damage(kwargs):
    from tensoraerospace.envs.b737_nonlinear import NonlinearB737Env

    with pytest.raises(NotImplementedError, match="damage"):
        NonlinearB737Env(initial_state=np.zeros(12), **kwargs)


@pytest.mark.parametrize("aircraft", ["b747", "b737"])
def test_transport_newton_euler_and_energy_balance(aircraft):
    from scripts.validate_transport_physics import audit_transport

    result = audit_transport(aircraft)
    assert result["energy_balance_max_error_ft_lbf_s"] < 1e-6
    assert result["moment_balance_max_error_lbf_ft"] < 1e-8
    assert result["integration"]["reference_success"]
    assert result["integration"]["max_altitude_ft"] > 36089
    assert result["integration"]["max_component_error"][-1] < 1e-4
