"""Tests for the nonlinear AAI RQ-7 Shadow tactical UAV model."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  (registers gym envs)
from tensoraerospace.aerospacemodel.aai_shadow.nonlinear import (
    NonlinearAAIShadow,
    AAIShadowParameters,
    default_parameters,
    isa_density_kg_m3,
    isa_speed_of_sound_m_s,
    set_initial_state,
    shadow_aero,
    shadow_ode_6dof,
    shadow_thrust,
    trim,
)
from tensoraerospace.aerospacemodel.aai_shadow.nonlinear.aero import AeroState
from tensoraerospace.envs.aai_shadow_nonlinear import NonlinearAAIShadowEnv


# ---- Geometry & atmosphere -------------------------------------------


def test_shadow_geometry_matches_published_rq7b():
    p = default_parameters()
    assert p.S_m2 == pytest.approx(4.42, rel=1e-3)
    assert p.b_m == pytest.approx(6.22, rel=1e-3)
    assert p.cbar_m == pytest.approx(0.71, rel=1e-2)
    assert p.mass_kg == pytest.approx(170.0)


def test_isa_density_at_sea_level_si():
    assert isa_density_kg_m3(0.0) == pytest.approx(1.225, rel=1e-3)


def test_aspect_ratio_high_for_surveillance_uav():
    """Shadow has AR ≈ 8.75 — typical class-II surveillance UAV."""
    p = default_parameters()
    AR = p.b_m * p.b_m / p.S_m2
    assert 8.0 <= AR <= 10.0


# ---- Aerodynamics ----------------------------------------------------


def test_aero_lift_increases_with_alpha():
    p = default_parameters()
    common = dict(beta=0.0, V=36.0, p=0.0, q=0.0, r=0.0,
                  altitude_m=1000.0, de=0.0, da=0.0, dr=0.0)
    L0 = shadow_aero(AeroState(alpha=math.radians(0.0), **common), p).L
    L4 = shadow_aero(AeroState(alpha=math.radians(4.0), **common), p).L
    assert L4 > L0


def test_aero_pitching_moment_static_stability():
    """Cmα = -1.5/rad → nose-down moment increases with α."""
    p = default_parameters()
    common = dict(beta=0.0, V=36.0, p=0.0, q=0.0, r=0.0,
                  altitude_m=1000.0, de=0.0, da=0.0, dr=0.0)
    m_a0 = shadow_aero(AeroState(alpha=math.radians(0.0), **common), p).m
    m_a5 = shadow_aero(AeroState(alpha=math.radians(5.0), **common), p).m
    assert m_a5 < m_a0


def test_aero_yaw_stiffness_positive():
    """Cnβ = +0.073 — weather-cock stability."""
    p = default_parameters()
    common = dict(alpha=math.radians(3.0), V=36.0,
                  p=0.0, q=0.0, r=0.0, altitude_m=1000.0,
                  de=0.0, da=0.0, dr=0.0)
    n0 = shadow_aero(AeroState(beta=0.0, **common), p).n
    n5 = shadow_aero(AeroState(beta=math.radians(5.0), **common), p).n
    assert n5 > n0


def test_aero_aileron_creates_roll_moment():
    p = default_parameters()
    common = dict(alpha=math.radians(3.0), beta=0.0, V=36.0,
                  p=0.0, q=0.0, r=0.0, altitude_m=1000.0,
                  de=0.0, dr=0.0)
    l_neutral = shadow_aero(AeroState(da=0.0, **common), p).l
    l_right = shadow_aero(AeroState(da=math.radians(5.0), **common), p).l
    assert l_right > l_neutral


def test_aero_rudder_creates_yaw_moment():
    """Cnδr = -0.069 → positive δr → negative yaw (nose-left)."""
    p = default_parameters()
    common = dict(alpha=math.radians(3.0), beta=0.0, V=36.0,
                  p=0.0, q=0.0, r=0.0, altitude_m=1000.0,
                  de=0.0, da=0.0)
    n_neutral = shadow_aero(AeroState(dr=0.0, **common), p).n
    n_right_rud = shadow_aero(AeroState(dr=math.radians(5.0), **common), p).n
    assert n_right_rud < n_neutral


def test_aero_drag_increases_with_lift():
    p = default_parameters()
    common = dict(beta=0.0, V=36.0, p=0.0, q=0.0, r=0.0,
                  altitude_m=1000.0, de=0.0, da=0.0, dr=0.0)
    D0 = shadow_aero(AeroState(alpha=math.radians(0.0), **common), p).D
    D8 = shadow_aero(AeroState(alpha=math.radians(8.0), **common), p).D
    assert D8 > D0


# ---- Engine ----------------------------------------------------------


def test_engine_static_full_throttle_matches_calibration():
    """Static full thrust ~ 380 N (calibrated)."""
    p = default_parameters()
    T, _ = shadow_thrust(throttle=1.0, V_m_s=0.0, altitude_m=0.0, params=p)
    assert 350.0 <= T <= 400.0


def test_engine_cruise_thrust_in_expected_range():
    """At 36 m/s, 70 % throttle: thrust ≈ 70-80 N (matches drag at trim)."""
    p = default_parameters()
    T, _ = shadow_thrust(throttle=0.7, V_m_s=36.0, altitude_m=1000.0, params=p)
    assert 60.0 <= T <= 100.0


def test_engine_thrust_drops_with_airspeed():
    p = default_parameters()
    T0, _ = shadow_thrust(throttle=0.7, V_m_s=0.0, altitude_m=0.0, params=p)
    T1, _ = shadow_thrust(throttle=0.7, V_m_s=30.0, altitude_m=0.0, params=p)
    assert T1 < T0


def test_engine_zero_throttle_gives_zero_thrust():
    p = default_parameters()
    T, _ = shadow_thrust(throttle=0.0, V_m_s=36.0, altitude_m=1000.0, params=p)
    assert T == 0.0


# ---- ODE smoke -------------------------------------------------------


def test_ode_returns_12_components():
    p = default_parameters()
    x = set_initial_state(altitude_m=1000.0, V_m_s=36.0, alpha_deg=3.0)
    u = np.array([0.0, 0.0, 0.0, 0.7])
    f = shadow_ode_6dof(x, u, 0.0, p)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_ode_throttle_increases_x_acceleration():
    p = default_parameters()
    x = set_initial_state(altitude_m=1000.0, V_m_s=36.0, alpha_deg=3.0)
    f_lo = shadow_ode_6dof(x, np.array([0.0, 0.0, 0.0, 0.3]), 0.0, p)
    f_hi = shadow_ode_6dof(x, np.array([0.0, 0.0, 0.0, 1.0]), 0.0, p)
    assert f_hi[0] > f_lo[0]


# ---- Trim ------------------------------------------------------------


def test_trim_at_typical_cruise_converges():
    """RQ-7B typical cruise: 1000 m AGL, 36 m/s (70 kt)."""
    r = trim(altitude_m=1000.0, V_m_s=36.0)
    assert r.converged
    assert r.residual < 1e-6
    assert 0.0 < r.throttle <= 1.0
    # Should be a reasonable cruise α
    assert 0.0 <= math.degrees(r.alpha_rad) <= 8.0


def test_trim_holds_steady_state():
    r = trim(altitude_m=1000.0, V_m_s=36.0)
    m = NonlinearAAIShadow(x0=r.to_state(), dt=0.02, integrator="rk4")
    u_trim = np.array([float(r.elevator_rad), 0.0, 0.0, float(r.throttle)])
    for _ in range(250):  # 5 sec
        m.run_step(u_trim)
    assert abs(m.airspeed_m_s - r.V_m_s) < 0.5
    assert abs(m.altitude_m - r.altitude_m) < 5.0


def test_trim_to_state_round_trip():
    r = trim(altitude_m=1000.0, V_m_s=36.0)
    x = r.to_state()
    assert x.shape == (12,)
    V = float(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))
    assert V == pytest.approx(r.V_m_s, rel=1e-6)


# ---- Env API --------------------------------------------------------


def test_env_make_via_gym_registry_works():
    env = gym.make("NonlinearAAIShadow-v0",
                   trim_at=(1000.0, 36.0), number_time_steps=10).unwrapped
    obs, _ = env.reset()
    assert obs.shape == (12,)


def test_env_step_without_reset_raises():
    env = NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), number_time_steps=10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_env_action_size_validated():
    env = NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), number_time_steps=10)
    env.reset()
    with pytest.raises(ValueError, match="4 elements"):
        env.step(np.zeros(3))


def test_env_normalized_action_space_is_pm_one():
    env = NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), number_time_steps=10,
                                  action_space="normalized")
    np.testing.assert_allclose(env.action_space.high, np.ones(4))
    np.testing.assert_allclose(env.action_space.low, -np.ones(4))


def test_env_truncates_at_step_count():
    env = NonlinearAAIShadowEnv(trim_at=(1000.0, 36.0), number_time_steps=5)
    env.reset()
    for _ in range(5):
        _, _, _, trunc, _ = env.step(np.array([0.0, 0.0, 0.0, 0.7]))
    assert trunc


def test_env_requires_one_initialiser():
    with pytest.raises(ValueError, match="must supply one of"):
        NonlinearAAIShadowEnv(number_time_steps=10)


def test_env_rejects_multiple_initialisers():
    with pytest.raises(ValueError, match="exactly one of"):
        NonlinearAAIShadowEnv(
            trim_at=(1000.0, 36.0),
            initial_state=np.zeros(12),
            number_time_steps=10,
        )


def test_env_holds_trim_briefly():
    r = trim(altitude_m=1000.0, V_m_s=36.0)
    env = NonlinearAAIShadowEnv(initial_state=r.to_state(),
                                  number_time_steps=200, dt=0.02)
    obs, _ = env.reset()
    u_trim = np.array([float(r.elevator_rad), 0.0, 0.0, float(r.throttle)])
    for _ in range(50):  # 1 s
        obs, _, _, _, _ = env.step(u_trim)
    V = float(np.sqrt(obs[0]**2 + obs[1]**2 + obs[2]**2))
    assert abs(V - r.V_m_s) < 1.0
