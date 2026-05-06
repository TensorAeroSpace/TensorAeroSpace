"""Tests for the nonlinear Boeing 737 model + Gymnasium env."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  (registers gym envs)
from tensoraerospace.aerospacemodel.b737.nonlinear import (
    B737Configuration,
    B737Engine,
    NonlinearB737,
    b737_aero,
    b737_ode_6dof,
    b737_thrust,
    b737_thrust_with_asymmetry,
    default_parameters,
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
    set_initial_state,
    trim,
)
from tensoraerospace.aerospacemodel.b737.nonlinear.aero import AeroState
from tensoraerospace.envs.b737_nonlinear import NonlinearB737Env

# ---- Geometry & atmosphere ------------------------------------------------


def test_b737_100_geometry_matches_jsbsim():
    p = default_parameters(B737Configuration.B737_100)
    assert p.S_ft2 == pytest.approx(1171.0)
    assert p.b_ft == pytest.approx(94.7)
    assert p.cbar_ft == pytest.approx(12.31)
    # JSBSim mass / inertia values
    assert p.weight_lb == pytest.approx(100_000.0)
    assert p.Ix == pytest.approx(562_000.0)
    assert p.Iy == pytest.approx(1_473_000.0)
    assert p.Iz == pytest.approx(1_894_000.0)


def test_b737_800_is_larger_and_heavier():
    p100 = default_parameters(B737Configuration.B737_100)
    p800 = default_parameters(B737Configuration.B737_800)
    assert p800.S_ft2 > p100.S_ft2
    assert p800.b_ft > p100.b_ft
    assert p800.weight_lb > p100.weight_lb
    assert p800.engine_thrust_max_lb > p100.engine_thrust_max_lb


def test_isa_density_at_sea_level_matches_standard():
    rho = isa_density_slug_ft3(0.0)
    assert rho == pytest.approx(0.002378, rel=1e-3)


# ---- Engine model --------------------------------------------------------


def test_engine_sls_thrust_equals_published_value():
    p = default_parameters(B737Configuration.B737_100)
    T = b737_thrust(throttle=1.0, mach=0.0, altitude_ft=0.0, params=p)
    # 2 × JT8D-9 = 2 × 14 500 lb = 29 000 lb (FAA TCDS A16WE)
    # 5 % idle margin folds into the 95 % effective PLA
    assert 27_500.0 <= T <= 29_500.0


def test_engine_thrust_drops_with_altitude():
    p = default_parameters()
    T_sl = b737_thrust(throttle=1.0, mach=0.0, altitude_ft=0.0, params=p)
    T_alt = b737_thrust(throttle=1.0, mach=0.0, altitude_ft=30_000.0, params=p)
    assert T_alt < T_sl


def test_engine_thrust_drops_with_mach_via_ram_recovery():
    p = default_parameters()
    T_static = b737_thrust(throttle=1.0, mach=0.0, altitude_ft=0.0, params=p)
    T_mach = b737_thrust(throttle=1.0, mach=0.5, altitude_ft=0.0, params=p)
    assert T_mach < T_static


def test_engine_idle_floor_below_zero_throttle():
    """At throttle=0 the idle fraction prevents zero thrust at sea level."""
    p = default_parameters()
    T = b737_thrust(throttle=0.0, mach=0.0, altitude_ft=0.0, params=p)
    assert T > 0.0
    assert T < p.engine_thrust_max_lb * 0.10


# ---- Asymmetric-thrust yaw moment ----------------------------------------


def test_asymmetric_thrust_zero_when_engines_healthy():
    p = default_parameters()
    T, N = b737_thrust_with_asymmetry(
        throttle=0.7, mach=0.5, altitude_ft=20_000.0, params=p
    )
    T_scalar = b737_thrust(throttle=0.7, mach=0.5, altitude_ft=20_000.0, params=p)
    assert T == pytest.approx(T_scalar, rel=1e-9)
    assert N == 0.0


def test_left_engine_failure_yaws_nose_left():
    """Left engine (#1) dead → surviving thrust is on the right →
    NED body axis: thrust at +y produces moment toward -z (nose left)."""
    from tensoraerospace.aerospacemodel.b737.nonlinear.engine import (
        ENGINE_Y_POSITIONS_FT,
    )

    p = default_parameters()

    # Mock damage state: left engine dead
    class _DamageState:
        engines_mu = {1: 0.0, 2: 1.0}

    p.damage_state = _DamageState()
    T, N = b737_thrust_with_asymmetry(throttle=1.0, mach=0.0, altitude_ft=0.0, params=p)
    # Total thrust ≈ half (one engine alive)
    T_full = b737_thrust(
        throttle=1.0, mach=0.0, altitude_ft=0.0, params=default_parameters()
    )
    assert 0.45 * T_full < T < 0.55 * T_full
    # N < 0 ⇒ nose left (toward dead engine)
    assert N < 0.0
    expected_arm = abs(ENGINE_Y_POSITIONS_FT[1])
    assert abs(N) > 0.5 * expected_arm * (T_full / 2.0)


# ---- Aerodynamics -------------------------------------------------------


def test_aero_lift_increases_with_alpha_in_pre_stall():
    p = default_parameters()
    common = dict(
        beta=0.0,
        V=820.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_ft=25_000.0,
        de=0.0,
        da=0.0,
        dr=0.0,
    )
    L0 = b737_aero(AeroState(alpha=math.radians(0.0), **common), p).L
    L5 = b737_aero(AeroState(alpha=math.radians(5.0), **common), p).L
    assert L5 > L0
    # Lift slope ~ 5/rad → L5 - L0 ≈ 5 × π/36 × q × S
    rho = isa_density_slug_ft3(25_000.0)
    qS = 0.5 * rho * 820.0**2 * p.S_ft2
    expected_dL = 5.5 * math.radians(5.0) * qS
    assert 0.5 * expected_dL < (L5 - L0) < 1.5 * expected_dL


def test_aero_pitching_moment_is_statically_stable():
    p = default_parameters()
    common = dict(
        beta=0.0,
        V=820.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_ft=25_000.0,
        de=0.0,
        da=0.0,
        dr=0.0,
    )
    m_low = b737_aero(AeroState(alpha=math.radians(2.0), **common), p).m
    m_high = b737_aero(AeroState(alpha=math.radians(6.0), **common), p).m
    assert m_high < m_low


def test_aero_elevator_creates_pitch_down_moment_when_de_positive():
    """Elevator deflection conventionally produces pitch-down at +δe."""
    p = default_parameters()
    common = dict(
        alpha=math.radians(2.0),
        beta=0.0,
        V=820.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_ft=25_000.0,
        da=0.0,
        dr=0.0,
    )
    m_neutral = b737_aero(AeroState(de=0.0, **common), p).m
    m_pos = b737_aero(AeroState(de=math.radians(5.0), **common), p).m
    assert m_pos < m_neutral


# ---- ODE smoke ----------------------------------------------------------


def test_ode_returns_12_components():
    p = default_parameters()
    x = set_initial_state(altitude_ft=25_000.0, V_ft_s=738.0, alpha_deg=2.0)
    u = np.array([0.0, 0.0, 0.0, 0.5])
    f = b737_ode_6dof(x, u, 0.0, p)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_throttle_increase_accelerates_along_x():
    p = default_parameters()
    x = set_initial_state(altitude_ft=25_000.0, V_ft_s=738.0, alpha_deg=2.0)
    u_lo = np.array([0.0, 0.0, 0.0, 0.3])
    u_hi = np.array([0.0, 0.0, 0.0, 0.9])
    f_lo = b737_ode_6dof(x, u_lo, 0.0, p)
    f_hi = b737_ode_6dof(x, u_hi, 0.0, p)
    assert f_hi[0] > f_lo[0]


# ---- Trim ---------------------------------------------------------------


def test_trim_b737_100_at_typical_cruise_converges():
    """737-100 cruise ~ FL250, M=0.74."""
    r = trim(altitude_ft=25_000.0, V_ft_s=738.0, config=B737Configuration.B737_100)
    assert r.converged
    assert r.residual < 1e-6
    assert 0.0 < r.throttle <= 1.0
    assert 0.0 <= math.degrees(r.alpha_rad) <= 6.0


def test_trim_b737_800_at_typical_cruise_converges():
    """737-NG cruise ~ FL350, M=0.83."""
    r = trim(altitude_ft=35_000.0, V_ft_s=820.0, config=B737Configuration.B737_800)
    assert r.converged
    assert r.residual < 1e-6
    assert 0.0 < r.throttle <= 1.0


def test_trim_holds_steady_state_for_5_seconds():
    r = trim(altitude_ft=25_000.0, V_ft_s=738.0)
    m = NonlinearB737(x0=r.to_state(), dt=0.05, integrator="rk4")
    u_trim = np.array([float(r.elevator_rad), 0.0, 0.0, float(r.throttle)])
    for _ in range(100):
        m.run_step(u_trim)
    # Should hold to mm-level precision
    assert abs(m.airspeed_ft_s - r.V_ft_s) < 0.5
    assert abs(m.altitude_ft - r.altitude_ft) < 5.0


# ---- Env API ------------------------------------------------------------


def test_env_make_via_gym_registry_works():
    env = gym.make(
        "NonlinearB737-v0", trim_at=(25_000.0, 738.0), number_time_steps=10
    ).unwrapped
    obs, _ = env.reset()
    assert obs.shape == (12,)


def test_env_step_without_reset_raises():
    env = NonlinearB737Env(trim_at=(25_000.0, 738.0), number_time_steps=10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_env_action_size_validated():
    env = NonlinearB737Env(trim_at=(25_000.0, 738.0), number_time_steps=10)
    env.reset()
    with pytest.raises(ValueError, match="4 elements"):
        env.step(np.zeros(3))


def test_env_normalized_action_space_is_pm_one():
    env = NonlinearB737Env(
        trim_at=(25_000.0, 738.0), number_time_steps=10, action_space="normalized"
    )
    np.testing.assert_allclose(env.action_space.high, np.ones(4))
    np.testing.assert_allclose(env.action_space.low, -np.ones(4))


def test_env_truncates_at_step_count():
    env = NonlinearB737Env(trim_at=(25_000.0, 738.0), number_time_steps=5)
    env.reset()
    for _ in range(5):
        _, _, _, trunc, _ = env.step(np.array([0.0, 0.0, 0.0, 0.5]))
    assert trunc


def test_env_requires_one_initialiser():
    with pytest.raises(ValueError, match="must supply one of"):
        NonlinearB737Env(number_time_steps=10)


def test_env_rejects_multiple_initialisers():
    with pytest.raises(ValueError, match="exactly one of"):
        NonlinearB737Env(
            trim_at=(25_000.0, 738.0),
            initial_state=np.zeros(12),
            number_time_steps=10,
        )


def test_env_with_initial_state_holds_trim_briefly():
    """End-to-end: trim + env construction + 1-second hold."""
    r = trim(altitude_ft=25_000.0, V_ft_s=738.0)
    env = NonlinearB737Env(initial_state=r.to_state(), number_time_steps=200)
    obs, _ = env.reset()
    u_trim_rad = np.array([float(r.elevator_rad), 0.0, 0.0, float(r.throttle)])
    for _ in range(20):  # 0.2 s
        obs, _, _, _, _ = env.step(u_trim_rad)
    # Should be very close to the trimmed state still
    V = float(np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2))
    assert abs(V - r.V_ft_s) < 1.0
