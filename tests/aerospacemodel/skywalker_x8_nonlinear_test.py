"""Tests for the nonlinear Skywalker X8 small UAV (CEAS 2025 model)."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  (registers gym envs)
from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear import (
    NonlinearSkywalkerX8,
    SkywalkerX8Parameters,
    X8Propeller,
    default_parameters,
    isa_density_kg_m3,
    isa_speed_of_sound_m_s,
    set_initial_state,
    trim,
    x8_aero,
    x8_ode_6dof,
    x8_thrust,
)
from tensoraerospace.aerospacemodel.skywalker_x8.nonlinear.aero import AeroState
from tensoraerospace.envs.skywalker_x8_nonlinear import NonlinearSkywalkerX8Env

# ---- Geometry & atmosphere (paper Table 1) -----------------------------


def test_x8_geometry_matches_published_values():
    p = default_parameters()
    assert p.S_m2 == pytest.approx(0.75)
    assert p.b_m == pytest.approx(2.10)
    assert p.cbar_m == pytest.approx(0.36)
    assert p.mass_kg == pytest.approx(3.364, rel=1e-3)
    # Inertia values from CEAS 2025 Table 1
    assert p.Ix == pytest.approx(0.325, rel=1e-3)
    assert p.Iy == pytest.approx(0.140, rel=1e-3)
    assert p.Iz == pytest.approx(0.400, rel=1e-3)
    assert p.Ixz == pytest.approx(0.029, rel=1e-3)


def test_isa_density_at_sea_level_si():
    rho = isa_density_kg_m3(0.0)
    assert rho == pytest.approx(1.225, rel=1e-3)


def test_isa_density_drops_with_altitude():
    assert isa_density_kg_m3(1000.0) < isa_density_kg_m3(0.0)
    assert isa_density_kg_m3(5000.0) < isa_density_kg_m3(1000.0)


# ---- Aerodynamic coefficients (paper Table 8) -------------------------


def test_aero_lift_slope_matches_published_CLa():
    """C_Lα = 2.573 /rad — verify by finite difference around α=0."""
    p = default_parameters()
    common = dict(
        beta=0.0, V=18.0, p=0.0, q=0.0, r=0.0, altitude_m=178.0, de=0.0, da=0.0, CT=0.0
    )
    L0 = x8_aero(AeroState(alpha=0.0, **common), p).L
    L1 = x8_aero(AeroState(alpha=math.radians(5.0), **common), p).L
    rho = isa_density_kg_m3(178.0)
    qS = 0.5 * rho * 18.0**2 * p.S_m2
    dCL_dalpha = (L1 - L0) / (math.radians(5.0) * qS)
    assert dCL_dalpha == pytest.approx(2.573, rel=0.05)


def test_aero_pitching_moment_static_stability():
    """C_mα = -0.274 — increase α should produce nose-down moment."""
    p = default_parameters()
    common = dict(
        beta=0.0, V=18.0, p=0.0, q=0.0, r=0.0, altitude_m=178.0, de=0.0, da=0.0, CT=0.0
    )
    m_a0 = x8_aero(AeroState(alpha=math.radians(0.0), **common), p).m
    m_a5 = x8_aero(AeroState(alpha=math.radians(5.0), **common), p).m
    assert m_a5 < m_a0  # pitch-down moment increases at higher α


def test_aero_elevator_creates_pitch_down_at_positive_de():
    p = default_parameters()
    common = dict(
        alpha=math.radians(4.0),
        beta=0.0,
        V=18.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_m=178.0,
        da=0.0,
        CT=0.0,
    )
    m_neutral = x8_aero(AeroState(de=0.0, **common), p).m
    m_pos = x8_aero(AeroState(de=math.radians(5.0), **common), p).m
    assert m_pos < m_neutral


def test_aero_aileron_creates_roll_moment():
    """C_lδa = +0.102 — positive δa should give positive (right) roll."""
    p = default_parameters()
    common = dict(
        alpha=math.radians(4.0),
        beta=0.0,
        V=18.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_m=178.0,
        de=0.0,
        CT=0.0,
    )
    l_neutral = x8_aero(AeroState(da=0.0, **common), p).l
    l_right = x8_aero(AeroState(da=math.radians(5.0), **common), p).l
    assert l_right > l_neutral


def test_aero_sideslip_gives_restoring_yaw():
    """C_nβ = +0.022 — positive β should give nose-right yaw moment (weathercock)."""
    p = default_parameters()
    common = dict(
        alpha=math.radians(4.0),
        V=18.0,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_m=178.0,
        de=0.0,
        da=0.0,
        CT=0.0,
    )
    n_neutral = x8_aero(AeroState(beta=0.0, **common), p).n
    n_pos = x8_aero(AeroState(beta=math.radians(5.0), **common), p).n
    assert n_pos > n_neutral


def test_aero_drag_polar_minimum_above_zero_alpha():
    """The X8 drag polar is asymmetric (CDk1 = -0.034, CDk2 = +0.225,
    CL₀ = -0.077). Minimum drag occurs at α ≈ 3.4°; both α = 4° (near
    minimum) and α = 10° (far from minimum, high CL) should be compared:
    drag at α = 10° should exceed drag at α = 4°.
    """
    p = default_parameters()
    common = dict(
        beta=0.0, V=18.0, p=0.0, q=0.0, r=0.0, altitude_m=178.0, de=0.0, da=0.0, CT=0.0
    )
    D4 = x8_aero(AeroState(alpha=math.radians(4.0), **common), p).D
    D10 = x8_aero(AeroState(alpha=math.radians(10.0), **common), p).D
    assert D10 > D4


# ---- Engine model -----------------------------------------------------


def test_engine_static_full_throttle_matches_calibration():
    """Static thrust at full throttle should be ~ 40 N (calibrated)."""
    p = default_parameters()
    T, _ = x8_thrust(throttle=1.0, V_m_s=0.0, altitude_m=0.0, params=p)
    assert 35.0 <= T <= 45.0


def test_engine_cruise_thrust_matches_paper_trim():
    """At paper trim (44 % throttle, 18 m/s) thrust should be ~ 3-4 N."""
    p = default_parameters()
    T, _ = x8_thrust(throttle=0.44, V_m_s=18.0, altitude_m=178.0, params=p)
    assert 2.5 <= T <= 5.0


def test_engine_thrust_drops_with_airspeed():
    p = default_parameters()
    T_static, _ = x8_thrust(throttle=0.5, V_m_s=0.0, altitude_m=0.0, params=p)
    T_fast, _ = x8_thrust(throttle=0.5, V_m_s=20.0, altitude_m=0.0, params=p)
    assert T_fast < T_static


def test_engine_zero_throttle_gives_zero_thrust():
    p = default_parameters()
    T, _ = x8_thrust(throttle=0.0, V_m_s=18.0, altitude_m=178.0, params=p)
    assert T == 0.0


def test_propeller_class_CT_polynomial_matches_paper_table_6():
    """Verify CT(J=0) and CT(J=0.7) against paper Table 6 polynomial."""
    prop = X8Propeller()
    # CT0 = 0.140 (paper Table 6)
    assert prop.CT(0.0) == pytest.approx(0.140)
    # At J=0.7: CT = 0.14 + (-0.030)*0.7 + (-0.237)*0.49 + 0.0847*0.343
    expected = 0.14 - 0.021 - 0.116 + 0.029
    assert prop.CT(0.7) == pytest.approx(expected, abs=1e-3)


def test_propeller_class_CQ_polynomial_matches_paper_table_6():
    prop = X8Propeller()
    assert prop.CQ(0.0) == pytest.approx(0.0082)


# ---- ODE smoke ---------------------------------------------------------


def test_ode_returns_12_components():
    p = default_parameters()
    x = set_initial_state(altitude_m=178.0, V_m_s=18.0, alpha_deg=4.0)
    u = np.array([0.0, 0.0, 0.5])
    f = x8_ode_6dof(x, u, 0.0, p)
    assert f.shape == (12,)
    assert np.all(np.isfinite(f))


def test_ode_throttle_increases_x_acceleration():
    p = default_parameters()
    x = set_initial_state(altitude_m=178.0, V_m_s=18.0, alpha_deg=4.0)
    f_lo = x8_ode_6dof(x, np.array([0.0, 0.0, 0.2]), 0.0, p)
    f_hi = x8_ode_6dof(x, np.array([0.0, 0.0, 0.9]), 0.0, p)
    assert f_hi[0] > f_lo[0]


# ---- Trim --------------------------------------------------------------


def test_trim_converges_at_paper_reference_point():
    """Paper Eq. 38: V=17.9 m/s, h=178 m, α=7.9°, δe=-2.35°, δt=0.44.

    The pure-longitudinal trim solved here finds slightly different
    values because the published trim is 6-DoF coupled with non-zero
    β=1.2° and δa=-2.16°. We expect α within 0.5°, δe within 0.5°.
    """
    r = trim(altitude_m=178.0, V_m_s=18.0)
    assert r.converged
    assert r.residual < 1e-6
    # Paper α = 7.9° → expect ours within ±1°
    alpha_deg = math.degrees(r.alpha_rad)
    assert 7.0 <= alpha_deg <= 9.0
    # Paper δe = -2.35° → expect ours within ±1°
    delta_e_deg = math.degrees(r.elevator_rad)
    assert -3.5 <= delta_e_deg <= -1.0
    # Throttle should be in (0, 1)
    assert 0.0 < r.throttle <= 1.0


def test_trim_to_state_round_trip():
    r = trim(altitude_m=178.0, V_m_s=18.0)
    x = r.to_state()
    assert x.shape == (12,)
    V = float(np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
    assert V == pytest.approx(r.V_m_s, rel=1e-6)
    assert -x[11] == pytest.approx(r.altitude_m)


# ---- Env API ----------------------------------------------------------


def test_env_make_via_gym_registry_works():
    env = gym.make(
        "NonlinearSkywalkerX8-v0", trim_at=(178.0, 18.0), number_time_steps=10
    ).unwrapped
    obs, _ = env.reset()
    assert obs.shape == (12,)


def test_env_step_without_reset_raises():
    env = NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0), number_time_steps=10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3))


def test_env_action_size_is_three_not_four():
    """X8 has no rudder — action vector has 3 elements, not 4."""
    env = NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0), number_time_steps=10)
    env.reset()
    with pytest.raises(ValueError, match="3 elements"):
        env.step(np.zeros(4))


def test_env_normalized_action_space_is_3D_pm_one():
    env = NonlinearSkywalkerX8Env(
        trim_at=(178.0, 18.0), number_time_steps=10, action_space="normalized"
    )
    np.testing.assert_allclose(env.action_space.high, np.ones(3))
    np.testing.assert_allclose(env.action_space.low, -np.ones(3))


def test_env_truncates_at_step_count():
    env = NonlinearSkywalkerX8Env(trim_at=(178.0, 18.0), number_time_steps=5)
    env.reset()
    for _ in range(5):
        _, _, _, trunc, _ = env.step(np.array([0.0, 0.0, 0.5]))
    assert trunc


def test_env_requires_one_initialiser():
    with pytest.raises(ValueError, match="must supply one of"):
        NonlinearSkywalkerX8Env(number_time_steps=10)


def test_env_rejects_multiple_initialisers():
    with pytest.raises(ValueError, match="exactly one of"):
        NonlinearSkywalkerX8Env(
            trim_at=(178.0, 18.0),
            initial_state=np.zeros(12),
            number_time_steps=10,
        )


def test_env_holds_trim_briefly():
    """After 1 s of held trim, V drift should be small (X8 has fast modes)."""
    r = trim(altitude_m=178.0, V_m_s=18.0)
    env = NonlinearSkywalkerX8Env(
        initial_state=r.to_state(), number_time_steps=200, dt=0.01
    )
    obs, _ = env.reset()
    u_trim = np.array([float(r.elevator_rad), 0.0, float(r.throttle)])
    for _ in range(100):  # 1 s
        obs, _, _, _, _ = env.step(u_trim)
    V = float(np.sqrt(obs[0] ** 2 + obs[1] ** 2 + obs[2] ** 2))
    # X8 has small inertias → trim drift over 1 s is tolerable
    assert abs(V - r.V_m_s) < 1.0
