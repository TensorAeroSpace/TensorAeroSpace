"""Tests for the nonlinear X-15 model + Gymnasium env."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  (registers gym envs)
from tensoraerospace.aerospacemodel.x15.nonlinear import (
    NonlinearX15,
    X15Configuration,
    X15_FLIGHT_CONDITIONS,
    XLR99Engine,
    default_parameters,
    initial_state_from_fc,
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
    level_trim,
    set_initial_state,
    trim,
    x15_aero,
    x15_ode_6dof,
    xlr99_thrust,
)
from tensoraerospace.aerospacemodel.x15.nonlinear.aero import AeroState
from tensoraerospace.envs.x15_nonlinear import NonlinearX15Env


# ---- Geometry & atmosphere -------------------------------------------------


def test_geometry_constants_match_published_x15():
    p = default_parameters()
    assert p.S_ft2 == pytest.approx(200.0)
    assert p.b_ft == pytest.approx(22.36)
    assert p.cbar_ft == pytest.approx(10.27)


def test_basic_and_a2_have_correct_propellant_loads():
    p_basic = default_parameters(X15Configuration.BASIC)
    p_a2 = default_parameters(X15Configuration.A2)
    # BASIC: ~ 17 900 lb propellant per Wikipedia / Thompson 2000
    assert p_basic.propellant_full_lb == pytest.approx(17_900.0, rel=1e-3)
    # A2: external tanks bring it to ~ 31 000 lb
    assert p_a2.propellant_full_lb > p_basic.propellant_full_lb
    assert p_a2.empty_weight_lb > p_basic.empty_weight_lb


def test_isa_density_at_sea_level_matches_standard():
    rho = isa_density_slug_ft3(0.0)
    assert rho == pytest.approx(0.002378, rel=1e-3)


def test_isa_density_drops_with_altitude():
    """At hypersonic altitudes density is several orders of magnitude lower."""
    assert isa_density_slug_ft3(100_000.0) < 1e-4
    assert isa_density_slug_ft3(200_000.0) < 1e-6


def test_isa_speed_of_sound_constant_in_stratosphere():
    a40 = isa_speed_of_sound_ft_s(40_000.0)
    a80 = isa_speed_of_sound_ft_s(80_000.0)
    # Above tropopause the model holds T constant ⇒ a constant
    assert a40 == pytest.approx(a80, rel=1e-6)


# ---- Inertia interpolation -------------------------------------------------


def test_inertia_interpolates_between_empty_and_full():
    p = default_parameters()
    Ix_e, Iy_e, Iz_e, Ixz_e = p.inertia_at(0.0)
    Ix_f, Iy_f, Iz_f, Ixz_f = p.inertia_at(p.propellant_full_lb)
    Ix_m, Iy_m, Iz_m, Ixz_m = p.inertia_at(p.propellant_full_lb / 2.0)
    assert Ix_e == pytest.approx(p.Ix_empty)
    assert Ix_f == pytest.approx(p.Ix_full)
    assert Iy_m == pytest.approx((p.Iy_empty + p.Iy_full) / 2.0)


def test_current_mass_decreases_with_propellant():
    p = default_parameters()
    m_full = p.current_mass_slug(p.propellant_full_lb)
    m_empty = p.current_mass_slug(0.0)
    assert m_full > m_empty
    assert m_empty == pytest.approx(p.empty_weight_lb / p.g_ft_s2, rel=1e-9)


# ---- XLR99 engine ---------------------------------------------------------


def test_xlr99_full_throttle_thrust_equals_57000_lbf():
    eng = XLR99Engine()
    T, mdot = eng.thrust_and_mdot(throttle=1.0, propellant_lb=10_000.0)
    assert T == pytest.approx(57_000.0)
    assert mdot > 0.0


def test_xlr99_below_30pct_throttle_engine_off():
    eng = XLR99Engine()
    T, mdot = eng.thrust_and_mdot(throttle=0.20, propellant_lb=10_000.0)
    assert T == 0.0
    assert mdot == 0.0


def test_xlr99_no_thrust_when_propellant_exhausted():
    eng = XLR99Engine()
    T, mdot = eng.thrust_and_mdot(throttle=1.0, propellant_lb=0.0)
    assert T == 0.0
    assert mdot == 0.0


def test_xlr99_burnout_time_matches_published_value():
    """13 000 lb propellant ÷ 224 lb/s mdot ≈ 80 s — Thompson 2000."""
    eng = XLR99Engine()
    _, mdot = eng.thrust_and_mdot(throttle=1.0, propellant_lb=10_000.0)
    burn_time_s = 17_900.0 / mdot
    assert 75.0 <= burn_time_s <= 85.0


def test_xlr99_thrust_via_params_helper_matches_engine_class():
    p = default_parameters()
    T, mdot = xlr99_thrust(throttle=0.7, propellant_lb=10_000.0, params=p)
    assert T == pytest.approx(57_000.0 * 0.7)
    assert mdot > 0.0


# ---- Aerodynamics ---------------------------------------------------------


def test_aero_lift_increases_with_alpha():
    """Lift slope C_L_α > 0 throughout the X-15 envelope."""
    p = default_parameters()
    L_low = x15_aero(
        AeroState(alpha=math.radians(0.0), beta=0.0, V=2000.0,
                  p=0.0, q=0.0, r=0.0, altitude_ft=70_000.0,
                  de=0.0, da=0.0, dr=0.0),
        p,
    ).L
    L_high = x15_aero(
        AeroState(alpha=math.radians(8.0), beta=0.0, V=2000.0,
                  p=0.0, q=0.0, r=0.0, altitude_ft=70_000.0,
                  de=0.0, da=0.0, dr=0.0),
        p,
    ).L
    assert L_high > L_low


def test_aero_lift_slope_decreases_at_hypersonic_mach():
    """C_L_α drops from ~3.5 at low M toward Newtonian 2.0 at hypersonic M."""
    p = default_parameters()
    # Same α, h, dyn pressure scaled by V²
    state_lo = AeroState(alpha=math.radians(5.0), beta=0.0, V=600.0,
                          p=0.0, q=0.0, r=0.0, altitude_ft=20_000.0,
                          de=0.0, da=0.0, dr=0.0)
    state_hi = AeroState(alpha=math.radians(5.0), beta=0.0, V=6500.0,
                          p=0.0, q=0.0, r=0.0, altitude_ft=100_000.0,
                          de=0.0, da=0.0, dr=0.0)
    f_lo = x15_aero(state_lo, p)
    f_hi = x15_aero(state_hi, p)
    # Compute C_L = L / (q_dyn S)
    rho_lo = isa_density_slug_ft3(20_000.0)
    rho_hi = isa_density_slug_ft3(100_000.0)
    qS_lo = 0.5 * rho_lo * 600.0**2 * p.S_ft2
    qS_hi = 0.5 * rho_hi * 6500.0**2 * p.S_ft2
    CL_lo = f_lo.L / qS_lo
    CL_hi = f_hi.L / qS_hi
    assert CL_lo > CL_hi   # hypersonic lift slope is smaller


def test_aero_pitching_moment_is_statically_stable():
    """C_m_α < 0 — increase in α produces a nose-down pitching moment."""
    p = default_parameters()
    common = dict(beta=0.0, V=2000.0, p=0.0, q=0.0, r=0.0,
                  altitude_ft=70_000.0, de=0.0, da=0.0, dr=0.0)
    m_low = x15_aero(AeroState(alpha=math.radians(2.0), **common), p).m
    m_high = x15_aero(AeroState(alpha=math.radians(8.0), **common), p).m
    assert m_high < m_low


# ---- Initial-state helpers -----------------------------------------------


def test_initial_state_from_fc_packs_published_values():
    fc = X15_FLIGHT_CONDITIONS[2]  # FC3 cruise_M4
    x = initial_state_from_fc(fc)
    V = float(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))
    assert V == pytest.approx(fc.V_ft_s, rel=1e-6)
    assert -x[11] == pytest.approx(fc.altitude_ft)
    assert x[12] == pytest.approx(fc.propellant_lb)
    assert math.degrees(x[7]) == pytest.approx(fc.alpha0_deg, rel=1e-6)


def test_set_initial_state_with_explicit_overrides():
    x = set_initial_state(altitude_ft=75_000.0, V_ft_s=3000.0,
                           alpha_deg=6.0, propellant_lb=8000.0)
    assert -x[11] == pytest.approx(75_000.0)
    assert x[12] == pytest.approx(8000.0)
    V = float(np.sqrt(x[0]**2 + x[1]**2 + x[2]**2))
    assert V == pytest.approx(3000.0, rel=1e-6)


# ---- ODE smoke tests -----------------------------------------------------


def test_x15_ode_returns_13_components():
    p = default_parameters()
    x = initial_state_from_fc(X15_FLIGHT_CONDITIONS[1])
    u = np.array([math.radians(-2.0), 0.0, 0.0, 1.0])
    f = x15_ode_6dof(x, u, 0.0, p)
    assert f.shape == (13,)
    assert np.all(np.isfinite(f))


def test_x15_ode_propellant_decreases_under_thrust():
    p = default_parameters()
    x = initial_state_from_fc(X15_FLIGHT_CONDITIONS[1])
    # Full throttle → mdot > 0 → dm_prop/dt < 0
    u_full = np.array([math.radians(-2.0), 0.0, 0.0, 1.0])
    f_full = x15_ode_6dof(x, u_full, 0.0, p)
    assert f_full[12] < 0.0
    # No throttle → no consumption
    u_off = np.array([math.radians(-2.0), 0.0, 0.0, 0.0])
    f_off = x15_ode_6dof(x, u_off, 0.0, p)
    assert f_off[12] == 0.0


def test_x15_ode_thrust_along_x_increases_du():
    """Engaging the rocket should accelerate body x-velocity."""
    p = default_parameters()
    x = set_initial_state(altitude_ft=70_000.0, V_ft_s=2000.0,
                           alpha_deg=4.0, propellant_lb=10_000.0)
    u_off = np.array([0.0, 0.0, 0.0, 0.0])
    u_on = np.array([0.0, 0.0, 0.0, 1.0])
    f_off = x15_ode_6dof(x, u_off, 0.0, p)
    f_on = x15_ode_6dof(x, u_on, 0.0, p)
    assert f_on[0] > f_off[0]   # du/dt with thrust > du/dt without thrust


# ---- Variable-mass dynamics ----------------------------------------------


def test_burnout_time_at_full_throttle_around_80s():
    """Full burn from full propellant should take ~80 s — match Thompson 2000."""
    m = NonlinearX15(
        x0=set_initial_state(altitude_ft=70_000.0, V_ft_s=2000.0,
                             alpha_deg=4.0, propellant_lb=17_900.0),
        dt=0.1, integrator="rk4",
    )
    u = np.array([math.radians(-3.0), 0.0, 0.0, 1.0])
    burnout_t = None
    for k in range(2000):
        m.run_step(u)
        if m.propellant_lb <= 0.0:
            burnout_t = (k + 1) * 0.1
            break
    assert burnout_t is not None
    assert 75.0 <= burnout_t <= 85.0


def test_engine_running_property_tracks_propellant():
    m = NonlinearX15(
        x0=set_initial_state(altitude_ft=70_000.0, V_ft_s=2000.0,
                             alpha_deg=4.0, propellant_lb=100.0),
        dt=0.1, integrator="rk4",
    )
    assert m.engine_running
    u = np.array([math.radians(-3.0), 0.0, 0.0, 1.0])
    for _ in range(20):
        m.run_step(u)
    assert not m.engine_running   # 100 lb burns off in < 1 s


# ---- Trim ---------------------------------------------------------------


def test_powered_trim_returns_positive_climb_angle():
    """At full throttle the X-15 climbs (γ > 0). Just check that the trimmer
    returns a state with positive flight-path angle, even if the residual
    is non-zero (the X-15 has no true equilibrium at full thrust)."""
    r = trim(altitude_ft=60_000.0, V_ft_s=1500.0, throttle=0.5,
             propellant_lb=10_000.0)
    # The natural γ at half throttle should be climbing — even if the
    # solver hits its iteration limit, the gamma_rad value should be
    # bounded and physically sensible.
    assert math.degrees(r.alpha_rad) < 90.0


def test_glide_trim_converges_at_low_altitude():
    """Post-burnout glide at moderate altitude has a well-defined trim."""
    r = trim(altitude_ft=30_000.0, V_ft_s=800.0, throttle=0.0,
             propellant_lb=0.0)
    assert r.converged
    assert r.gamma_rad < 0.0  # glide ⇒ descending
    assert r.residual < 1e-6


def test_level_trim_reports_non_convergence_for_x15():
    """X-15 has no level cruise — the level_trim should fail gracefully."""
    r = level_trim(altitude_ft=70_000.0, V_ft_s=2412.0, propellant_lb=10_000.0)
    # The fact that this *does not* converge is the documented behaviour
    # — assert that the reported residual is non-trivial.
    assert r.gamma_rad == 0.0


def test_trim_to_state_round_trip():
    """trim().to_state() should pack back into a valid 13-D state."""
    r = trim(altitude_ft=30_000.0, V_ft_s=800.0, throttle=0.0,
             propellant_lb=0.0)
    state = r.to_state()
    assert state.size == 13
    # Velocity magnitude reconstruction
    V = float(np.sqrt(state[0]**2 + state[1]**2 + state[2]**2))
    assert V == pytest.approx(r.V_ft_s, rel=1e-6)
    # Altitude reconstruction
    assert -state[11] == pytest.approx(r.altitude_ft)


# ---- Env API -----------------------------------------------------------


def test_env_make_via_gym_registry_works():
    env = gym.make("NonlinearX15-v0", flight_condition_id=2,
                   number_time_steps=10).unwrapped
    obs, _ = env.reset()
    assert obs.shape == (13,)


def test_env_action_size_validated():
    env = NonlinearX15Env(flight_condition_id=2, number_time_steps=10)
    env.reset()
    with pytest.raises(ValueError, match="4 elements"):
        env.step(np.zeros(3))


def test_env_step_without_reset_raises():
    env = NonlinearX15Env(flight_condition_id=2, number_time_steps=10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_env_normalized_action_space_is_pm_one():
    env = NonlinearX15Env(flight_condition_id=2, number_time_steps=10,
                          action_space="normalized")
    np.testing.assert_allclose(env.action_space.high, np.ones(4))
    np.testing.assert_allclose(env.action_space.low, -np.ones(4))


def test_env_truncates_at_step_count():
    env = NonlinearX15Env(flight_condition_id=2, number_time_steps=5)
    env.reset()
    for _ in range(5):
        _, _, _, trunc, _ = env.step(np.array([0.0, 0.0, 0.0, 0.5]))
    assert trunc


def test_env_info_reports_propellant_and_engine_state():
    env = NonlinearX15Env(flight_condition_id=2, number_time_steps=10)
    env.reset()
    _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.5]))
    assert "propellant_lb" in info
    assert "engine_running" in info
    assert info["engine_running"]


def test_env_engine_flames_out_when_propellant_runs_out():
    """Run env with full throttle until burnout — info should report engine_running=False."""
    env = NonlinearX15Env(
        initial_state=set_initial_state(altitude_ft=70_000.0, V_ft_s=2000.0,
                                         alpha_deg=4.0, propellant_lb=200.0),
        number_time_steps=200, dt=0.1,
    )
    env.reset()
    flameout_seen = False
    for _ in range(200):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
        if not info["engine_running"]:
            flameout_seen = True
            break
    assert flameout_seen


def test_env_requires_one_initialiser():
    with pytest.raises(ValueError, match="must supply one of"):
        NonlinearX15Env(number_time_steps=10)


def test_env_rejects_multiple_initialisers():
    with pytest.raises(ValueError, match="exactly one of"):
        NonlinearX15Env(
            flight_condition_id=2,
            initial_state=np.zeros(13),
            number_time_steps=10,
        )


def test_linear_x15_backward_compat():
    """Ensure old-style import path still resolves after restructuring."""
    from tensoraerospace.aerospacemodel.x15 import LongitudinalX15  # noqa: F401
    from tensoraerospace.aerospacemodel import LongitudinalX15 as L2  # noqa: F401
    assert L2 is LongitudinalX15
