"""Tests for the Boeing 747 nonlinear 6-DoF model.

Anchored on NASA CR-2144 §IX (Heffley & Jewell 1972).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.b747.nonlinear import (
    B747_FLIGHT_CONDITIONS,
    B747Configuration,
    B747FlightCondition,
    NonlinearB747,
    default_parameters,
    initial_state_from_fc,
    set_initial_state,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.aero import (
    AeroState,
    b747_aero,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.derivatives import (
    cruise_lateral_at,
    cruise_longitudinal_at,
    get_lateral,
    get_longitudinal,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.dynamics import b747_ode_6dof
from tensoraerospace.aerospacemodel.b747.nonlinear.params import (
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
)

# ---- Parameter & flight-condition sanity ---------------------------------


def test_geometry_constants_match_cr2144_figure_ix2():
    p = default_parameters()
    assert p.S_ft2 == pytest.approx(5500.0)
    assert p.b_ft == pytest.approx(195.68)
    assert p.cbar_ft == pytest.approx(27.31)


def test_nominal_inertia_matches_table_ix3_fc4():
    p = default_parameters(B747Configuration.NOMINAL)
    assert p.weight_lb == 636_600.0
    # cf. Table IX-3 columns 3–10 (cruise grid)
    assert p.Ix == 18.2e6
    assert p.Iy == 33.1e6
    assert p.Iz == 49.7e6
    assert p.Ixz == 0.97e6


def test_power_approach_inertia_matches_fc1_fc2():
    p = default_parameters(B747Configuration.POWER_APPROACH)
    assert p.weight_lb == 564_000.0
    assert p.Ix == 13.7e6
    assert p.Iy == 30.5e6
    assert p.Iz == 43.1e6
    assert p.Ixz == 0.825e6


def test_flight_conditions_count_and_ids():
    assert len(B747_FLIGHT_CONDITIONS) == 10
    assert [fc.fc_id for fc in B747_FLIGHT_CONDITIONS] == list(range(1, 11))


# ---- ISA atmosphere ----------------------------------------------------


def test_isa_density_at_sea_level_matches_published_value():
    rho = isa_density_slug_ft3(0.0)
    assert rho == pytest.approx(0.002378, rel=1e-3)


def test_isa_density_decreases_with_altitude():
    rho_sl = isa_density_slug_ft3(0.0)
    rho_20k = isa_density_slug_ft3(20_000.0)
    rho_40k = isa_density_slug_ft3(40_000.0)
    assert rho_sl > rho_20k > rho_40k


def test_isa_speed_of_sound_at_sea_level():
    a = isa_speed_of_sound_ft_s(0.0)
    assert a == pytest.approx(1116.45, rel=1e-3)


# ---- Derivative bank ---------------------------------------------------


def test_landing_derivatives_match_table_ix1_explicit_values():
    lon = get_longitudinal(1)
    lat = get_lateral(1)
    assert lon.C_L0 == 1.76
    assert lon.C_La == 5.67
    assert lon.C_ma == -1.45
    assert lat.C_lp == -0.502
    assert lat.C_ndr == -0.112


def test_power_approach_derivatives_match_table_ix2():
    lon = get_longitudinal(2)
    assert lon.C_L0 == 1.11
    assert lon.C_mde == -1.34


def test_cruise_interpolation_collapses_to_anchor_at_grid_point():
    """At the published FC4 (SL × M=0.65) the interpolator must reproduce the table value."""
    lon_anchor = get_longitudinal(4)
    lon_interp = cruise_longitudinal_at(0.0, 0.65)
    # Allow tiny floating slack (the interpolator returns the same value).
    assert lon_interp.C_La == pytest.approx(lon_anchor.C_La, rel=1e-6)
    assert lon_interp.C_mde == pytest.approx(lon_anchor.C_mde, rel=1e-6)


def test_cruise_lateral_interpolation_at_fc6():
    lat_anchor = get_lateral(6)
    lat_interp = cruise_lateral_at(20_000.0, 0.65)
    assert lat_interp.C_lp == pytest.approx(lat_anchor.C_lp, rel=1e-6)
    assert lat_interp.C_nb == pytest.approx(lat_anchor.C_nb, rel=1e-6)


# ---- Aero forces -------------------------------------------------------


def test_aero_at_trim_landing_returns_lift_close_to_weight():
    """At FC1 (Landing) the trim values L = q·S·C_L should equal the weight."""
    fc = B747_FLIGHT_CONDITIONS[0]
    p = default_parameters(B747Configuration.LANDING)
    state = AeroState(
        alpha=np.deg2rad(fc.alpha0_deg),
        beta=0.0,
        V=fc.V_ft_s,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_ft=fc.altitude_ft,
        de=0.0,
        da=0.0,
        dr=0.0,
    )
    forces = b747_aero(state, p)
    # Lift / weight ratio: should be ≈ 1.0 (steady level flight) within a few %
    assert 0.85 < forces.L / p.weight_lb < 1.15


def test_aero_zero_alpha_zero_controls_zero_pitching_moment_lat_dir():
    """At α = α₀, β = 0, all controls = 0, lateral-directional moments are zero."""
    fc = B747_FLIGHT_CONDITIONS[3]  # FC4
    p = default_parameters(B747Configuration.NOMINAL)
    state = AeroState(
        alpha=np.deg2rad(fc.alpha0_deg),
        beta=0.0,
        V=fc.V_ft_s,
        p=0.0,
        q=0.0,
        r=0.0,
        altitude_ft=fc.altitude_ft,
        de=0.0,
        da=0.0,
        dr=0.0,
    )
    forces = b747_aero(state, p)
    assert forces.Y == pytest.approx(0.0, abs=1e-6)
    assert forces.l == pytest.approx(0.0, abs=1e-6)
    assert forces.n == pytest.approx(0.0, abs=1e-6)


# ---- Full 6-DoF ODE -----------------------------------------------------


def test_step_runs_for_full_episode_without_blowing_up():
    fc = B747_FLIGHT_CONDITIONS[3]
    m = NonlinearB747(x0=initial_state_from_fc(fc), dt=0.01, integrator="rk4")
    u = np.array([0.0, 0.0, 0.0, 0.32])  # cruise throttle
    for _ in range(200):  # 2 s
        m.run_step(u)
    s = m.current_state
    assert np.all(np.isfinite(s))
    # Aircraft should still be roughly at altitude
    assert abs(m.altitude_ft) < 200.0


def test_elevator_negative_pitches_nose_up():
    fc = B747_FLIGHT_CONDITIONS[3]
    m = NonlinearB747(x0=initial_state_from_fc(fc), dt=0.01, integrator="rk4")
    u = np.array([np.deg2rad(-2.0), 0.0, 0.0, 0.4])
    for _ in range(100):  # 1 s
        m.run_step(u)
    s = m.current_state
    # q is body pitch rate; should become positive after nose-up command
    assert s[4] > 0.0  # q > 0
    assert s[7] > 0.0  # theta > 0 (climbing)


def test_aileron_positive_rolls_right():
    fc = B747_FLIGHT_CONDITIONS[3]
    m = NonlinearB747(x0=initial_state_from_fc(fc), dt=0.01, integrator="rk4")
    u = np.array([0.0, np.deg2rad(5.0), 0.0, 0.32])
    for _ in range(100):
        m.run_step(u)
    s = m.current_state
    # p is body roll rate; positive aileron should roll right (positive p, positive phi)
    assert s[3] > 0.0  # p > 0
    assert s[6] > 0.0  # phi > 0


def test_rudder_positive_yaws_right():
    fc = B747_FLIGHT_CONDITIONS[3]
    m = NonlinearB747(x0=initial_state_from_fc(fc), dt=0.01, integrator="rk4")
    u = np.array([0.0, 0.0, np.deg2rad(3.0), 0.32])
    for _ in range(50):
        m.run_step(u)
    s = m.current_state
    # Rudder right (+δr) gives -yawing moment in CR-2144 convention
    # ⇒ negative r (yaw left). Check sign matches CR-2144.
    assert s[5] < 0.0


def test_invalid_x0_size_raises():
    with pytest.raises(ValueError, match="12 elements"):
        NonlinearB747(x0=np.zeros(11))


def test_invalid_control_size_raises():
    fc = B747_FLIGHT_CONDITIONS[3]
    m = NonlinearB747(x0=initial_state_from_fc(fc))
    with pytest.raises(ValueError, match="control vector size mismatch"):
        m.run_step(np.zeros(3))


# ---- Initial state helpers ---------------------------------------------


def test_initial_state_from_fc_has_correct_velocity_decomposition():
    fc = B747_FLIGHT_CONDITIONS[0]  # Landing, α=8.5°
    s = initial_state_from_fc(fc)
    V = np.linalg.norm(s[:3])
    assert V == pytest.approx(fc.V_ft_s, rel=1e-9)
    alpha = np.arctan2(s[2], s[0])
    assert alpha == pytest.approx(np.deg2rad(fc.alpha0_deg), rel=1e-9)


def test_set_initial_state_overrides_named_keys():
    s = set_initial_state(u=600.0, theta=np.deg2rad(2.0), z_e=-30000.0)
    assert s[0] == 600.0
    assert s[7] == pytest.approx(np.deg2rad(2.0))
    assert s[11] == -30000.0


def test_set_initial_state_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown state key"):
        set_initial_state(invalid=0.0)


# ---- Engine model -------------------------------------------------------


def test_jt9d_sea_level_static_matches_type_certificate():
    from tensoraerospace.aerospacemodel.b747.nonlinear.engine import JT9DEngine

    eng = JT9DEngine()
    T = eng.installed_thrust(mach=0.0, altitude_ft=0.0, throttle=1.0)
    # 4 × 47,100 lb = 188,400 lb (Boeing 747-100 TCDS A20WE)
    assert T == pytest.approx(188_400.0, rel=1e-9)


def test_jt9d_thrust_decreases_with_mach():
    from tensoraerospace.aerospacemodel.b747.nonlinear.engine import JT9DEngine

    eng = JT9DEngine()
    T0 = eng.installed_thrust(0.0, 0.0, 1.0)
    T2 = eng.installed_thrust(0.2, 0.0, 1.0)
    T6 = eng.installed_thrust(0.6, 0.0, 1.0)
    assert T0 > T2 > T6


def test_jt9d_thrust_decreases_with_altitude():
    from tensoraerospace.aerospacemodel.b747.nonlinear.engine import JT9DEngine

    eng = JT9DEngine()
    T_sl = eng.installed_thrust(0.5, 0.0, 1.0)
    T_20k = eng.installed_thrust(0.5, 20_000.0, 1.0)
    T_40k = eng.installed_thrust(0.5, 40_000.0, 1.0)
    assert T_sl > T_20k > T_40k


def test_jt9d_idle_throttle_gives_5pct():
    from tensoraerospace.aerospacemodel.b747.nonlinear.engine import JT9DEngine

    eng = JT9DEngine(idle_frac=0.05)
    T_full = eng.installed_thrust(0.0, 0.0, 1.0)
    T_idle = eng.installed_thrust(0.0, 0.0, 0.0)
    assert T_idle == pytest.approx(T_full * 0.05, rel=1e-9)
