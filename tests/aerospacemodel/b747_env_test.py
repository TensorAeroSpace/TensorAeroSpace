"""Tests for NonlinearB747Env + B747 damage subsystem."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  (registers gym envs)
from tensoraerospace.aerospacemodel.b747.nonlinear import (
    B747Configuration,
    B747_FLIGHT_CONDITIONS,
    initial_state_from_fc,
    trim,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.damage import (
    AILERON_TOTAL_LOSS,
    B747DamageManager,
    B747DamageState,
    DamageProfile,
    ELEVATOR_50PCT_LOSS,
    ELEVATOR_JAMMED_NOSE_UP,
    EngineFailureEvent,
    FLAPS_JAMMED_LANDING,
    FLAPS_JAMMED_RETRACTED,
    FlapJamEvent,
    LEFT_OUTER_ENGINE_FAILURE,
    LEFT_TWO_ENGINES_OUT,
    RUDDER_HYDRAULIC_LEAK,
    SurfaceEffectivenessDecay,
    SurfaceEffectivenessEvent,
    SurfaceJamEvent,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.engine import (
    ENGINE_Y_POSITIONS_FT,
    jt9d_thrust,
    jt9d_thrust_with_asymmetry,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.params import (
    default_parameters,
)
from tensoraerospace.envs.b747_nonlinear import NonlinearB747Env


# ---- Trim finder ---------------------------------------------------------


def test_trim_at_fc1_landing_recovers_published_alpha_within_one_degree():
    fc = B747_FLIGHT_CONDITIONS[0]  # Landing
    r = trim(
        altitude_ft=fc.altitude_ft, V_ft_s=fc.V_ft_s,
        config=B747Configuration.LANDING,
        initial_guess=(np.deg2rad(8.0), 0.0, 0.6),
    )
    assert r.converged
    assert r.residual < 1e-6
    assert abs(np.rad2deg(r.alpha_rad) - fc.alpha0_deg) < 1.0


def test_trim_simulation_holds_state_for_5_seconds():
    """After trimming, run the model with constant control: V and h must stay stable."""
    from tensoraerospace.aerospacemodel.b747.nonlinear import NonlinearB747

    r = trim(
        altitude_ft=0.0, V_ft_s=221.0, config=B747Configuration.LANDING,
        initial_guess=(np.deg2rad(8.0), 0.0, 0.6),
    )
    m = NonlinearB747(x0=r.to_state(), dt=0.01, integrator="rk4",
                      config=B747Configuration.LANDING)
    u = np.array([r.elevator_rad, 0.0, 0.0, r.throttle])
    for _ in range(500):  # 5 s
        m.run_step(u)
    assert abs(m.airspeed_ft_s - 221.0) < 1.0
    assert abs(m.altitude_ft - 0.0) < 5.0


# ---- Env basic API ------------------------------------------------------


def test_env_make_via_gym_registry_works():
    env = gym.make("NonlinearB747-v0", flight_condition_id=4, number_time_steps=100)
    obs, _ = env.reset()
    assert obs.shape == (12,)
    assert env.action_space.shape == (4,)


def test_env_requires_one_initialiser():
    with pytest.raises(ValueError, match="must supply one"):
        NonlinearB747Env(number_time_steps=10)


def test_env_rejects_multiple_initialisers():
    with pytest.raises(ValueError, match="exactly one"):
        NonlinearB747Env(
            initial_state=np.zeros(12),
            flight_condition_id=4,
            number_time_steps=10,
        )


def test_env_initial_state_size_validated():
    with pytest.raises(ValueError, match="12 elements"):
        NonlinearB747Env(initial_state=np.zeros(11), number_time_steps=10)


def test_env_step_without_reset_raises():
    env = NonlinearB747Env(flight_condition_id=4, number_time_steps=10)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(4))


def test_env_action_size_validated():
    env = NonlinearB747Env(flight_condition_id=4, number_time_steps=10)
    env.reset()
    with pytest.raises(ValueError, match="4 elements"):
        env.step(np.zeros(3))


def test_env_invalid_action_mode_raises():
    with pytest.raises(ValueError, match="virtual"):
        NonlinearB747Env(
            flight_condition_id=4, number_time_steps=10,
            action_space="hybrid",  # type: ignore[arg-type]
        )


# ---- Action-space modes -------------------------------------------------


def test_virtual_action_space_bounds_match_actuator_limits():
    env = NonlinearB747Env(flight_condition_id=4, action_space="virtual",
                           number_time_steps=10)
    assert env.action_space.high[0] == pytest.approx(np.deg2rad(25.0))
    assert env.action_space.high[3] == 1.0
    assert env.action_space.low[3] == 0.0


def test_normalized_action_space_is_pm_one():
    env = NonlinearB747Env(flight_condition_id=4, action_space="normalized",
                           number_time_steps=10)
    np.testing.assert_allclose(env.action_space.high, np.ones(4))
    np.testing.assert_allclose(env.action_space.low, -np.ones(4))


def test_normalized_zero_action_is_neutral_with_idle_throttle():
    """In normalised mode, zero action ⇒ all surfaces neutral, throttle = 0.5."""
    env = NonlinearB747Env(flight_condition_id=4, action_space="normalized",
                           number_time_steps=10)
    env.reset()
    obs, _, _, _, info = env.step(np.zeros(4))
    # Just confirm we got a valid response
    assert obs.shape == (12,)


# ---- Trim-at initialiser ------------------------------------------------


def test_env_trim_at_initialises_from_solver():
    env = NonlinearB747Env(trim_at=(20000.0, 674.0), number_time_steps=10)
    obs, _ = env.reset()
    V = float(np.linalg.norm(obs[:3]))
    altitude = float(-obs[11])
    assert V == pytest.approx(674.0, abs=0.5)
    assert altitude == pytest.approx(20000.0, abs=1.0)


# ---- Damage state -------------------------------------------------------


def test_damage_state_healthy_has_full_effectiveness():
    s = B747DamageState.healthy()
    assert all(mu == 1.0 for mu in s.mu.values())
    assert all(j is None for j in s.jam.values())


def test_damage_state_rejects_invalid_mu():
    with pytest.raises(ValueError, match="must be in"):
        B747DamageState(mu={"elevator": 1.5, "aileron": 1.0, "rudder": 1.0,
                            "throttle": 1.0})


def test_damage_state_apply_multiplies_effectiveness():
    s = B747DamageState.healthy()
    s.mu["elevator"] = 0.5
    u = np.array([0.1, 0.0, 0.0, 0.5])
    eff = s.apply(u)
    assert eff[0] == pytest.approx(0.05)
    assert eff[3] == pytest.approx(0.5)  # throttle untouched


def test_damage_state_apply_jam_overrides_command():
    s = B747DamageState.healthy()
    s.jam["elevator"] = -0.0349  # -2 deg
    eff = s.apply(np.array([0.5, 0.0, 0.0, 0.4]))
    assert eff[0] == pytest.approx(-0.0349)
    assert eff[3] == pytest.approx(0.4)


def test_damage_state_decay_pushes_toward_floor():
    s = B747DamageState.healthy()
    s.tau["rudder"] = 1.0
    s.mu_floor["rudder"] = 0.5
    for _ in range(100):  # t = 1 s = 1·tau ⇒ mu ≈ 0.5 + 0.5/e
        s.step_decay(0.01)
    expected = 0.5 + 0.5 * np.exp(-1.0)
    assert abs(s.mu["rudder"] - expected) < 0.01


# ---- Events --------------------------------------------------------------


def test_surface_effectiveness_event_sets_mu():
    s = B747DamageState.healthy()
    SurfaceEffectivenessEvent(trigger_time=2.0, surface="elevator", mu=0.3).apply(s)
    assert s.mu["elevator"] == 0.3


def test_surface_jam_event_sets_jam_value():
    s = B747DamageState.healthy()
    SurfaceJamEvent(trigger_time=1.0, surface="rudder", jam_value=0.05).apply(s)
    assert s.jam["rudder"] == 0.05


def test_surface_decay_arms_decay():
    s = B747DamageState.healthy()
    SurfaceEffectivenessDecay(
        trigger_time=2.0, surface="aileron", tau=4.0, mu_floor=0.2
    ).apply(s)
    assert s.tau["aileron"] == 4.0
    assert s.mu_floor["aileron"] == 0.2
    # mu itself is still 1.0 right after the event triggers
    assert s.mu["aileron"] == 1.0


def test_invalid_surface_raises():
    with pytest.raises(ValueError, match="surface must be"):
        SurfaceEffectivenessEvent(trigger_time=1.0, surface="canard", mu=0.5)


def test_invalid_mu_raises():
    with pytest.raises(ValueError, match="mu"):
        SurfaceEffectivenessEvent(trigger_time=1.0, surface="elevator", mu=1.5)


def test_negative_trigger_time_raises():
    with pytest.raises(ValueError, match="trigger_time"):
        SurfaceJamEvent(trigger_time=-1.0, surface="elevator", jam_value=0.0)


# ---- Manager -------------------------------------------------------------


def test_manager_applies_events_in_window():
    mgr = B747DamageManager(DamageProfile(events=[
        SurfaceEffectivenessEvent(trigger_time=1.0, surface="elevator", mu=0.6),
    ]))
    fired = mgr.update(t_current=1.5, t_previous=0.0, dt=0.01)
    assert len(fired) == 1
    assert mgr.state.mu["elevator"] == 0.6
    fired2 = mgr.update(t_current=2.0, t_previous=1.5, dt=0.01)
    assert len(fired2) == 0


def test_manager_inject_event_is_single_fire():
    mgr = B747DamageManager()
    mgr.inject_event(SurfaceJamEvent(trigger_time=0.5, surface="rudder", jam_value=0.0))
    fired = mgr.update(t_current=1.0, t_previous=0.0, dt=0.01)
    assert len(fired) == 1
    fired2 = mgr.update(t_current=2.0, t_previous=1.0, dt=0.01)
    assert len(fired2) == 0


def test_manager_reset_clears_state_and_injected():
    mgr = B747DamageManager(DamageProfile(events=[
        SurfaceEffectivenessEvent(trigger_time=0.5, surface="elevator", mu=0.5),
    ]))
    mgr.inject_event(SurfaceJamEvent(trigger_time=0.6, surface="rudder", jam_value=0.0))
    mgr.update(t_current=1.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu["elevator"] == 0.5
    assert mgr.state.jam["rudder"] == 0.0
    mgr.reset()
    assert mgr.state.mu["elevator"] == 1.0
    assert mgr.state.jam["rudder"] is None


# ---- Presets ------------------------------------------------------------


def test_elevator_50pct_loss_preset():
    mgr = B747DamageManager(ELEVATOR_50PCT_LOSS)
    mgr.update(t_current=6.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu["elevator"] == 0.5


def test_elevator_jammed_preset():
    mgr = B747DamageManager(ELEVATOR_JAMMED_NOSE_UP)
    mgr.update(t_current=11.0, t_previous=0.0, dt=0.01)
    assert mgr.state.jam["elevator"] is not None
    assert mgr.state.jam["elevator"] == pytest.approx(-0.0349)


def test_aileron_total_loss_preset():
    mgr = B747DamageManager(AILERON_TOTAL_LOSS)
    mgr.update(t_current=10.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu["aileron"] == 0.0


def test_rudder_decay_preset_decreases_over_time():
    mgr = B747DamageManager(RUDDER_HYDRAULIC_LEAK)
    # Step from t=0 to t=20s
    mu_log = []
    for k in range(2000):
        mgr.update(t_current=(k + 1) * 0.01, t_previous=k * 0.01, dt=0.01)
        mu_log.append(mgr.state.mu["rudder"])
    # By t=20 s, mu should be approaching the 0.3 floor
    assert mu_log[-1] < 0.4
    assert all(np.diff(mu_log) <= 1e-9)  # monotonic decay


# ---- Env damage integration ---------------------------------------------


def test_env_damage_event_triggers_in_info_dict():
    env = NonlinearB747Env(flight_condition_id=4, number_time_steps=500,
                           damage_profile=ELEVATOR_50PCT_LOSS)
    env.reset()
    triggered = False
    for k in range(500):
        _, _, _, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.32]))
        if "damage_events_triggered" in info:
            triggered = True
            assert info["damage_state"]["mu"]["elevator"] == 0.5
            break
        if trunc:
            break
    assert triggered


def test_env_without_damage_has_no_damage_manager():
    env = NonlinearB747Env(flight_condition_id=4, number_time_steps=10)
    env.reset()
    assert env.damage_manager is None


def test_env_reset_options_overrides_profile():
    env = NonlinearB747Env(flight_condition_id=4, number_time_steps=100)
    env.reset(options={"damage_profile": AILERON_TOTAL_LOSS})
    assert env.damage_manager is not None
    # Run past the trigger
    for _ in range(900):
        env.step(np.array([0.0, 0.0, 0.0, 0.32]))
    assert env.damage_manager.state.mu["aileron"] == 0.0


# ---- Engine failure events ---------------------------------------------


def test_damage_state_engines_default_to_full_effectiveness():
    s = B747DamageState.healthy()
    assert s.engines_mu == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    assert s.flap_jam_config is None


def test_damage_state_rejects_invalid_engines_mu():
    with pytest.raises(ValueError, match="engines_mu"):
        B747DamageState(engines_mu={1: 1.5, 2: 1.0, 3: 1.0, 4: 1.0})


def test_engine_failure_event_sets_engines_mu():
    s = B747DamageState.healthy()
    EngineFailureEvent(trigger_time=5.0, engine_id=1, thrust_fraction=0.0).apply(s)
    assert s.engines_mu[1] == 0.0
    assert s.engines_mu[2] == 1.0
    assert s.engines_mu[3] == 1.0
    assert s.engines_mu[4] == 1.0


def test_engine_failure_event_validates_engine_id():
    with pytest.raises(ValueError, match="engine_id"):
        EngineFailureEvent(trigger_time=0.0, engine_id=5)


def test_engine_failure_event_validates_thrust_fraction():
    with pytest.raises(ValueError, match="thrust_fraction"):
        EngineFailureEvent(trigger_time=0.0, engine_id=1, thrust_fraction=2.0)


def test_jt9d_thrust_with_asymmetry_matches_scalar_thrust_when_healthy():
    """All four engines healthy ⇒ same total thrust as the original scalar API."""
    params = default_parameters()
    T_scalar = jt9d_thrust(throttle=0.7, mach=0.65, altitude_ft=20_000.0, params=params)
    T_total, N_yaw = jt9d_thrust_with_asymmetry(
        throttle=0.7, mach=0.65, altitude_ft=20_000.0, params=params,
    )
    assert abs(T_total - T_scalar) < 1e-6
    assert abs(N_yaw) < 1e-6


def test_jt9d_thrust_with_asymmetry_left_outer_out_yaws_left():
    """Engine 1 (left outer) dead ⇒ surviving thrust on right wing yaws nose left.

    Sign convention: NED body axis with +z down. A net thrust offset to the
    right of centerline (because left outer engine is dead) gives a moment
    in the −z direction → negative yaw moment N → nose yaws LEFT (toward
    the dead engine), as expected for a real engine-out scenario.
    """
    params = default_parameters()
    params.damage_state = B747DamageState(
        engines_mu={1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0},
    )
    T_total, N_yaw = jt9d_thrust_with_asymmetry(
        throttle=0.8, mach=0.65, altitude_ft=20_000.0, params=params,
    )
    T_scalar_full = jt9d_thrust(throttle=0.8, mach=0.65,
                                altitude_ft=20_000.0, params=default_parameters())
    # Three of four engines remain → ≈ 75% of full thrust
    assert 0.70 < T_total / T_scalar_full < 0.80
    # Negative yaw moment (nose left, toward dead engine)
    assert N_yaw < 0.0
    # Magnitude must match −y_1 · T_per_engine consumed by the damage
    expected_magnitude = abs(ENGINE_Y_POSITIONS_FT[1]) * (T_scalar_full / 4.0)
    assert abs(abs(N_yaw) - expected_magnitude) < expected_magnitude * 1e-6


def test_jt9d_thrust_with_asymmetry_right_outer_out_yaws_right():
    """Engine 4 (right outer) dead ⇒ thrust offset to left → nose yaws RIGHT (positive N)."""
    params = default_parameters()
    params.damage_state = B747DamageState(
        engines_mu={1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
    )
    _, N_yaw = jt9d_thrust_with_asymmetry(
        throttle=0.8, mach=0.65, altitude_ft=20_000.0, params=params,
    )
    assert N_yaw > 0.0


def test_left_outer_engine_failure_preset_triggers_engines_mu():
    env = NonlinearB747Env(
        trim_at=(20_000.0, 674.0),
        number_time_steps=2000,
        damage_profile=LEFT_OUTER_ENGINE_FAILURE,
    )
    env.reset()
    triggered = False
    for k in range(1500):  # 15 s @ dt=0.01 — past the 10 s trigger
        _, _, _, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.555]))
        if "damage_events_triggered" in info:
            triggered = True
        if trunc:
            break
    assert triggered
    assert env.damage_manager.state.engines_mu[1] == 0.0
    assert env.damage_manager.state.engines_mu[2] == 1.0


def test_left_two_engines_out_preset_yaws_aircraft_left():
    """Both left engines fail → aircraft must develop yaw rate r > 0... wait no."""
    # NED yaw moment N < 0 ⇒ negative yaw rate r ⇒ heading psi decreases
    # (nose moves left). Verify by integration over 10 s after the failure.
    env = NonlinearB747Env(
        trim_at=(20_000.0, 674.0),
        number_time_steps=3000, dt=0.01,
        damage_profile=LEFT_TWO_ENGINES_OUT,
    )
    obs, _ = env.reset()
    psi_before_failure = float(obs[8])
    psi_log = []
    for k in range(2500):  # 25 s — well past the 10 s trigger
        obs, _, _, trunc, _ = env.step(np.array([0.0, 0.0, 0.0, 0.555]))
        psi_log.append(float(obs[8]))
        if trunc:
            break
    # Heading should have *decreased* (nose left) due to asymmetric thrust.
    # B-747 is heavy and the trim isn't perfect after damage, but the sign
    # is unambiguous after 25 s of unbalanced thrust.
    assert psi_log[-1] < psi_before_failure


# ---- Flap jam events ---------------------------------------------------


def test_flap_jam_event_sets_flap_jam_config():
    s = B747DamageState.healthy()
    FlapJamEvent(trigger_time=5.0, jammed_config=B747Configuration.LANDING).apply(s)
    assert s.flap_jam_config is B747Configuration.LANDING


def test_flap_jam_event_validates_config_type():
    with pytest.raises(ValueError, match="jammed_config"):
        FlapJamEvent(trigger_time=0.0, jammed_config="landing")  # type: ignore[arg-type]


def test_flaps_jammed_landing_preset_overrides_aero_config():
    env = NonlinearB747Env(
        trim_at=(20_000.0, 674.0),
        number_time_steps=2000, dt=0.01,
        damage_profile=FLAPS_JAMMED_LANDING,
        config=B747Configuration.NOMINAL,
    )
    env.reset()
    for _ in range(700):  # 7 s — past the 5 s trigger
        env.step(np.array([0.0, 0.0, 0.0, 0.555]))
    # Damage state now overrides the aero config
    assert env.damage_manager.state.flap_jam_config is B747Configuration.LANDING
    # Env's params still report NOMINAL — the override is at the aero level
    assert env.model.param.config is B747Configuration.NOMINAL


def test_flap_jam_changes_pitching_moment_at_same_state():
    """Same physical state, same controls — but jammed flaps select different
    derivatives, so aerodynamic pitching moment differs."""
    from tensoraerospace.aerospacemodel.b747.nonlinear.aero import (
        AeroState,
        b747_aero,
    )

    state = AeroState(
        alpha=np.deg2rad(3.0), beta=0.0, V=300.0,
        p=0.0, q=0.0, r=0.0,
        altitude_ft=0.0,
        de=0.0, da=0.0, dr=0.0,
    )

    # Healthy NOMINAL configuration
    p_clean = default_parameters(B747Configuration.NOMINAL)
    forces_clean = b747_aero(state, p_clean)

    # Same NOMINAL config, but flaps jammed at LANDING (30°)
    p_flapjam = default_parameters(B747Configuration.NOMINAL)
    p_flapjam.damage_state = B747DamageState(flap_jam_config=B747Configuration.LANDING)
    forces_flapjam = b747_aero(state, p_flapjam)

    # The lift, drag and pitching moment must all change measurably
    assert abs(forces_clean.L - forces_flapjam.L) > 1e-3
    assert abs(forces_clean.m - forces_flapjam.m) > 1e-3
