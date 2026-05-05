"""Tests for the rotor-level damage subsystem and the env integration."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.quadrotor.damage import (
    LANZON_M1_LOSS,
    LU_M1_50PCT_LOSS,
    WEAR_DEGRADATION_M3,
    DamageProfile,
    MotorEfficiencyDecay,
    RotorDamageEvent,
    RotorDamageManager,
    RotorDamageState,
    RotorLossEvent,
)

# ---- RotorDamageState ------------------------------------------------------


def test_healthy_state_has_full_effectiveness():
    s = RotorDamageState.healthy()
    assert np.allclose(s.mu, 1.0)
    assert np.allclose(s.tau, 0.0)


def test_state_validation_rejects_out_of_bounds_mu():
    with pytest.raises(ValueError, match=r"mu must be in \[0, 1\]"):
        RotorDamageState(mu=np.array([0.5, 0.5, 0.5, 1.5]))


def test_state_step_decay_does_nothing_when_tau_is_zero():
    s = RotorDamageState.healthy()
    s.step_decay(0.01)
    assert np.allclose(s.mu, 1.0)


def test_state_step_decay_pushes_mu_towards_floor():
    s = RotorDamageState.healthy()
    s.tau[2] = 1.0
    s.mu_floor[2] = 0.5
    # 100 steps of dt=0.01 → t = 1 s = 1·tau → mu should be close to 1/e
    # of the initial gap above the floor: 0.5 + 0.5/e ≈ 0.5 + 0.184
    for _ in range(100):
        s.step_decay(0.01)
    expected = 0.5 + 0.5 * np.exp(-1.0)
    assert abs(s.mu[2] - expected) < 0.01


# ---- Events ---------------------------------------------------------------


def test_rotor_damage_event_sets_mu_to_supplied_value():
    s = RotorDamageState.healthy()
    RotorDamageEvent(trigger_time=1.0, rotor_id=2, mu=0.4).apply(s)
    assert np.isclose(s.mu[2], 0.4)
    # Other rotors unchanged
    assert np.allclose(s.mu[[0, 1, 3]], 1.0)


def test_rotor_loss_event_zeros_mu():
    s = RotorDamageState.healthy()
    s.tau[0] = 5.0  # pre-existing decay
    RotorLossEvent(trigger_time=1.0, rotor_id=0).apply(s)
    assert s.mu[0] == 0.0
    # Cancels active decay on the lost rotor
    assert s.tau[0] == 0.0


def test_motor_efficiency_decay_arms_decay():
    s = RotorDamageState.healthy()
    MotorEfficiencyDecay(trigger_time=2.0, rotor_id=1, tau=4.0, mu_floor=0.2).apply(s)
    assert s.tau[1] == 4.0
    assert s.mu_floor[1] == 0.2
    # mu itself is still 1.0 right after the event triggers
    assert s.mu[1] == 1.0


def test_invalid_rotor_id_raises():
    with pytest.raises(ValueError, match="rotor_id"):
        RotorLossEvent(trigger_time=1.0, rotor_id=4)


def test_invalid_mu_raises():
    with pytest.raises(ValueError, match=r"mu must be in \[0, 1\]"):
        RotorDamageEvent(trigger_time=1.0, rotor_id=0, mu=1.5)


def test_negative_trigger_time_raises():
    with pytest.raises(ValueError, match="trigger_time"):
        RotorLossEvent(trigger_time=-1.0, rotor_id=0)


# ---- DamageProfile + Manager ---------------------------------------------


def test_damage_profile_pending_events_window():
    profile = DamageProfile(
        events=[
            RotorLossEvent(trigger_time=1.0, rotor_id=0),
            RotorLossEvent(trigger_time=5.0, rotor_id=1),
        ]
    )
    assert len(profile.get_pending_events(0.5, 0.0)) == 0
    assert len(profile.get_pending_events(1.5, 0.5)) == 1
    assert len(profile.get_pending_events(6.0, 0.5)) == 2


def test_manager_applies_events_in_window():
    mgr = RotorDamageManager(
        DamageProfile(
            events=[
                RotorDamageEvent(trigger_time=1.0, rotor_id=0, mu=0.7),
            ]
        )
    )
    fired = mgr.update(t_current=1.5, t_previous=0.0, dt=0.01)
    assert len(fired) == 1
    assert mgr.state.mu[0] == 0.7
    # Subsequent windows past trigger_time do nothing
    fired2 = mgr.update(t_current=2.0, t_previous=1.5, dt=0.01)
    assert len(fired2) == 0


def test_manager_inject_event_is_single_fire():
    mgr = RotorDamageManager()
    mgr.inject_event(RotorLossEvent(trigger_time=0.5, rotor_id=2))
    fired = mgr.update(t_current=1.0, t_previous=0.0, dt=0.01)
    assert len(fired) == 1
    fired2 = mgr.update(t_current=2.0, t_previous=1.0, dt=0.01)
    assert len(fired2) == 0


def test_manager_reset_clears_state_and_injected():
    mgr = RotorDamageManager(
        DamageProfile(events=[RotorLossEvent(trigger_time=0.5, rotor_id=0)])
    )
    mgr.inject_event(RotorLossEvent(trigger_time=0.6, rotor_id=1))
    mgr.update(t_current=1.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu[0] == 0.0
    assert mgr.state.mu[1] == 0.0

    mgr.reset()
    assert np.allclose(mgr.state.mu, 1.0)
    # The base profile remains; the injected one-shot is gone
    fired = mgr.update(t_current=1.0, t_previous=0.0, dt=0.01)
    assert len(fired) == 1  # only the profile event re-fires


# ---- Presets --------------------------------------------------------------


def test_lanzon_preset_is_complete_motor1_loss():
    mgr = RotorDamageManager(LANZON_M1_LOSS)
    mgr.update(t_current=6.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu[0] == 0.0


def test_lu_preset_is_50pct_motor1_loss():
    mgr = RotorDamageManager(LU_M1_50PCT_LOSS)
    mgr.update(t_current=6.0, t_previous=0.0, dt=0.01)
    assert mgr.state.mu[0] == 0.5


def test_wear_preset_decays_motor3_over_time():
    mgr = RotorDamageManager(WEAR_DEGRADATION_M3)
    # Step from t=0 to t=20s in 0.01 chunks (2000 steps)
    mu_history = []
    for k in range(2000):
        mgr.update(t_current=(k + 1) * 0.01, t_previous=k * 0.01, dt=0.01)
        mu_history.append(mgr.state.mu[2])
    # By t=20s the rotor should be close (not equal) to mu_floor=0.3
    assert mu_history[-1] < 0.4
    # Monotonic decay
    deltas = np.diff(mu_history)
    assert np.all(deltas <= 1e-9)  # tiny floating-point slack


# ---- Env integration ------------------------------------------------------


def test_env_virtual_mode_no_damage_matches_bare_model():
    """Env in 'virtual' mode without damage = passthrough to the ODE."""
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    env = NonlinearQuadrotorEnv(
        initial_state=np.zeros(12),
        number_time_steps=100,
        action_space="virtual",
    )
    obs, _ = env.reset()
    T_hover = env.model.hover_thrust  # type: ignore[union-attr]
    for _ in range(100):
        obs, r, term, trunc, info = env.step(np.array([T_hover, 0.0, 0.0, 0.0]))
        if trunc:
            break
    assert np.max(np.abs(obs)) < 1e-6  # exact hover


def test_env_rotor_mode_hover_is_4_equal_omega2():
    """In 'rotor' mode, hover requires all 4 ω² equal to T_hover/(4·k_T)."""
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    env = NonlinearQuadrotorEnv(
        initial_state=np.zeros(12),
        number_time_steps=100,
        action_space="rotor",
    )
    obs, _ = env.reset()
    T_hover = 1.5 * 9.81
    omega2_hover = T_hover / (4 * env.allocator.k_T)
    action = np.full(4, omega2_hover)

    for _ in range(100):
        obs, r, term, trunc, info = env.step(action)
        if trunc:
            break
    # Allow tiny numerical drift
    assert np.max(np.abs(obs)) < 1e-3


def test_env_with_lanzon_preset_destabilises_after_trigger():
    """With M1 lost at t=5s, hover thrust no longer holds → drone falls."""
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    env = NonlinearQuadrotorEnv(
        initial_state=np.zeros(12),
        number_time_steps=1000,
        action_space="virtual",
        damage_profile=LANZON_M1_LOSS,
    )
    obs, _ = env.reset()
    T_hover = env.model.hover_thrust  # type: ignore[union-attr]
    triggered_observed = False
    for k in range(1000):
        obs, r, term, trunc, info = env.step(np.array([T_hover, 0.0, 0.0, 0.0]))
        if "damage_events_triggered" in info:
            triggered_observed = True
        if trunc:
            break
    assert triggered_observed, "Lanzon event should fire at t=5s"
    # After M1 loss, body should NOT remain at zero (some channel disturbed)
    assert np.max(np.abs(obs)) > 0.05


def test_env_invalid_action_size_raises():
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    env = NonlinearQuadrotorEnv(
        initial_state=np.zeros(12), number_time_steps=10, action_space="virtual"
    )
    env.reset()
    with pytest.raises(ValueError, match="must have 4"):
        env.step(np.zeros(3))


def test_env_damage_profile_via_reset_options():
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    env = NonlinearQuadrotorEnv(
        initial_state=np.zeros(12),
        number_time_steps=1000,
        action_space="virtual",
    )
    env.reset(options={"damage_profile": LU_M1_50PCT_LOSS})
    assert env.damage_manager is not None
    T_hover = env.model.hover_thrust  # type: ignore[union-attr]
    for _ in range(1000):
        obs, r, term, trunc, info = env.step(np.array([T_hover, 0.0, 0.0, 0.0]))
        if trunc:
            break
    assert env.damage_manager.state.mu[0] == 0.5


def test_env_invalid_action_space_mode_raises():
    from tensoraerospace.envs.quadrotor import NonlinearQuadrotorEnv

    with pytest.raises(ValueError, match='action_space must be "virtual" or "rotor"'):
        NonlinearQuadrotorEnv(
            initial_state=np.zeros(12),
            number_time_steps=10,
            action_space="thrust_only",  # type: ignore[arg-type]
        )
