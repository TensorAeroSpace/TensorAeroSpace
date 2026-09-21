"""Damage must change the F-16 only from the event's actual timestamp."""

import json

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageEvent,
    DamageManager,
    DamageProfile,
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    ControlFailure,
    DamageState,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.model import (
    LongitudinalF16,
)
from tensoraerospace.agent.sac import SAC
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
from tensoraerospace.envs.f16.nonlinear_longitudinal import NonlinearLongitudinalF16


def make_env(kind, profile=None, **kwargs):
    common = dict(
        number_time_steps=20,
        dt=0.02,
        integrator="rk4",
        damage_profile=profile,
        **kwargs,
    )
    if kind == "angular":
        return NonlinearAngularF16(
            np.zeros(14), track_altitude=True, thrust_mode="control", **common
        )
    return NonlinearLongitudinalF16(np.zeros(2), np.zeros((1, 30)), **common)


def action_for(kind):
    return np.array([0.0, 0.0, 0.0, 20000.0]) if kind == "angular" else np.array([0.0])


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_end_boundary_does_not_affect_preceding_interval(kind):
    event = DamageEvent(
        0.02, "section_loss", {"section": "left_tip", "loss_fraction": 0.5}
    )
    healthy = make_env(kind, DamageProfile([]))
    damaged = make_env(kind, DamageProfile([event]))
    healthy.reset()
    damaged.reset()
    healthy.step(action_for(kind))
    damaged.step(action_for(kind))
    np.testing.assert_array_equal(
        damaged.model.current_state, healthy.model.current_state
    )
    assert damaged.damage_manager.state.section_loss["left_tip"] == 0.5
    assert len(damaged.model.x_history) == 2
    assert damaged.damage_events_log[0]["time"] == 0.02


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_start_events_are_applied_before_reset_observation(kind):
    profile = DamageProfile([DamageEvent(0, "engine_failure", {"thrust_factor": 0.3})])
    env = make_env(kind, profile, damage_observable=True)
    obs, _ = env.reset()
    assert obs[-1] == pytest.approx(0.3)
    assert env.damage_events_log[0]["time"] == 0
    assert env.damage_state_log[-1]["time"] == 0
    env.step(action_for(kind))
    assert len(env.damage_events_log) == 1
    env.reset()
    assert len(env.damage_events_log) == 1


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_reset_options_enable_damage_without_constructor_profile(kind):
    profile = DamageProfile(
        [DamageEvent(0.007, "engine_failure", {"thrust_factor": 0.3})]
    )
    env = make_env(kind)
    env.reset(options={"damage_profile": profile})
    env.step(action_for(kind))
    assert env.damage_manager.state.engine.thrust_factor == 0.3
    assert env.damage_events_log[0]["time"] == 0.007
    env.reset()
    assert env.damage_manager is None


def test_manager_merges_profile_and_injected_events_in_time_order_once():
    late = DamageEvent(0.018, "engine_failure", {"thrust_factor": 0.8})
    early = DamageEvent(0.005, "engine_failure", {"thrust_factor": 0.2})
    middle = DamageEvent(0.012, "engine_failure", {"thrust_factor": 0.5})
    manager = DamageManager(
        load_f16_geometry(), default_parameters(), DamageProfile([late, early])
    )
    manager.inject_event(middle)
    assert manager.update(0.02, 0) == [early, middle, late]
    assert manager.state.engine.thrust_factor == 0.8
    assert manager.update(0.02, 0) == []
    manager.reset()
    assert manager.update(0.02, 0) == [early, late]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_event_time_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        DamageEvent(bad, "engine_failure", {})


@pytest.mark.parametrize("mode", ["jam", "lost", "efficiency_loss"])
def test_longitudinal_model_applies_stabilator_failure(mode):
    model = LongitudinalF16(np.zeros(4), dt=0.01, integrator="rk4")
    model.damage_state = DamageState.healthy(load_f16_geometry())
    model.damage_state.set_control_failure(
        "stab_left", ControlFailure(mode=mode, jam_position_rad=0, efficiency=0)
    )
    state = model.run_step([0.1]).reshape(-1)
    np.testing.assert_array_equal(state[2:], [0, 0])
    np.testing.assert_array_equal(model.u_history[-1].reshape(-1), [0])


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_sac_checkpoint_roundtrip_with_damage_profile(tmp_path, kind):
    profile = DamageProfile(
        [DamageEvent(0.007, "engine_failure", {"thrust_factor": 0.3})], seed=17
    )
    env = make_env(kind, profile, damage_observable=True)
    obs, _ = env.reset()
    agent = SAC(env, hidden_size=8, log_dir=tmp_path / "log", seed=5)
    try:
        expected = agent.select_action(obs, evaluate=True)
        agent.save(tmp_path / "saved")
        folder = next((tmp_path / "saved").iterdir())
        config = json.loads((folder / "config.json").read_text())
        assert (
            config["env"]["params"]["damage_profile"]["events"][0]["trigger_time"]
            == 0.007
        )
        restored = SAC.from_pretrained(str(folder))
        try:
            restored_obs, _ = restored.env.reset()
            np.testing.assert_array_equal(restored_obs, obs)
            np.testing.assert_array_equal(
                restored.select_action(restored_obs, evaluate=True), expected
            )
            restored.env.step(action_for(kind))
            assert restored.env.damage_events_log[0]["time"] == 0.007
        finally:
            restored.close()
    finally:
        agent.close()


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_midstep_event_matches_two_explicit_integration_intervals(kind):
    event = DamageEvent(
        0.007, "section_loss", {"section": "left_tip", "loss_fraction": 0.5}
    )
    damaged = make_env(kind, DamageProfile([event]))
    reference = make_env(kind, DamageProfile([]))
    damaged.reset()
    reference.reset()
    command = action_for(kind)
    model_command = command.copy()
    if kind == "angular":
        model_command[:-1] = np.deg2rad(model_command[:-1])
    else:
        model_command = np.deg2rad(model_command)
    reference.model.dt = 0.007
    reference.model.run_step(model_command)
    reference.damage_manager.inject_event(event)
    reference.damage_manager.update(0.007, 0)
    reference.model.dt = 0.013
    reference.model.run_step(model_command)
    damaged.step(command)
    np.testing.assert_allclose(
        damaged.model.current_state, reference.model.current_state, rtol=0, atol=1e-12
    )
    assert len(damaged.model.x_history) == 2
    assert len(damaged.model.u_history) == 1
    assert damaged.model.time_step == 2
    assert damaged.damage_state_log[-1]["time"] == 0.007


@pytest.mark.parametrize("command", [[np.nan], [np.inf], [1, 2]])
def test_longitudinal_bad_action_does_not_advance_time_or_damage(command):
    env = make_env(
        "longitudinal",
        DamageProfile([DamageEvent(0.007, "engine_failure", {"thrust_factor": 0})]),
    )
    obs, _ = env.reset()
    with pytest.raises(ValueError, match="finite"):
        env.step(command)
    assert env.current_step == 0
    assert len(env.model.x_history) == 1
    assert env.damage_manager.state.engine.thrust_factor == 1


def test_longitudinal_horizon_keeps_sac_bootstrap_enabled(tmp_path):
    from unittest.mock import MagicMock

    from tensoraerospace.agent.metrics.writer import MetricWriter

    env = make_env("longitudinal")
    agent = SAC(env, batch_size=64, hidden_size=8, log_dir=tmp_path)
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    agent.train(verbose=False)
    assert agent.memory.buffer[-1][4] == 0
    assert agent.writer.log_episode.call_args.kwargs["truncated"] is True
    assert agent.writer.log_episode.call_args.kwargs["terminated"] is False


@pytest.mark.parametrize("kind", ["angular", "longitudinal"])
def test_callbacks_are_not_silently_dropped_by_serialization(kind):
    env = make_env(kind, damage_event_callback=lambda *args: None)
    with pytest.raises(ValueError, match="callback"):
        env.get_init_args()


@pytest.mark.parametrize("offset", [-1, np.inf, np.nan, 0.03])
def test_invalid_event_offset_cannot_change_model_state(offset):
    model = LongitudinalF16(np.zeros(4), dt=0.02)
    with pytest.raises(ValueError, match="offset"):
        model.run_step([0], events=[(offset, lambda: None)])
    assert len(model.x_history) == 1


@pytest.mark.parametrize("factor,hard", [(0.4, False), (1.0, True)])
def test_engine_failure_changes_all_translational_force_projections(factor, hard):
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import EngineState

    params = default_parameters()
    params.T_active = thrust = 20000.0
    state = np.zeros(16)
    state[0], state[1] = alpha, beta = 0.1, 0.05
    state[14:] = [3000, 120]
    healthy = f16_ode_6dof(state, np.zeros(3), 0, params)
    damage = DamageState.healthy(load_f16_geometry())
    damage.engine = EngineState(thrust_factor=factor, hard_failure=hard)
    params.damage_state = damage
    damaged = f16_ode_6dof(state, np.zeros(3), 0, params)
    delta_thrust = thrust * ((0 if hard else factor) - 1)
    expected = np.array(
        [
            -np.sin(alpha) * delta_thrust / (params.m * state[15] * np.cos(beta)),
            -np.cos(alpha) * np.sin(beta) * delta_thrust / (params.m * state[15]),
            np.cos(alpha) * np.cos(beta) * delta_thrust / params.m,
        ]
    )
    np.testing.assert_allclose(
        (damaged - healthy)[[0, 1, 15]], expected, rtol=0, atol=1e-14
    )
