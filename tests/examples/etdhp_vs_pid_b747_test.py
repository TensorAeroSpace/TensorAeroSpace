"""Causality, physical fidelity and active-learning checks for B747 comparison."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_vs_pid_b747 as demo,
)
from tensoraerospace.aerospacemodel.b747.nonlinear import default_parameters
from tensoraerospace.aerospacemodel.b747.nonlinear.damage.state import B747DamageState
from tensoraerospace.aerospacemodel.b747.nonlinear.dynamics import b747_ode_6dof
from tensoraerospace.aerospacemodel.b747.nonlinear.engine import (
    ENGINE_Y_POSITIONS_FT,
    jt9d_thrust_with_asymmetry,
)
from tensoraerospace.aerospacemodel.b747.nonlinear.params import isa_speed_of_sound_ft_s


def test_engine_event_cannot_change_the_trajectory_before_its_boundary():
    cfg = demo.Experiment(duration=1, fault_time=0.2)
    healthy = demo.make_env(cfg, fault=False)
    faulty = demo.make_env(cfg, fault=True)
    a, _ = healthy.reset()
    b, _ = faulty.reset()
    tr = demo.nominal_trim()
    u = [tr.elevator_rad, 0, 0, tr.throttle]
    for _ in range(4):
        a, *_ = healthy.step(u)
        b, *_ = faulty.step(u)
        np.testing.assert_array_equal(a, b)
    assert faulty.damage_events_log == []
    a, *_ = healthy.step(u)
    b, *_ = faulty.step(u)
    assert not np.array_equal(a, b)
    assert faulty.damage_events_log == [
        {
            "time": 0.2,
            "engine_id": 1,
            "thrust_fraction": 0.5,
            "label": "EngineFailureEvent",
            "kind": "EngineFailureEvent",
        }
    ]
    assert b[5] < a[5]
    faulty.reset()
    assert faulty.damage_events_log == []
    assert faulty.model.damage_state.engines_mu[1] == 1


def test_engine_event_at_zero_and_invalid_off_grid_time():
    env = demo.make_env(demo.Experiment(duration=1, fault_time=0), fault=True)
    env.reset()
    assert env.model.damage_state.engines_mu[1] == 0.5
    with pytest.raises(ValueError, match="align"):
        demo.Experiment(duration=1, fault_time=0.023)


@pytest.mark.parametrize("engine_id,sign", [(1, -1), (4, 1)])
def test_failure_force_and_yaw_moment_obey_force_balance(engine_id, sign):
    tr = demo.nominal_trim()
    params = default_parameters()
    mach = demo.SPEED / isa_speed_of_sound_ft_s(demo.ALTITUDE)
    thrust, zero = jt9d_thrust_with_asymmetry(tr.throttle, mach, demo.ALTITUDE, params)
    assert zero == pytest.approx(0, abs=1e-8)
    params.damage_state = B747DamageState.healthy()
    params.damage_state.engines_mu[engine_id] = 0.5
    damaged, moment = jt9d_thrust_with_asymmetry(
        tr.throttle, mach, demo.ALTITUDE, params
    )
    assert damaged == pytest.approx(thrust * 0.875)
    assert moment == pytest.approx(ENGINE_Y_POSITIONS_FT[engine_id] * thrust / 8)
    assert np.sign(moment) == sign
    x = tr.to_state()
    u = np.array([tr.elevator_rad, 0, 0, tr.throttle])
    healthy_dx = b747_ode_6dof(x, u, 0, default_parameters())
    faulty_dx = b747_ode_6dof(x, u, 0, params)
    assert faulty_dx[0] - healthy_dx[0] == pytest.approx(-thrust / 8 / params.mass_slug)
    gamma = params.Ix * params.Iz - params.Ixz**2
    assert faulty_dx[3] - healthy_dx[3] == pytest.approx(params.Ixz * moment / gamma)
    assert faulty_dx[5] - healthy_dx[5] == pytest.approx(params.Ix * moment / gamma)


def test_local_transition_uses_the_nonlinear_model_and_measured_integrals():
    x = np.array([0.4, 0.1, -0.2, 0.3, 0.7, 0.2, -0.1])
    u = np.array([0.4, -0.3])
    dt = 0.05
    predicted = demo.nominal_transition(x, u, dt)
    cfg = demo.Experiment(
        duration=1, fault_time=0.5, initial_heading_deg=x[4], initial_roll_deg=x[3]
    )
    env = demo.make_env(cfg, fault=False)
    obs = env.initial_state.copy()
    obs[1] = demo.SPEED * np.sin(np.deg2rad(x[0]))
    obs[:3] *= demo.SPEED / np.linalg.norm(obs[:3])
    obs[[3, 5]] = np.deg2rad(x[1:3])
    env.initial_state = obs
    env.reset()
    tr = demo.nominal_trim()
    nxt, *_ = env.step(np.r_[tr.elevator_rad, np.deg2rad(u), tr.throttle])
    integral = x[5:] / demo.INTEGRAL_SCALE + dt * np.rad2deg(nxt[[6, 8]])
    np.testing.assert_allclose(
        demo.lateral_state(nxt, integral), predicted, atol=1e-12, rtol=0
    )


def test_model_initial_jacobian_matches_healthy_nonlinear_linearization():
    agent, info = demo.make_agent(online_model_fit=True)
    jac = (
        torch.autograd.functional.jacobian(agent.plant_model, torch.zeros(9))
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(
        jac, np.column_stack([info["A"], info["B"]]), atol=2e-7, rtol=2e-6
    )
    x = torch.zeros(7)
    torch.testing.assert_close(agent.actor(x)[0], torch.zeros(2))
    assert (
        agent.cfg.online_model_fit
        and min(agent.cfg.actor_lr, agent.cfg.critic_lr, agent.cfg.model_lr) > 0
    )


def test_diagnostic_fault_time_cannot_change_healthy_controller_commands():
    torch.set_num_threads(1)
    cfg = demo.Experiment(duration=1, fault_time=0.2)
    agent, _ = demo.make_agent(online_model_fit=True)
    first, _ = demo.rollout(cfg, "etdhp", fault=False, template=agent)
    second, _ = demo.rollout(
        replace(cfg, fault_time=0.7), "etdhp", fault=False, template=agent
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first[-1, 15:18] > 0)


def test_event_trigger_holds_applied_command_exactly_between_updates():
    torch.set_num_threads(1)
    agent, _ = demo.make_agent(rho=0.1, floor=0.05)
    cfg = demo.Experiment(duration=3, fault_time=1)
    trace, diag = demo.rollout(cfg, "etdhp", fault=True, template=agent)
    skipped = np.flatnonzero(trace[:-1, 14] == 0)
    assert len(skipped) > 0 and diag["triggers"] > 0
    np.testing.assert_array_equal(trace[skipped, 10:12], trace[skipped + 1, 10:12])
    assert np.all(np.abs(trace[:, 10:12]) <= demo.BOUND)


def test_refining_physics_preserves_healthy_open_loop():
    cfg = demo.Experiment(duration=1, fault_time=0.5)
    coarse = demo.make_env(cfg, fault=True)
    fine = demo.make_env(replace(cfg, substeps=2), fault=True)
    coarse.reset()
    fine.reset()
    tr = demo.nominal_trim()
    u = [tr.elevator_rad, 0.005, -0.008, tr.throttle]
    for _ in range(20):
        a, *_ = coarse.step(u)
        fine.step(u)
        b, *_ = fine.step(u)
    np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-7)


def test_primary_etdhp_keeps_learning_actor_and_critic_after_unknown_event():
    torch.set_num_threads(1)
    cfg = demo.Experiment(duration=3, fault_time=1)
    template, _ = demo.make_agent()
    original = [
        demo.parameter_vector(net)
        for net in (template.actor, template.critic, template.plant_model)
    ]
    trace, diagnostics = demo.rollout(cfg, "etdhp", fault=True, template=template)
    assert diagnostics["status"] == "complete"
    actor, critic, plant = diagnostics["parameter_change_after_event"]
    assert actor > 0 and critic > 0 and plant == 0
    assert diagnostics["triggers_after_event"] > 0
    for net, old in zip(
        (template.actor, template.critic, template.plant_model), original
    ):
        np.testing.assert_array_equal(demo.parameter_vector(net), old)
    assert trace.shape == (60, len(demo.TRACE_NAMES))
