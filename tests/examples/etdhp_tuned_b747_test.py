"""Contracts for healthy-only ET-DHP tuning and unchanged B747 physics."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from example.reinforcement_learning.incremental_adp import (
    example_etdhp_tuned_b747 as tune,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"actor_lr": 0},
        {"critic_lr": 0},
        {"epochs": 0},
        {"rho": 0.5},
        {"trigger_floor": -1},
        {"embedding_scale": 0},
        {"q": [1] * 6},
        {"r": [np.nan, 1]},
    ],
)
def test_tuning_requires_active_learning_and_valid_dimensions(changes):
    with pytest.raises(ValueError):
        tune.Tuning(**changes)


def test_initial_policy_is_nominal_lqr_inside_trainable_networks():
    agent, info = tune.make_agent(tune.TUNED)
    x = torch.zeros(7, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: agent.actor(value)[0], x
    )
    np.testing.assert_allclose(
        jacobian.detach().numpy(), -np.asarray(info["K_initial"]), rtol=2e-6, atol=1e-6
    )
    a, b = np.asarray(info["A"]), np.asarray(info["B"])
    assert np.max(np.abs(np.linalg.eigvals(a - b @ np.asarray(info["K_initial"])))) < 1
    assert all(p.requires_grad for p in agent.actor.parameters())
    assert all(p.requires_grad for p in agent.critic.parameters())
    assert agent.actor_opt.param_groups[0]["lr"] == tune.TUNED.actor_lr > 0
    assert agent.critic_opt.param_groups[0]["lr"] == tune.TUNED.critic_lr > 0


def test_substeps_preserve_physical_duration_and_unknown_fault_boundary():
    cfg = tune.base.Experiment(duration=1, fault_time=0.2, substeps=2)
    env = tune.base.make_env(cfg, fault=True)
    env.reset()
    wrapped = tune.PhysicsSubsteps(env, cfg.substeps)
    tr = tune.base.nominal_trim()
    action = [tr.elevator_rad, 0, 0, tr.throttle]
    for _ in range(4):
        wrapped.step(action)
    assert env._step_index == 8
    assert wrapped.damage_events_log == []
    wrapped.step(action)
    assert env._step_index == 10
    assert wrapped.damage_events_log[0]["time"] == 0.2


def test_failed_warmup_is_reported_instead_of_discarding_candidate(monkeypatch):
    def fail(self, ref):
        raise RuntimeError("flight-envelope guard")

    monkeypatch.setattr(tune.steps.TrialState, "advance", fail)
    trace, report = tune.run_case(
        tune.TUNED, warmup=1, response=1, step_time=0.2, fault_time=0.5
    )
    assert trace.shape == (1, len(tune.steps.TRACE_COLUMNS))
    assert report["status"] == "failed" and report["stage"] == "warmup"
    assert tune.tuning_score(report) == 1e6


def test_common_cost_rescaling_preserves_the_initial_policy():
    first, a = tune.make_agent(tune.TUNED)
    second, b = tune.make_agent(
        replace(tune.TUNED, cost_scale=10 * tune.TUNED.cost_scale)
    )
    np.testing.assert_allclose(a["K_initial"], b["K_initial"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(a["P_initial"]) * 10, b["P_initial"], rtol=1e-8, atol=1e-8
    )
    for x in [np.zeros(7), np.array([0.2, 0.1, -0.3, 0.5, 1.0, 0.02, -0.1])]:
        np.testing.assert_allclose(
            first.predict(x), second.predict(x), rtol=1e-6, atol=1e-6
        )
        first.reset()
        second.reset()
