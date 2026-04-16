"""Unit tests for the Event-Triggered DHP agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensoraerospace.agent.et_dhp import (
    ETDHPActor,
    ETDHPAgent,
    ETDHPConfig,
    ETDHPCritic,
    EventTrigger,
    PlantModelNN,
)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def test_actor_output_within_bound():
    actor = ETDHPActor(n_state=3, n_control=2, u_bound=5.0)
    out, d_nn = actor(torch.randn(3))
    assert out.shape == (2,)
    assert d_nn.shape == (2,)
    assert torch.all(out.abs() <= 5.0 + 1e-6)


def test_actor_zero_state_gives_zero_control():
    """No-bias actor must produce u=0 at x=0 — critical for a regulator."""
    actor = ETDHPActor(n_state=3, n_control=2, u_bound=5.0)
    out, _ = actor(torch.zeros(3))
    assert torch.allclose(out, torch.zeros(2), atol=1e-6)


def test_critic_returns_costate_shape():
    critic = ETDHPCritic(n_state=4)
    lam = critic(torch.randn(4))
    assert lam.shape == (4,)


def test_plant_model_predicts_next_state():
    model = PlantModelNN(n_state=2, n_control=1)
    xu = torch.cat([torch.randn(2), torch.randn(1)])
    pred = model(xu)
    assert pred.shape == (2,)


# ---------------------------------------------------------------------------
# Event trigger
# ---------------------------------------------------------------------------
def test_event_trigger_rejects_out_of_range_rho():
    with pytest.raises(ValueError):
        EventTrigger(rho=0.5)
    with pytest.raises(ValueError):
        EventTrigger(rho=0.0)


def test_event_trigger_fires_on_first_call_by_default():
    et = EventTrigger(rho=0.1)
    assert et.should_trigger(np.array([0.1, 0.0]), step=0) is True
    assert et.num_triggers == 1


def test_event_trigger_holds_when_state_stays_close():
    et = EventTrigger(rho=0.1, min_floor=0.0)
    et.should_trigger(np.array([1.0, 0.0]), step=0)  # initial trigger
    # A nearly-identical state right after should not re-trigger.
    assert et.should_trigger(np.array([1.0001, 0.0]), step=1) is False


def test_event_trigger_fires_when_state_deviates_enough():
    et = EventTrigger(rho=0.1, min_floor=0.0)
    et.should_trigger(np.array([1.0, 0.0]), step=0)
    # Force ||x − x_et|| way above the threshold → must retrigger.
    assert et.should_trigger(np.array([5.0, 0.0]), step=5) is True
    assert et.num_triggers == 2


def test_event_trigger_reset_re_arms():
    et = EventTrigger(rho=0.1)
    et.should_trigger(np.array([1.0]), step=0)
    et.should_trigger(np.array([1.0]), step=1)
    et.reset()
    assert et.num_triggers == 0
    assert et.state_at_trigger is None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
def _make_agent(**cfg_kwargs) -> ETDHPAgent:
    cfg = ETDHPConfig(
        actor_hidden=(8, 8),
        critic_hidden=(8, 8),
        model_hidden=(8, 8),
        Q=[1.0, 0.5],
        R=[1.0],
        gamma=0.95,
        num_epochs_per_trigger=2,
        u_bound=5.0,
        rho=0.1,
        trigger_floor=0.0,
        model_epochs=5,
        seed=0,
        **cfg_kwargs,
    )
    return ETDHPAgent(n_state=2, n_control=1, config=cfg)


def test_agent_config_validation():
    with pytest.raises(ValueError, match="Q has length"):
        ETDHPAgent(n_state=3, n_control=1, config=ETDHPConfig(Q=[1.0], R=[1.0]))
    with pytest.raises(ValueError, match="R has length"):
        ETDHPAgent(
            n_state=2,
            n_control=2,
            config=ETDHPConfig(Q=[1.0, 1.0], R=[1.0]),
        )


def test_agent_predict_returns_bounded_action():
    agent = _make_agent()
    a = agent.predict(np.array([0.1, 0.0]), None, 0)
    assert a.shape == (1,)
    assert abs(a[0]) <= 5.0 + 1e-6


def test_agent_fit_plant_model_converges_on_linear_system():
    """Offline plant-model fit should get well below the initial MSE
    on a trivial linear system."""
    agent = _make_agent()
    rng = np.random.default_rng(0)
    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    B = np.array([[0.05], [0.1]])
    N = 400
    x = rng.normal(0, 0.1, size=(N, 2))
    u = rng.normal(0, 1.0, size=(N, 1))
    x_next = x @ A.T + u @ B.T
    losses = agent.fit_plant_model(x, u, x_next, epochs=80, batch_size=64)
    assert losses[-1] < 0.5 * losses[0]


def test_agent_predict_learn_cycle_updates_only_on_trigger():
    """After the initial trigger, tiny deviations must not re-fire."""
    agent = _make_agent()
    obs = np.array([0.1, 0.0])
    agent.reset()
    # First predict+learn should trigger once.
    agent.predict(obs, None, 0)
    m1 = agent.learn(obs + 1e-6, None, 0, dt=0.01)
    assert m1["triggered"] == 1.0
    # Repeat with near-identical obs → should hold (no trigger).
    agent.predict(obs + 1e-6, None, 1)
    m2 = agent.learn(obs + 2e-6, None, 1, dt=0.01)
    assert m2["triggered"] == 0.0


def test_agent_last_action_tracks_predict_between_triggers():
    agent = _make_agent()
    obs = np.array([0.1, 0.0])
    agent.reset()
    a = agent.predict(obs, None, 0)
    agent.learn(obs + 1e-6, None, 0, dt=0.01)
    assert agent.last_action().shape == (1,)
    # After a non-trigger step, last_action should be unchanged relative to
    # the most recent trigger (no new update).
    kept = agent.last_action().copy()
    agent.predict(obs + 1e-6, None, 1)
    agent.learn(obs + 2e-6, None, 1, dt=0.01)
    assert np.allclose(agent.last_action(), kept)


def test_agent_reset_rearms_trigger():
    agent = _make_agent()
    agent.predict(np.array([0.1, 0.0]), None, 0)
    agent.learn(np.array([0.1, 0.0]), None, 0, dt=0.01)
    assert agent.event_trigger.num_triggers == 1
    agent.reset()
    assert agent.event_trigger.num_triggers == 0
