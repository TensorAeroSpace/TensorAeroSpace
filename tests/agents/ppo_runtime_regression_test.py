"""Regressions for PPO initialization and resource cleanup."""

from unittest.mock import MagicMock

import gymnasium as gym
import pytest
import torch

from tensoraerospace.agent.ppo import model as ppo_mod


@pytest.fixture
def make_agent(monkeypatch):
    agents = []
    monkeypatch.setattr(ppo_mod, "create_metric_writer", lambda **kwargs: MagicMock())

    def make(**kwargs):
        agent = ppo_mod.PPO(
            gym.make("Pendulum-v1"),
            device="cpu",
            actor_hidden_dim=8,
            critic_hidden_dim=8,
            **kwargs,
        )
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent._best_saver = None
        agent.close()
        agent.env.close()


@pytest.mark.parametrize("auxiliary_coef", [0.0, 0.2])
def test_seed_reproduces_actor_and_critic_initialization(make_agent, auxiliary_coef):
    torch.manual_seed(123)
    first = make_agent(seed=17, auxiliary_coef=auxiliary_coef)
    torch.rand(31)
    second = make_agent(seed=17, auxiliary_coef=auxiliary_coef)

    for name in ("actor", "critic"):
        expected = getattr(first, name).state_dict()
        actual = getattr(second, name).state_dict()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_distinct_seeds_produce_distinct_weights(make_agent):
    torch.manual_seed(123)
    first = make_agent(seed=17)
    torch.manual_seed(123)
    second = make_agent(seed=29)

    assert not torch.equal(first.actor.d1.weight, second.actor.d1.weight)
    assert not torch.equal(first.critic.d1.weight, second.critic.d1.weight)


@pytest.mark.parametrize("with_saver", [False, True])
def test_close_releases_metrics_with_or_without_checkpoint_saver(
    make_agent, with_saver
):
    agent = make_agent()
    saver = MagicMock() if with_saver else None
    agent._best_saver = saver

    agent.close()

    agent.writer.close.assert_called_once_with()
    if saver is not None:
        saver.flush.assert_called_once()
        saver.close.assert_called_once()
        assert agent._best_saver is None


def test_close_releases_metrics_when_checkpoint_flush_fails(make_agent):
    agent = make_agent()
    agent._best_saver = MagicMock()
    agent._best_saver.flush.side_effect = OSError("checkpoint unavailable")
    saver = agent._best_saver

    with pytest.raises(OSError, match="checkpoint unavailable"):
        agent.close()

    agent.writer.close.assert_called_once_with()
    saver.close.assert_called_once()
    assert agent._best_saver is None
