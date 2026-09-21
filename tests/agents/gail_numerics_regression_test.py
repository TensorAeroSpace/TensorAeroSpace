"""Adversarial losses must recover from confidently wrong classifications."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.gail.model import GAIL
from tests.agents.gail_rollout_regression_test import CountingEnv


def agent_with_logit(logit):
    agent = GAIL(
        CountingEnv(), 1e-3, 2, 2, 1, np.zeros((4, 2), dtype=np.float32), device="cpu"
    )
    with torch.no_grad():
        for p in agent.discriminator.parameters():
            p.zero_()
        agent.discriminator.linear3.bias.fill_(logit)
    return agent


@pytest.mark.parametrize(
    "logit,target,gradient", [(100.0, 0.0, 1.0), (-100.0, 1.0, -1.0)]
)
def test_discriminator_recovers_from_saturation(logit, target, gradient):
    agent = agent_with_logit(logit)
    prediction = agent.discriminator.logits(torch.zeros((1, 2)))
    loss = agent.discrim_criterion(prediction, torch.tensor([[target]]))
    loss.backward()
    assert loss.item() == pytest.approx(100.0)
    assert agent.discriminator.linear3.bias.grad.item() == pytest.approx(gradient)


def test_imitation_reward_does_not_flatten_at_underflow_limit():
    agent = agent_with_logit(-1000.0)
    reward = agent.expert_reward(torch.zeros((1, 1)), np.zeros((1, 1)))
    assert reward.item() == pytest.approx(1000.0)


def test_probability_api_is_preserved():
    agent = agent_with_logit(2.0)
    probability = agent.discriminator(torch.zeros((1, 2)))
    assert probability.item() == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_policy_returns_use_the_updated_discriminator(monkeypatch):
    from tensoraerospace.agent.gail import model as module

    agent = agent_with_logit(0.0)
    agent.env.terminal = True
    agent.env.horizon = 1

    def update_discriminator():
        with torch.no_grad():
            agent.discriminator.linear3.bias.fill_(-2.0)

    agent.optimizer_discrim.step = update_discriminator
    agent.ppo_update = lambda *args, **kwargs: None
    captured = []
    original = module.compute_gae

    def capture(next_value, rewards, masks, values):
        returns = original(next_value, rewards, masks, values)
        captured.append(returns[0].item())
        return returns

    monkeypatch.setattr(module, "compute_gae", capture)
    agent.learn(1, float("inf"))
    assert captured == pytest.approx([np.log1p(np.exp(2.0))])
