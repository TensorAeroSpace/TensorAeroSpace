"""Evaluation must leave the training RNG and deterministic-policy noise alone."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.metrics.writer import MetricWriter
from tensoraerospace.agent.sac import SAC


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (2,), np.float32)
    action_space = gym.spaces.Box(-2, 3, (1,), np.float32)

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.t += 1
        return (
            np.array([self.t / 10, action[0]], np.float32),
            -float(action[0] ** 2),
            False,
            self.t == 8,
            {},
        )


def make_agent(tmp_path, policy):
    agent = SAC(
        TinyEnv(),
        hidden_size=8,
        batch_size=4,
        memory_capacity=100,
        seed=11,
        policy_type=policy,
        log_dir=tmp_path,
    )
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    return agent


@pytest.mark.parametrize("policy", ["Gaussian", "Deterministic"])
@pytest.mark.parametrize("batch", [False, True])
def test_evaluation_is_rng_neutral_and_preserves_action(tmp_path, policy, batch):
    agent = make_agent(tmp_path, policy)
    obs = torch.tensor([[0.1, -0.2], [0.3, 0.5]])
    with torch.no_grad():
        expected = agent.policy.sample(obs if batch else obs[:1])[2]
    before = torch.get_rng_state().clone()
    noise_before = getattr(agent.policy, "noise", torch.empty(0)).clone()
    if batch:
        actual = agent.select_action_batch(obs, evaluate=True, return_tensor=True)
    else:
        actual = torch.from_numpy(
            agent.select_action(obs[0].numpy(), evaluate=True)
        ).unsqueeze(0)
        expected = expected[:1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(
        getattr(agent.policy, "noise", torch.empty(0)), noise_before, rtol=0, atol=0
    )
    assert not actual.requires_grad


@pytest.mark.parametrize("policy", ["Gaussian", "Deterministic"])
def test_evaluation_between_episodes_does_not_change_training(tmp_path, policy):
    agents = []
    for evaluate in [False, True]:
        agent = make_agent(tmp_path / str(evaluate), policy)
        for _ in range(3):
            agent.train(verbose=False)
            if evaluate:
                for _ in range(5):
                    agent.select_action(np.array([0.3, 0.5]), evaluate=True)
        agents.append(agent)
    for name in ["policy", "critic", "critic_target"]:
        for a, b in zip(
            getattr(agents[0], name).parameters(), getattr(agents[1], name).parameters()
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
