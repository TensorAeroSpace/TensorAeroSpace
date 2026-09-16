"""Check Bellman targets after real collection of terminated/truncated episodes."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.agent.dqn import model as dqn
from tensoraerospace.agent.metrics.writer import MetricWriter


class OneStepEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (1,), dtype=np.float32)

    def __init__(self, terminated, truncated, *, continuous=False):
        self.terminated = terminated
        self.truncated = truncated
        self.action_space = (
            gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            if continuous
            else gym.spaces.Discrete(2)
        )
        self.resets = 0
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        self.finished = False
        return np.array([1], dtype=np.float32), {}

    def step(self, action):
        assert not self.finished, "An ended episode must be reset before stepping"
        self.finished = True
        return np.array([2], dtype=np.float32), 1.0, self.terminated, self.truncated, {}


class LinearQ(nn.Linear):
    def __init__(self, slope):
        super().__init__(1, 2, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[0.0], [slope]]))

    def action_value(self, obs):
        with torch.no_grad():
            values = self(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()
        return int(values.argmax()), values


@pytest.mark.parametrize("agent_cls", [dqn.DQNAgent, dqn.PERNARXAgent])
@pytest.mark.parametrize(
    "terminated,truncated", [(False, True), (True, False), (True, True)]
)
def test_dqn_time_limit_preserves_future_return(
    monkeypatch, agent_cls, terminated, truncated
):
    monkeypatch.setattr(dqn, "_DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        dqn, "create_metric_writer", lambda **kwargs: MagicMock(spec=MetricWriter)
    )
    env = OneStepEnv(terminated, truncated)
    agent = agent_cls(
        LinearQ(1.0),
        LinearQ(2.0),
        env,
        gamma=0.5,
        epsilon=0.0,
        buffer_size=8,
        batch_size=2,
        train_nums=2,
    )
    try:
        agent.train()
        assert env.resets == 3
        assert agent.num_in_buffer == 2
        # Q(s,a)=1, r=1, Q_target(s_next,a_next)=4. Only termination
        # removes the bootstrap term: losses are (1-1)^2 or (1-3)^2.
        assert agent.train_step() == pytest.approx(0.0 if terminated else 4.0)
        for transition in agent.replay_buffer.transitions[:2]:
            assert transition[-1] is terminated
            np.testing.assert_array_equal(transition[3], [2.0])
    finally:
        agent.close()
        env.close()


@pytest.mark.parametrize(
    "terminated,truncated", [(False, True), (True, False), (True, True)]
)
def test_ddpg_time_limit_preserves_future_return(terminated, truncated):
    env = OneStepEnv(terminated, truncated, continuous=True)
    agent = DDPG(
        env,
        value_lr=1e-3,
        policy_lr=1e-4,
        replay_buffer_size=8,
        normalize_observations=False,
        device="cpu",
    )
    agent.writer = MagicMock(spec=MetricWriter)
    targets = []
    hook = agent.value_criterion.register_forward_hook(
        lambda module, args, output: targets.append(args[1].detach().clone())
    )
    try:
        agent.train(num_episodes=2, max_steps=1, batch_size=2, warmup_frames=10)
        assert env.resets == 2
        assert len(agent.replay_buffer) == 2
        with torch.no_grad():
            for parameter in agent.target_value_net.parameters():
                parameter.zero_()
            agent.target_value_net.linear3.bias.fill_(4.0)
        agent.ddpg_update(batch_size=2, gamma=0.5)
        # Reward 1 plus discounted future return 0.5 * 4 at a time limit.
        torch.testing.assert_close(
            targets[0], torch.full((2, 1), 1.0 if terminated else 3.0)
        )
        assert all(
            transition[-1] is terminated for transition in agent.replay_buffer.buffer
        )
    finally:
        hook.remove()
        agent.writer.close()
        env.close()
