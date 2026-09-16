"""Training budgets and discrete action indices must survive actual collection."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.agent.dqn import model as dqn
from tensoraerospace.agent.metrics import schema
from tensoraerospace.agent.metrics.writer import MetricWriter


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)

    def __init__(self, *, continuous=False, n_actions=2, episode_length=1):
        self.action_space = (
            gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
            if continuous
            else gym.spaces.Discrete(n_actions)
        )
        self.episode_length = episode_length
        self.total_steps = 0
        self.steps = 0
        self.resets = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        return np.array([1], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        return (
            np.array([2], dtype=np.float32),
            1.0,
            self.steps >= self.episode_length,
            False,
            {},
        )


class ConstantQ(nn.Linear):
    def __init__(self, n_actions):
        super().__init__(1, n_actions)
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()

    def action_value(self, obs):
        with torch.no_grad():
            values = self(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()
        return int(values.argmax()), values


@pytest.fixture(params=[dqn.DQNAgent, dqn.PERNARXAgent])
def make_dqn(request, monkeypatch):
    monkeypatch.setattr(dqn, "_DEVICE", torch.device("cpu"))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    agents = []

    def make(env=None, **kwargs):
        env = env or CountingEnv()
        options = dict(epsilon=0.0, buffer_size=8, batch_size=2, train_nums=5)
        options.update(kwargs)
        agent = request.param(
            ConstantQ(env.action_space.n), ConstantQ(env.action_space.n), env, **options
        )
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()
        agent.env.close()


@pytest.mark.parametrize("action", [128, 255, 256])
def test_dqn_large_action_index_updates_the_selected_output(make_dqn, action):
    agent = make_dqn(CountingEnv(n_actions=300))
    for _ in range(2):
        agent.store_transition(1.0, np.array([1.0]), action, 1.0, np.array([2.0]), True)
    agent.num_in_buffer = 2
    assert agent.train_step() == pytest.approx(1.0)
    np.testing.assert_array_equal(agent.b_actions, [action, action])
    assert agent.model.bias[action].item() > 0.0
    unchanged = agent.model.bias.detach().clone()
    unchanged[action] = 0.0
    torch.testing.assert_close(unchanged, torch.zeros_like(unchanged))


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, 5),
        ({"max_steps": 1}, 1),
        ({"max_steps": 3}, 3),
        ({"num_episodes": 2, "max_steps": 3}, 6),
    ],
)
def test_dqn_collects_exact_requested_number_of_steps(make_dqn, kwargs, expected):
    agent = make_dqn()
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    agent.train(**kwargs)
    assert agent.env.total_steps == expected
    assert agent.global_env_step == expected
    assert agent.num_in_buffer == expected


def test_dqn_warmup_only_training_has_valid_zero_update_metrics(make_dqn):
    agent = make_dqn()
    agent.writer.add_scalar = MagicMock(wraps=agent.writer.add_scalar)
    agent.train(max_steps=2)
    assert agent.global_step == 0
    agent.writer.add_scalar.assert_any_call(schema.TRAIN_UPDATES, 0, env_step=0)
    agent.writer.add_scalar.assert_any_call(schema.TRAIN_LR, agent.lr, env_step=0)
    agent.writer.assert_contract_satisfied()


@pytest.mark.parametrize("max_frames,max_steps", [(3, 10), (7, 3)])
def test_ddpg_stops_at_frame_budget_and_marks_internal_time_limits(
    max_frames, max_steps
):
    env = CountingEnv(continuous=True, episode_length=100)
    agent = DDPG(
        env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=20, device="cpu"
    )
    agent.writer = MagicMock(spec=MetricWriter)
    try:
        result = agent.train(
            num_episodes=1,
            max_steps=max_steps,
            max_frames=max_frames,
            warmup_frames=100,
        )
        assert result["frame_idx"] == env.total_steps == max_frames
        assert len(agent.replay_buffer) == max_frames
        lengths = [c.kwargs["length"] for c in agent.writer.log_episode.call_args_list]
        assert sum(lengths) == max_frames
        assert max(lengths) <= max_steps
        assert all(
            c.kwargs["truncated"] for c in agent.writer.log_episode.call_args_list
        )
        assert all(not transition[-1] for transition in agent.replay_buffer.buffer)
    finally:
        agent.writer.close()
        env.close()


def test_ddpg_warmup_only_training_has_valid_zero_update_metrics():
    env = CountingEnv(continuous=True)
    agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=8, device="cpu")
    agent.writer = MetricWriter()
    agent.writer.add_scalar = MagicMock(wraps=agent.writer.add_scalar)
    try:
        agent.train(num_episodes=1, max_steps=1, warmup_frames=100)
        assert agent.update_count == 0
        agent.writer.add_scalar.assert_any_call(schema.TRAIN_UPDATES, 0, env_step=0)
        agent.writer.add_scalar.assert_any_call(schema.TRAIN_LR, 1e-4, env_step=0)
        agent.writer.assert_contract_satisfied()
    finally:
        agent.writer.close()
        env.close()


def test_dqn_budget_can_end_inside_an_episode(make_dqn):
    agent = make_dqn(CountingEnv(episode_length=100))
    agent.writer.log_episode = MagicMock(wraps=agent.writer.log_episode)
    result = agent.train(max_steps=3)
    assert agent.env.total_steps == 3
    assert result["episodes"] == 1
    agent.writer.log_episode.assert_called_once_with(
        reward=3.0,
        length=3,
        env_step=3,
        terminated=False,
        truncated=True,
    )
    assert all(not transition[-1] for transition in agent.replay_buffer.transitions[:3])
    agent.writer.assert_contract_satisfied()


@pytest.mark.parametrize("budget", [0, -1])
def test_dqn_rejects_nonpositive_budget_before_reset(make_dqn, budget):
    agent = make_dqn()
    with pytest.raises(ValueError, match="positive"):
        agent.train(max_steps=budget)
    assert agent.env.resets == 0


@pytest.mark.parametrize("max_frames,max_steps", [(0, 3), (-1, 3), (3, 0), (3, -1)])
def test_ddpg_rejects_nonpositive_budget_before_collecting(max_frames, max_steps):
    env = CountingEnv(continuous=True)
    agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=8, device="cpu")
    agent.writer = MagicMock(spec=MetricWriter)
    # Fail immediately if validation is missing: max_steps=0 would otherwise loop forever.
    env.reset = MagicMock(side_effect=AssertionError("Collection must not start"))
    try:
        with pytest.raises(ValueError, match="positive"):
            agent.learn(max_frames=max_frames, max_steps=max_steps, batch_size=2)
        env.reset.assert_not_called()
    finally:
        agent.writer.close()
        env.close()
