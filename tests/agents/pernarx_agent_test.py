"""Numerical and lifecycle regression tests for the PER-NARX DQN agent."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from tensoraerospace.agent.dqn import model as dqn
from tensoraerospace.agent.metrics import schema


class LinearQ(nn.Linear):
    """Small, real Q network with analytically tractable outputs."""

    def __init__(self):
        super().__init__(1, 2, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[0.2], [0.4]]))

    def predict(self, obs):
        with torch.no_grad():
            return self(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def action_value(self, obs):
        values = self.predict(obs)[0]
        return int(values.argmax()), values


class EpisodeEnv(gym.Env):
    """Two-step episodes with known rewards and selectable termination API."""

    observation_space = gym.spaces.Box(-10.0, 10.0, (1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, *, truncated=False, legacy=False):
        self.truncated = truncated
        self.legacy = legacy
        self.steps = 0
        self.resets = 0
        self.renders = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.resets += 1
        obs = np.array([1.0], dtype=np.float32)
        return obs if self.legacy else (obs, {})

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1
        obs = np.array([1.0 + self.steps], dtype=np.float32)
        done = self.steps == 2
        if self.legacy:
            return obs, float(self.steps), done, {}
        return (
            obs,
            float(self.steps),
            done and not self.truncated,
            done and self.truncated,
            {},
        )

    def render(self):
        self.renders += 1

    def close(self):
        self.closed = True


@pytest.fixture
def agent_factory(monkeypatch):
    monkeypatch.setattr(dqn, "_DEVICE", torch.device("cpu"))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    np.random.seed(17)
    agents = []

    def make(**kwargs):
        settings = dict(
            env=EpisodeEnv(),
            buffer_size=4,
            batch_size=2,
            gamma=0.5,
            alpha=0.5,
            beta=0.5,
            beta_increment_per_sample=0.0,
            epsilon=0.0,
            target_update_iter=2,
            train_nums=12,
        )
        settings.update(kwargs)
        agent = dqn.PERNARXAgent(LinearQ(), LinearQ(), **settings)
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()
        agent.env.close()


def fill_two_transitions(agent):
    for priority, state, action, reward, done in (
        (1.0, 1.0, 0, 0.1, False),
        (4.0, 2.0, 1, -0.4, True),
    ):
        agent.store_transition(
            priority,
            np.array([state], dtype=np.float32),
            action,
            reward,
            np.array([2.0], dtype=np.float32),
            done,
        )
    agent.num_in_buffer = 2


def test_sample_uses_filled_priorities_and_normalized_importance_weights(
    agent_factory,
    monkeypatch,
):
    agent = agent_factory(beta=0.9, beta_increment_per_sample=0.2)
    fill_two_transitions(agent)
    # Select one point inside each populated leaf. Unfilled leaves stay at zero.
    draws = iter([0.5, 3.0])
    monkeypatch.setattr(np.random, "uniform", lambda low, high: next(draws))
    indices, weights = agent.sum_tree_sample(2)

    assert indices == [3, 4]
    assert agent.beta == 1.0
    np.testing.assert_allclose(weights, [[1.0], [0.25]])
    np.testing.assert_array_equal(agent.b_obs, [[1.0], [2.0]])
    np.testing.assert_array_equal(agent.b_next_states, [[2.0], [2.0]])
    np.testing.assert_array_equal(agent.b_actions, [0, 1])
    np.testing.assert_allclose(agent.b_rewards, [0.1, -0.4])
    np.testing.assert_array_equal(agent.b_dones, [False, True])


def test_double_dqn_loss_gradients_priorities_and_frozen_target(
    agent_factory, monkeypatch
):
    agent = agent_factory()
    fill_two_transitions(agent)
    with torch.no_grad():
        # Target prefers action 0, online prefers 1: Double DQN must select 1.
        agent.target_model.weight.copy_(torch.tensor([[2.0], [0.5]]))
    target_before = agent.target_model.weight.detach().clone()
    online_before = agent.model.weight.detach().clone()
    draws = iter([0.5, 3.0])
    monkeypatch.setattr(np.random, "uniform", lambda low, high: next(draws))

    loss = agent.train_step()

    # Targets: [0.1 + 0.5 * 1.0, -0.4]; predictions: [0.2, 0.8].
    # Importance weights: [1, 0.5]. Terminal transition has no bootstrap.
    assert loss == pytest.approx(0.44, abs=1e-7)
    torch.testing.assert_close(agent.model.weight.grad, torch.tensor([[-0.4], [1.2]]))
    assert agent.model.weight[0] > online_before[0]
    assert agent.model.weight[1] < online_before[1]
    torch.testing.assert_close(agent.target_model.weight, target_before)
    assert agent.target_model.weight.grad is None
    np.testing.assert_allclose(
        agent.replay_buffer.tree[3:5], np.sqrt([0.41, 1.01]), rtol=1e-6
    )
    assert agent.global_step == 1

    agent.update_target_model()
    torch.testing.assert_close(agent.target_model.weight, agent.model.weight)
    np.testing.assert_allclose(
        agent.get_target_value(np.array([[2.0]], dtype=np.float32)),
        agent.model.predict(np.array([[2.0]], dtype=np.float32)),
    )


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize(
    "budget,expected_budget",
    [({}, 12), ({"max_steps": 14}, 14), ({"num_episodes": 3, "max_steps": 6}, 18)],
)
def test_train_updates_network_and_records_complete_episodes(
    agent_factory, tmp_path, truncated, budget, expected_budget
):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    env = EpisodeEnv(truncated=truncated)
    agent = agent_factory(env=env, log_dir=tmp_path / "metrics")
    before = agent.model.weight.detach().clone()
    result = agent.train(**budget)
    agent.writer.flush()
    events = EventAccumulator(str(tmp_path / "metrics")).Reload()

    assert agent.train_nums == expected_budget
    assert result["episodes"] > 0
    assert result["episodes"] == env.resets - 1
    assert agent.num_in_buffer == agent.buffer_size
    assert agent.global_step > 0
    assert not torch.equal(before, agent.model.weight)
    assert all(
        item.value == 3.0 for item in events.Scalars(schema.ROLLOUT_EPISODE_REWARD)
    )
    assert all(
        item.value == 2.0 for item in events.Scalars(schema.ROLLOUT_EPISODE_LENGTH)
    )
    assert all(
        item.value == float(truncated)
        for item in events.Scalars(schema.DIAG_TRUNCATED_COUNT)
    )
    assert all(
        item.value == float(not truncated)
        for item in events.Scalars(schema.DIAG_TERMINATED_COUNT)
    )
    assert [event.step for event in events.Scalars(schema.DQN.TARGET_UPDATE)] == list(
        range(
            agent.target_update_iter,
            agent.global_env_step + 1,
            agent.target_update_iter,
        )
    )
    assert all(np.isfinite(item.value) for item in events.Scalars(schema.DQN.LOSS_Q))


@pytest.mark.parametrize(
    "legacy,truncated,render",
    [(False, False, False), (False, True, True), (True, False, True)],
)
def test_evaluation_reward_and_environment_lifecycle(
    agent_factory, monkeypatch, legacy, truncated, render
):
    monkeypatch.setattr(dqn.time, "sleep", lambda _: None)
    agent = agent_factory()
    env = EpisodeEnv(legacy=legacy, truncated=truncated)
    before = agent.model.weight.detach().clone()
    assert agent.evaluation(env, render=render) == 3.0
    assert env.steps == 2
    assert env.renders == (2 if render else 0)
    assert env.closed
    torch.testing.assert_close(agent.model.weight, before)


def test_exploration_and_epsilon_decay(agent_factory, monkeypatch):
    agent = agent_factory(epsilon_dacay=0.5)
    monkeypatch.setattr(agent.env.action_space, "sample", lambda: 0)
    assert agent.get_action(1) == 1
    agent.epsilon = 1.0
    assert agent.get_action(1) == 0
    agent.e_decay()
    assert agent.epsilon == 0.5


@pytest.mark.parametrize("save_gradients", [False, True])
def test_save_restores_predictions_and_optimizer_update(
    agent_factory, tmp_path, save_gradients
):
    agent = agent_factory()
    fill_two_transitions(agent)
    agent.train_step()
    path = tmp_path / "checkpoint" if save_gradients else Path.cwd() / "dqn_pernarx"
    agent.save(path if save_gradients else None, save_gradients=save_gradients)
    config = json.loads((path / "config.json").read_text())
    assert config["checkpoint_format"] == "state_dict"
    assert config["obs_shape"] == [1]
    assert config["num_actions"] == 2
    assert config["gamma"] == agent.gamma
    assert (path / "optimizer.pth").exists() == save_gradients

    restored = agent_factory()
    restored.model.load_state_dict(torch.load(path / "model.pth", weights_only=True))
    restored.target_model.load_state_dict(
        torch.load(path / "target_model.pth", weights_only=True)
    )
    obs = np.array([[1.0], [-2.0]], dtype=np.float32)
    np.testing.assert_array_equal(restored.model.predict(obs), agent.model.predict(obs))
    np.testing.assert_array_equal(
        restored.get_target_value(obs), agent.get_target_value(obs)
    )
    if save_gradients:
        restored.optimizer.load_state_dict(
            torch.load(path / "optimizer.pth", weights_only=True)
        )
        # Identical data and sampling must produce the same next optimizer update.
        # Reset priorities after the previous training step.
        for current in (agent, restored):
            current.replay_buffer = dqn.SumTree(current.buffer_size)
            fill_two_transitions(current)
        np.random.seed(41)
        loss = agent.train_step()
        np.random.seed(41)
        assert restored.train_step() == pytest.approx(loss)
        torch.testing.assert_close(restored.model.weight, agent.model.weight)
