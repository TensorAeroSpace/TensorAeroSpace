"""Exercise PPO collection contracts before optimization, including time limits."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.metrics.writer import MetricWriter
from tensoraerospace.agent.ppo import model as ppo


class EndingEnv:
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def __init__(self, vector=False, terminated=False, metadata=True):
        self.vector = vector
        self.terminated = terminated
        self.metadata = metadata
        self.auto_reset = vector
        self.num_envs = 2 if vector else 1
        self.buffer = (
            torch.ones((self.num_envs, 1)) if vector else np.ones(1, np.float32)
        )

    def reset(self, **kwargs):
        self.buffer[...] = 1
        return self.buffer, {}

    def step(self, action):
        if not self.vector:
            self.buffer[...] = 4
            return self.buffer, 1.0, self.terminated, not self.terminated, {}
        info = {}
        if self.metadata:
            info = {
                "final_observation": torch.full((2, 1), 4.0),
                "_final_observation": torch.ones(2, dtype=torch.bool),
            }
        return (
            self.buffer,
            torch.ones(2),
            torch.full((2,), self.terminated),
            torch.full((2,), not self.terminated),
            info,
        )


@pytest.fixture
def make_agent(monkeypatch):
    monkeypatch.setattr(
        ppo, "create_metric_writer", lambda **kwargs: MagicMock(spec=MetricWriter)
    )
    agents = []

    def make(env, **kwargs):
        options = dict(
            device="cpu",
            seed=11,
            gamma=0.5,
            rollout_len=2,
            num_epochs=1,
            max_episodes=1,
            batch_size=16,
            normalize_obs=False,
            normalize_reward=False,
            save_best_model=False,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
        )
        options.update(kwargs)
        agent = ppo.PPO(env, **options)
        agents.append(agent)
        return agent

    yield make
    for agent in agents:
        agent.close()


def capture_update(agent):
    batches = []

    def learn(states, actions, advantages, log_probs, returns, rewards, values):
        batches.append(
            {
                "states": states.clone(),
                "returns": returns.clone(),
                "rewards": rewards.clone(),
            }
        )
        return dict(
            actor_loss=0.0,
            critic_loss=0.0,
            auxiliary_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
        )

    agent.learn = learn
    agent.critic = torch.nn.Linear(1, 1)
    with torch.no_grad():
        agent.critic.weight.fill_(2)
        agent.critic.bias.zero_()
    return batches


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
def test_timeout_targets_use_final_state_without_crossing_episode_boundary(
    make_agent, vector, terminated
):
    agent = make_agent(EndingEnv(vector=vector, terminated=terminated))
    batches = capture_update(agent)
    agent.train(verbose=False)
    targets = torch.cat([batch["returns"].reshape(-1) for batch in batches])
    torch.testing.assert_close(
        targets, torch.full_like(targets, 1.0 if terminated else 5.0)
    )
    # Auxiliary reward prediction must still receive the physical reward, not r + gamma*V.
    for batch in batches:
        torch.testing.assert_close(batch["rewards"], torch.ones_like(batch["rewards"]))
        torch.testing.assert_close(batch["states"], torch.ones_like(batch["states"]))


def test_legacy_auto_reset_without_final_observation_does_not_bootstrap_reset_state(
    make_agent,
):
    agent = make_agent(EndingEnv(vector=True, metadata=False))
    batches = capture_update(agent)
    agent.train(verbose=False)
    for batch in batches:
        torch.testing.assert_close(batch["returns"], torch.ones_like(batch["returns"]))


def test_vector_rollout_uses_same_normalization_as_inference(make_agent):
    agent = make_agent(EndingEnv(vector=True), normalize_obs=True)
    agent.obs_rms.mean[:] = -3.0
    agent.obs_rms.var[:] = 4.0
    batches = capture_update(agent)
    agent.train(verbose=False)
    for batch in batches:
        torch.testing.assert_close(
            batch["states"], torch.full_like(batch["states"], 2.0)
        )
        torch.testing.assert_close(
            batch["returns"], torch.full_like(batch["returns"], 4.5)
        )
    assert agent.obs_rms.count > 4


@pytest.mark.parametrize("vector", [False, True])
def test_single_sample_rollout_keeps_network_parameters_finite(make_agent, vector):
    env = EndingEnv(vector=vector)
    if vector:
        env.num_envs = 1
        env.buffer = torch.ones((1, 1))
        env.step = lambda action: (
            env.buffer,
            torch.ones(1),
            torch.ones(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {},
        )
    agent = make_agent(env, max_episodes=2, rollout_len=1, batch_size=1)
    agent.train(verbose=False)
    for network in [agent.actor, agent.critic]:
        assert all(torch.isfinite(p).all() for p in network.parameters())


def test_default_actor_starts_with_usable_exploration_and_roundtrips_weights():
    torch.manual_seed(11)
    actor = ppo.Actor(3, 1, hidden_dim=8)
    _, dist = actor(torch.zeros((256, 3)))
    assert torch.all(dist.stddev > 0.3)
    assert torch.all(dist.stddev < 0.9)
    restored = ppo.Actor(3, 1, hidden_dim=8)
    restored.load_state_dict(actor.state_dict())
    _, restored_dist = restored(torch.zeros((256, 3)))
    torch.testing.assert_close(restored_dist.mean, dist.mean)
    torch.testing.assert_close(restored_dist.stddev, dist.stddev)
