"""Splitting training at episode boundaries must preserve SAC's update schedule."""

import json
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.metrics.writer import MetricWriter
from tensoraerospace.agent.sac import SAC


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10.0, 10.0, (1,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        return (
            np.array([self.t / 10], np.float32),
            float(1 - np.square(action).sum()),
            False,
            self.t == 7,
            {},
        )


class TinyVector(TinyEnv):
    auto_reset = True

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return torch.zeros((2, 1)), {}

    def step(self, action):
        self.t += 1
        final = torch.full((2, 1), self.t / 10)
        done = self.t == 7
        if done:
            self.t = 0
        obs = torch.full((2, 1), self.t / 10)
        return (
            obs,
            1 - action.square().flatten(),
            torch.zeros(2, dtype=torch.bool),
            torch.full((2,), done),
            {"final_observation": final, "_final_observation": torch.full((2,), done)},
        )


def make_agent(tmp_path, env=None, **kwargs):
    agent = SAC(
        env or TinyEnv(),
        hidden_size=8,
        batch_size=4,
        memory_capacity=200,
        target_update_interval=4,
        tau=0.3,
        seed=13,
        log_dir=tmp_path,
        **kwargs,
    )
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    return agent


@pytest.mark.parametrize("vector", [False, True])
def test_episode_chunks_match_one_training_call(tmp_path, vector):
    agents = []
    for chunked in (False, True):
        agent = make_agent(
            tmp_path / str(chunked), TinyVector() if vector else TinyEnv()
        )
        if vector:
            for _ in range(3 if chunked else 1):
                agent.train_vector(total_steps=7 if chunked else 21, warmup_steps=0)
        else:
            for _ in range(3 if chunked else 1):
                agent.train(num_episodes=1 if chunked else 3, verbose=False)
        agents.append(agent)
    for name in ("policy", "critic", "critic_target"):
        for a, b in zip(
            getattr(agents[0], name).parameters(), getattr(agents[1], name).parameters()
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    steps = [
        call.kwargs["env_step"] for call in agents[1].writer.log_episode.call_args_list
    ]
    assert steps == ([14, 14, 28, 28, 42, 42] if vector else [7, 14, 21])
    assert agents[1].total_env_steps == (42 if vector else 21)
    assert agents[1].total_updates == (20 if vector else 16)


def test_best_reward_does_not_depend_on_saving(tmp_path):
    agent = make_agent(tmp_path)
    result = agent.train(num_episodes=2, verbose=False)
    assert result["best_reward"] == max(result["episode_rewards"])
    assert result["updates"] == 9
    result2 = agent.train(verbose=False)
    assert result2["updates"] == 7


@pytest.mark.parametrize("policy_type", ["Gaussian", "Deterministic"])
def test_policy_learning_rate_is_independent_of_critic(tmp_path, policy_type):
    agent = make_agent(tmp_path, lr=1e-3, policy_lr=1e-5, policy_type=policy_type)
    assert agent.policy_optim.param_groups[0]["lr"] == 1e-5
    assert agent.get_param_env()["policy"]["params"]["policy_lr"] == 1e-5


@pytest.mark.parametrize("gradients", [False, True])
def test_checkpoint_restores_training_counters_and_policy_lr(tmp_path, gradients):
    agent = make_agent(tmp_path / "logs", lr=1e-3, policy_lr=1e-5)
    agent.train(num_episodes=2, verbose=False)
    agent.save(tmp_path / "saved", save_gradients=gradients)
    folder = next((tmp_path / "saved").iterdir())
    with patch(
        "tensoraerospace.agent.sac.sac.get_class_from_string", return_value=TinyEnv
    ):
        restored = SAC.from_pretrained(str(folder), load_gradients=gradients)
    try:
        assert restored.policy_optim.param_groups[0]["lr"] == 1e-5
        assert restored.total_updates == 9
        assert restored.total_env_steps == 14
        assert restored._last_env_step == 14
    finally:
        restored.close()
    config_path = folder / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("training", None)
    config["policy"]["params"].pop("policy_lr", None)
    config_path.write_text(json.dumps(config))
    with patch(
        "tensoraerospace.agent.sac.sac.get_class_from_string", return_value=TinyEnv
    ):
        legacy = SAC.from_pretrained(str(folder))
    try:
        assert legacy.total_updates == legacy.total_env_steps == 0
    finally:
        legacy.close()
