"""DSAC must learn physical transitions and preserve training across calls."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.dsac import DSAC
from tensoraerospace.agent.metrics.writer import MetricWriter
from tensoraerospace.envs.b747_vec_torch import ImprovedB747VecEnvTorch


class ScalarBuffer(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def __init__(self, final_key=None, terminated=False):
        self.buffer = np.zeros(1, np.float32)
        self.final_key, self.terminated = final_key, terminated

    def reset(self, **kwargs):
        self.buffer[:] = 1
        return self.buffer, {}

    def step(self, action):
        self.buffer += 1
        final = self.buffer.copy()
        if self.final_key:
            self.buffer[:] = 99
        return (
            self.buffer,
            2.0,
            self.terminated,
            bool(self.final_key),
            ({self.final_key: final} if self.final_key else {}),
        )


class VectorBuffer:
    auto_reset = True
    observation_space = ScalarBuffer.observation_space
    action_space = ScalarBuffer.action_space

    def __init__(self, final=True, terminated=False):
        self.buffer = torch.zeros(2, 1)
        self.final, self.terminated = final, terminated

    def reset(self):
        self.buffer[:] = 1
        return self.buffer, {}

    def step(self, action):
        self.buffer[:] = 99
        info = (
            {
                "final_observation": torch.tensor([[2.0], [3.0]]),
                "_final_observation": torch.tensor([True, False]),
            }
            if self.final
            else {}
        )
        return (
            self.buffer,
            torch.ones(2),
            torch.full((2,), self.terminated),
            torch.ones(2, dtype=torch.bool),
            info,
        )


def make_agent(env, **kwargs):
    agent = DSAC(
        env,
        hidden_size=8,
        embedding_dim=4,
        num_quantiles=4,
        batch_size=2,
        memory_capacity=100,
        learning_starts=0,
        automatic_entropy_tuning=False,
        caps_lambda_smoothness=0,
        caps_lambda_temporal=0,
        device="cpu",
        **kwargs,
    )
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    return agent


@pytest.mark.parametrize("key", [None, "final_observation", "terminal_observation"])
@pytest.mark.parametrize("terminated", [False, True])
def test_scalar_replay_owns_pre_step_and_terminal_observations(key, terminated):
    agent = make_agent(ScalarBuffer(key, terminated))
    try:
        agent.train(1, max_steps=1, verbose=False)
        state, _, _, next_state, terminal = agent.memory.buffer[0]
        np.testing.assert_array_equal(state, [1])
        np.testing.assert_array_equal(next_state, [2])
        assert terminal == float(terminated)
        agent.env.buffer[:] = 700
        np.testing.assert_array_equal(state, [1])
        np.testing.assert_array_equal(next_state, [2])
    finally:
        agent.close()


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
def test_vector_replay_preserves_pre_step_and_available_final_observations(
    final, terminated
):
    agent = make_agent(VectorBuffer(final, terminated))
    try:
        agent.train_vector(total_steps=1, warmup_steps=1, log_every=1)
        first, second = agent.memory.buffer
        np.testing.assert_array_equal(first[0], [1])
        np.testing.assert_array_equal(first[3], [2 if final else 99])
        assert first[4] == float(terminated or not final)
        np.testing.assert_array_equal(second[3], [99])
        assert second[4] == 1.0  # no final observation: conservative fallback
        assert agent.writer.log_episode.call_args_list[0].kwargs["env_step"] == 2
    finally:
        agent.close()


@pytest.mark.parametrize("batch", [False, True])
def test_evaluation_ignores_exploration_noise_and_preserves_rng(batch):
    agent = make_agent(ScalarBuffer(), exploration_noise_std=0.5)
    try:
        obs = torch.ones(3, 1) if batch else np.ones(1, np.float32)
        before = torch.get_rng_state().clone()
        call = agent.select_action_batch if batch else agent.select_action
        first = call(obs, evaluate=True)
        second = call(obs, evaluate=True)
        np.testing.assert_array_equal(first, second)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        call(obs, evaluate=False)
        assert not torch.equal(torch.get_rng_state(), before)
    finally:
        agent.close()


def test_repeated_scalar_training_continues_warmup_and_update_indices():
    agent = make_agent(ScalarBuffer())
    agent.learning_starts = 2
    agent.update_parameters = MagicMock(return_value=(0, 0, 0, 0, 0))
    agent.env.action_space.sample = MagicMock(return_value=np.zeros(1, np.float32))
    try:
        first = agent.train(1, max_steps=3, verbose=False)
        second = agent.train(1, max_steps=3, verbose=False)
        assert agent.env.action_space.sample.call_count == 2
        assert [c.args[2] for c in agent.update_parameters.call_args_list] == [
            0,
            1,
            2,
            3,
        ]
        assert agent.total_env_steps == 6
        assert agent.total_updates == 4
        assert first["total_steps"] == second["total_steps"] == 3
        assert first["best_reward"] == second["best_reward"] == 6
        logged = agent.writer.log_episode.call_args_list
        assert [c.kwargs["env_step"] for c in logged] == [3, 6]
        assert all(c.kwargs["truncated"] for c in logged)
    finally:
        agent.close()


def test_short_training_before_warmup_satisfies_real_metrics_contract(tmp_path):
    agent = DSAC(
        ScalarBuffer(),
        batch_size=4,
        learning_starts=20,
        hidden_size=8,
        log_dir=tmp_path,
        device="cpu",
    )
    try:
        result = agent.train(1, max_steps=2, verbose=False)
        assert result["updates"] == 0
        assert result["best_reward"] == 4
    finally:
        agent.close()


def test_training_counters_survive_checkpoint(tmp_path):
    env = ImprovedB747VecEnvTorch(num_envs=2, device="cpu", dt=0.05, tn=1.0)
    agent = make_agent(env)
    restored = None
    try:
        agent.train_vector(total_steps=3, warmup_steps=3, log_every=1)
        agent.total_updates = 17
        path = agent.save(tmp_path)
        restored = DSAC.from_pretrained(path)
        assert restored.total_env_steps == 6
        assert restored.total_updates == 17
    finally:
        agent.close()
        if restored is not None:
            restored.close()


def test_chunked_scalar_training_matches_continuous_network_updates():
    def train(chunks):
        agent = make_agent(ScalarBuffer(), target_update_interval=3, seed=29)
        agent.learning_starts = 2
        agent.env.action_space.seed(29)
        try:
            for count in chunks:
                agent.train(count, max_steps=4, verbose=False)
            return {
                name: {
                    k: v.detach().clone()
                    for k, v in getattr(agent, name).state_dict().items()
                }
                for name in ["policy", "Z1", "Z2", "Z1_target", "Z2_target"]
            }
        finally:
            agent.close()

    continuous, chunked = train([2]), train([1, 1])
    for name in continuous:
        for key in continuous[name]:
            torch.testing.assert_close(
                continuous[name][key], chunked[name][key], rtol=0, atol=0
            )
