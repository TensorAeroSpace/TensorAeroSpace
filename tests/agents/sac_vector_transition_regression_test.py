"""SAC must store physical transitions across vector environment auto-resets."""

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.metrics.writer import MetricWriter
from tensoraerospace.agent.sac import SAC
from tensoraerospace.agent.sac.replay_memory import ReplayMemory
from tensoraerospace.envs.b747_vec_torch import ImprovedB747VecEnvTorch


class AutoResetVector:
    auto_reset = True
    action_space = gym.spaces.Box(2.0, 4.0, (1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-100.0, 100.0, (1,), dtype=np.float32)

    def __init__(self, terminated=False, with_final=True):
        self.terminated = terminated
        self.with_final = with_final
        self.actions = []
        self.observation = torch.zeros((2, 1))

    def reset(self):
        self.observation.fill_(1.0)
        return self.observation, {}

    def step(self, action):
        self.actions.append(action.clone())
        self.observation.fill_(99.0)
        info = (
            {
                "final_observation": torch.tensor([[2.0], [3.0]]),
                "_final_observation": torch.tensor([True, True]),
            }
            if self.with_final
            else {}
        )
        return (
            self.observation,
            torch.ones(2),
            torch.full((2,), self.terminated),
            torch.ones(2, dtype=torch.bool),
            info,
        )


@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("with_final", [False, True])
def test_auto_reset_replay_uses_final_state_and_true_termination(
    terminated, with_final
):
    env = AutoResetVector(terminated, with_final)
    agent = SAC(env, hidden_size=8, batch_size=32, memory_capacity=16, device="cpu")
    agent.writer.close()
    agent.writer = MagicMock(spec=MetricWriter)
    try:
        agent.train_vector(total_steps=2, warmup_steps=3, save_best=False)
        transitions = agent.memory.buffer
        np.testing.assert_array_equal(transitions[0][0], [1.0])
        np.testing.assert_array_equal(transitions[0][3], [2.0 if with_final else 99.0])
        np.testing.assert_array_equal(transitions[1][3], [3.0 if with_final else 99.0])
        assert transitions[0][4] == float(terminated or not with_final)
        assert all(torch.all((a >= 2.0) & (a <= 4.0)) for a in env.actions)
        assert agent.writer.log_episode.call_args_list[0].kwargs["env_step"] == 2
    finally:
        agent.close()


def test_b747_exposes_observation_before_auto_reset():
    options = dict(
        num_envs=2,
        dt=0.1,
        tn=0.2,
        device="cpu",
        seed=4,
        initial_state=np.array([0.0, 0.0, 0.01, 0.02]),
    )
    auto = ImprovedB747VecEnvTorch(**options, auto_reset=True)
    manual = ImprovedB747VecEnvTorch(**options, auto_reset=False)
    expected, _, _, expected_truncated, _ = manual.step(torch.zeros((2, 1)))
    actual, _, _, truncated, info = auto.step(torch.zeros((2, 1)))
    assert torch.all(truncated == expected_truncated)
    torch.testing.assert_close(info["final_observation"], expected)
    assert torch.all(info["_final_observation"])
    assert not torch.allclose(actual, expected)
    saved = info["final_observation"].clone()
    auto.step(torch.ones((2, 1)))
    torch.testing.assert_close(info["final_observation"], saved)


def test_replay_buffer_owns_transition_arrays():
    memory = ReplayMemory(4, seed=0)
    state, action, reward, next_state = (
        np.array([x], dtype=float) for x in [1, 2, 3, 4]
    )
    memory.push(state, action, reward, next_state, False)
    for array in (state, action, reward, next_state):
        array[:] = 99
    for actual, expected in zip(memory.buffer[0][:4], [1, 2, 3, 4]):
        np.testing.assert_array_equal(actual, [expected])
