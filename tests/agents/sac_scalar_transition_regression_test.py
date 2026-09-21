"""Scalar SAC must snapshot observations before the environment reuses them."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from tensoraerospace.agent.sac.sac import SAC


class BufferEnv(gym.Env):
    def __init__(self, final_key=None, terminal=False):
        self.observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.state = np.zeros(1, dtype=np.float32)
        self.final_key = final_key
        self.terminal = terminal

    def reset(self, **kwargs):
        self.state[:] = 1
        return self.state, {}

    def step(self, action):
        self.state += 1
        done = self.state[0] >= 3
        info = {}
        if done and self.final_key:
            info[self.final_key] = self.state.copy()
            self.state[:] = -50  # Same-step autoreset observation.
        return self.state, 1.0, done and self.terminal, done and not self.terminal, info


@pytest.mark.parametrize(
    "final_key", [None, "final_observation", "terminal_observation"]
)
@pytest.mark.parametrize("terminal", [False, True])
def test_scalar_replay_contains_actual_transition(final_key, terminal):
    agent = SAC(
        BufferEnv(final_key, terminal), hidden_size=8, batch_size=16, memory_capacity=30
    )
    agent.writer.close()
    agent.writer = Mock()
    agent.writer.assert_contract_satisfied = Mock()
    agent.train(num_episodes=2, verbose=False)
    states, actions, rewards, next_states, terminals = map(
        np.stack, zip(*agent.memory.buffer)
    )
    np.testing.assert_array_equal(states.ravel(), [1, 2, 1, 2])
    np.testing.assert_array_equal(next_states.ravel(), [2, 3, 2, 3])
    np.testing.assert_array_equal(terminals, [0, int(terminal), 0, int(terminal)])


def test_user_step_cap_is_logged_as_truncation():
    agent = SAC(BufferEnv(), hidden_size=8, batch_size=16, memory_capacity=30)
    agent.writer.close()
    agent.writer = Mock()
    agent.writer.assert_contract_satisfied = Mock()
    agent.train(num_episodes=1, max_steps=1, verbose=False)
    assert agent.writer.log_episode.call_args.kwargs["truncated"] is True
    assert agent.memory.buffer[0][4] == 0.0


def test_short_training_satisfies_metrics_contract_without_optimizer_updates(tmp_path):
    agent = SAC(
        BufferEnv(), hidden_size=8, batch_size=16, memory_capacity=30, log_dir=tmp_path
    )
    try:
        metrics = agent.train(num_episodes=1, verbose=False)
        assert metrics["updates"] == 0
        assert len(agent.memory.buffer) == 2
        agent.writer.assert_contract_satisfied()
    finally:
        agent.close()
