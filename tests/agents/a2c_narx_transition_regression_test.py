"""Regression tests for the temporal and gradient contracts of NARX A2C."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.a2c import narx


def test_next_critic_state_uses_current_state_and_history_resets():
    memory = [
        (np.array([0.1, 0.2]), 1.0, np.array([2.0]), np.array([3.0]), True),
        (np.array([0.3, 0.4]), 2.0, np.array([8.0]), np.array([9.0]), False),
    ]
    actions, _, _, next_states, _, critic_states = narx.process_memory_narx(memory)
    assert actions.shape == (2, 2)
    torch.testing.assert_close(next_states, torch.tensor([[3.0, 2.0], [9.0, 8.0]]))
    torch.testing.assert_close(critic_states, torch.tensor([[2.0, 0.0], [8.0, 0.0]]))


class ReusingEnv(gym.Env):
    def __init__(self, terminal=False):
        self.observation_space = gym.spaces.Box(-100.0, 100.0, (1,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,))
        self.buffer = np.zeros(1, dtype=np.float32)
        self.terminal = terminal

    def reset(self, **kwargs):
        self.buffer[:] = 1.0
        return self.buffer, {}

    def step(self, action):
        self.buffer += 1.0
        done = bool(self.buffer[0] >= 3)
        return self.buffer, 1.0, done and self.terminal, done and not self.terminal, {}


def test_runner_snapshots_observations_and_preserves_history_across_batches():
    runner = narx.Runner(ReusingEnv(), narx.Actor(1, 2), Mock())
    first = runner.run(1)
    second = runner.run(2)
    np.testing.assert_array_equal(first[0][2], [1.0])
    np.testing.assert_array_equal(first[0][3], [2.0])
    _, _, _, _, _, history = narx.process_memory_narx(second)
    torch.testing.assert_close(history, torch.tensor([[2.0, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize("discount", [True, False])
@pytest.mark.parametrize("terminal", [True, False])
def test_learner_bootstraps_time_limits_with_detached_targets(
    monkeypatch, discount, terminal
):
    actor = narx.Actor(1, 2)
    critic = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[1.0, 0.0]]))
    learner = narx.A2CLearner(actor, critic, gamma=0.5, actor_lr=0.0, critic_lr=0.0)
    learner.writer.close()
    learner.writer = Mock()
    runner = narx.Runner(ReusingEnv(terminal), actor, Mock())
    memory = runner.run(2)
    captured = {}
    mse = narx.F.mse_loss

    def capture(value, target):
        captured["target"] = target.detach().clone()
        captured["requires_grad"] = target.requires_grad
        return mse(value, target)

    monkeypatch.setattr(narx.F, "mse_loss", capture)
    learner.learn(memory, 2, discount_rewards=discount)
    final_target = 1.0 if terminal else 2.5
    first_target = 1.0 + 0.5 * final_target if discount else 2.0
    torch.testing.assert_close(
        captured["target"], torch.tensor([[first_target], [final_target]])
    )
    assert not captured["requires_grad"]
    assert all(torch.isfinite(p).all() for p in actor.parameters())


def test_transition_metadata_survives_serialization():
    import pickle

    runner = narx.Runner(ReusingEnv(), narx.Actor(1, 2), Mock())
    memory = runner.run(2)
    restored = pickle.loads(pickle.dumps(memory))
    assert len(restored[-1]) == 5
    assert restored[-1][4] and not restored[-1].terminated
    np.testing.assert_array_equal(restored[-1].previous_state, [1.0])


def test_empty_memory_has_clear_error():
    with pytest.raises(ValueError, match="memory"):
        narx.process_memory_narx([])
