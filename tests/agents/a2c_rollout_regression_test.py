"""Check real A2C updates and physical transition histories."""

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.a2c import model as a2c
from tensoraerospace.agent.a2c.narx_critic import NARXCritic


class BufferEnv(gym.Env):
    def __init__(self, terminal=False):
        self.observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-0.1, 0.1, (1,), dtype=np.float32)
        self.state = np.zeros(1, dtype=np.float32)
        self.terminal = terminal
        self.command = None

    def reset(self, **kwargs):
        self.state[:] = 1.0
        return self.state, {}

    def step(self, command):
        self.command = command.copy()
        self.state += 1.0
        done = bool(self.state[0] >= 3.0)
        return self.state, 1.0, done and self.terminal, done and not self.terminal, {}


def make_agent(narx=False, terminal=False):
    torch.manual_seed(11)
    actor = a2c.Actor(1, 1)
    critic = NARXCritic(1, 1, 2) if narx else a2c.Critic(1)
    cls = a2c.A2CWithNARXCritic if narx else a2c.A2C
    agent = cls(
        BufferEnv(terminal),
        actor,
        critic,
        device="cpu",
        **({"history_length": 2} if narx else {}),
    )
    agent.writer.close()
    agent.writer = Mock()
    return agent


@pytest.mark.parametrize("narx", [False, True])
def test_single_transition_update_keeps_weights_and_metrics_finite(narx):
    agent = make_agent(narx)
    agent.learn(agent.run_episode(1), steps=1)
    assert all(
        torch.isfinite(p).all()
        for network in [agent.actor, agent.critic]
        for p in network.parameters()
    )
    assert all(
        np.isfinite(call.args[1]) for call in agent.writer.add_scalar.call_args_list
    )


def test_collection_snapshots_states_and_keeps_sampled_action():
    agent = make_agent()
    with torch.no_grad():
        for p in agent.actor.parameters():
            p.zero_()
        agent.actor.model[-1].bias.fill_(3.0)
        agent.actor.logstds.fill_(-10.0)
    memory = agent.run_episode(3)
    assert memory[0][0][0] > 2.9  # Gaussian sample, not the clipped command.
    assert abs(agent.env.command[0]) <= 0.100001
    np.testing.assert_array_equal(memory[0][2], [1.0])
    np.testing.assert_array_equal(memory[0][3], [2.0])
    np.testing.assert_array_equal(memory[1][3], [3.0])


@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_value_targets_distinguish_time_limit_from_terminal(
    monkeypatch, discount, terminal
):
    agent = make_agent(terminal=terminal)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        critic.weight.fill_(1.0)
    agent.critic = critic
    agent.critic_optim = torch.optim.SGD(critic.parameters(), lr=0.0)
    agent.gamma = 0.5
    targets = []
    mse = a2c.F.mse_loss

    def capture(value, target):
        assert not target.requires_grad
        targets.append(target.detach().clone())
        return mse(value, target)

    monkeypatch.setattr(a2c.F, "mse_loss", capture)
    agent.learn(agent.run_episode(2), 2, discount_rewards=discount)
    last = 1.0 if terminal else 2.5
    first = 1.0 + 0.5 * last if discount else 2.0
    torch.testing.assert_close(targets[0], torch.tensor([[first], [last]]))


def test_narx_history_tracks_actual_commands_across_rollouts_and_resets():
    agent = make_agent(narx=True)
    first = agent.run_episode(1)
    second = agent.run_episode(2)
    states_seen = []
    handle = agent.critic.register_forward_pre_hook(
        lambda module, args: states_seen.append(args[0].detach().clone())
    )
    agent.learn(second, 3, discount_rewards=False)
    handle.remove()
    # Features: [current state, previous state, previous input, input before that].
    command0 = float(first[0].executed_action[0])
    current = torch.tensor([[2.0, 1.0, command0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert any(torch.allclose(features, current) for features in states_seen)
    # Next critic state includes this transition's *executed* input.
    expected_next = torch.tensor(
        [
            [3.0, 2.0, float(second[0].executed_action[0]), command0],
            [2.0, 1.0, float(second[1].executed_action[0]), 0.0],
        ]
    )
    assert any(torch.allclose(features, expected_next) for features in states_seen)


@pytest.mark.parametrize("narx", [False, True])
def test_single_transition_preserves_policy_gradient(narx):
    agent = make_agent(narx)
    agent.entropy_beta = 0.0
    for group in agent.critic_optim.param_groups:
        group["lr"] = 0.0
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.zero_()
        state = torch.ones((1, 1))
        before = agent.actor(state).mean.item()
    memory = [
        (
            np.array([before + 0.2], dtype=np.float32),
            1.0,
            np.array([1.0]),
            np.array([2.0]),
            True,
        )
    ]
    agent.learn(memory, 1)
    assert agent.actor(state).mean.item() > before
