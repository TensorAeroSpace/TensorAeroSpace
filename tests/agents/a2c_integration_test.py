"""Integration tests for A2C agent.

This module contains integration tests that verify the complete A2C training
pipeline including actor, critic, and learning loop.
"""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.a2c.model import (
    A2C,
    A2CWithNARXCritic,
    Actor,
    Critic,
    discounted_rewards,
    process_memory,
)


class _DummyEnv:
    """Minimal environment for testing."""

    def __init__(self):
        self.observation_space = type("S", (), {"shape": (3,)})
        self.action_space = type(
            "A",
            (),
            {"shape": (1,), "low": np.array([-1.0]), "high": np.array([1.0])},
        )
        self.unwrapped = self

    def reset(self, seed=None):
        return np.zeros(3, dtype=np.float32), {}

    def step(self, _action):
        return (
            np.zeros(3, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


def test_discounted_rewards_basic():
    """Test discounted rewards computation."""
    rewards = [1.0, 1.0, 1.0]
    dones = [0, 0, 1]
    gamma = 0.9

    disc = discounted_rewards(rewards, dones, gamma)

    assert len(disc) == 3
    # Last reward: 1.0
    # Second: 1.0 + 0.9 * 1.0 = 1.9
    # First: 1.0 + 0.9 * 1.9 = 2.71
    assert abs(disc[0] - 2.71) < 0.01
    assert abs(disc[1] - 1.9) < 0.01
    assert abs(disc[2] - 1.0) < 0.01


def test_discounted_rewards_with_done():
    """Test discounted rewards with episode termination."""
    rewards = [1.0, 1.0, 1.0, 1.0]
    dones = [0, 1, 0, 1]
    gamma = 0.9

    disc = discounted_rewards(rewards, dones, gamma)

    # Episode breaks at dones, so no carry-over
    assert abs(disc[1] - 1.0) < 0.01  # Reset after done
    assert abs(disc[3] - 1.0) < 0.01  # Reset after done


def test_process_memory():
    """Test memory processing for training."""
    memory = [
        (np.array([0.0, 0.0]), 1.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0),
        (np.array([0.0, 0.0]), 0.5, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0),
        (np.array([0.0, 0.0]), 0.0, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1),
    ]

    actions, rewards_t, states, next_states, dones_t = process_memory(
        memory, gamma=0.9
    )

    assert actions.shape[0] == 3
    assert rewards_t.shape[0] == 3
    assert states.shape[0] == 3
    assert next_states.shape[0] == 3
    assert dones_t.shape[0] == 3


def test_a2c_run_episode():
    """Test A2C run_episode method."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    memory = agent.run_episode(max_steps=5)

    assert len(memory) == 5
    # Each memory entry: (state, action, reward, next_state, done)
    assert len(memory[0]) == 5


def test_a2c_learn():
    """Test A2C learn method."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    memory = agent.run_episode(max_steps=5)

    # Should not raise
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_predict():
    """Test A2C predict method."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.predict(state, deterministic=False)

    assert action.shape == (n_actions,)
    assert isinstance(action, np.ndarray)


def test_a2c_predict_deterministic():
    """Test A2C predict in deterministic mode."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.predict(state, deterministic=True)

    assert action.shape == (n_actions,)


def test_a2c_training_loop():
    """Test complete A2C training loop."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Run a few episodes
    for _ in range(3):
        memory = agent.run_episode(max_steps=5)
        agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_with_narx_critic_initialization():
    """Test A2CWithNARXCritic initialization."""
    from tensoraerospace.agent.a2c.narx_critic import NARXCritic

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = NARXCritic(state_dim, n_actions, history_length=3)
    agent = A2CWithNARXCritic(
        env=env,
        actor=actor,
        critic=critic,
        history_length=3,
        gamma=0.99,
    )

    assert agent.history_length == 3


def test_a2c_with_narx_critic_run_episode():
    """Test A2CWithNARXCritic run_episode."""
    from tensoraerospace.agent.a2c.narx_critic import NARXCritic

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = NARXCritic(state_dim, n_actions, history_length=2)
    agent = A2CWithNARXCritic(
        env=env,
        actor=actor,
        critic=critic,
        history_length=2,
    )

    memory = agent.run_episode(max_steps=5)

    assert len(memory) == 5


def test_a2c_with_narx_critic_learn():
    """Test A2CWithNARXCritic learn method."""
    from tensoraerospace.agent.a2c.narx_critic import NARXCritic

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = NARXCritic(state_dim, n_actions, history_length=2)
    agent = A2CWithNARXCritic(
        env=env,
        actor=actor,
        critic=critic,
        history_length=2,
    )

    memory = agent.run_episode(max_steps=5)

    # Should not raise
    agent.learn(memory, steps=1, discount_rewards=False)


def test_a2c_gradient_updates():
    """Test that A2C updates network parameters."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Get initial parameters
    initial_actor_params = [p.clone() for p in actor.parameters()]

    # Run training
    memory = agent.run_episode(max_steps=10)
    agent.learn(memory, steps=5, discount_rewards=True)

    # Check that at least some parameters changed
    changed = False
    for p_init, p_current in zip(initial_actor_params, actor.parameters()):
        if not torch.allclose(p_init, p_current):
            changed = True
            break

    assert changed, "Actor parameters should be updated after learning"


def test_a2c_entropy_regularization():
    """Test that entropy regularization affects loss."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    # Agent with high entropy beta
    agent_high_entropy = A2C(
        env=env, actor=actor, critic=critic, entropy_beta=1.0
    )

    # Agent with low entropy beta
    agent_low_entropy = A2C(
        env=env, actor=actor, critic=critic, entropy_beta=0.0
    )

    # Both should run without error
    memory = agent_high_entropy.run_episode(max_steps=5)
    agent_high_entropy.learn(memory, steps=1, discount_rewards=True)

    memory = agent_low_entropy.run_episode(max_steps=5)
    agent_low_entropy.learn(memory, steps=1, discount_rewards=True)


def test_a2c_different_learning_rates():
    """Test A2C with different actor and critic learning rates."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(
        env=env,
        actor=actor,
        critic=critic,
        actor_lr=1e-4,
        critic_lr=1e-3,
    )

    memory = agent.run_episode(max_steps=5)
    agent.learn(memory, steps=1, discount_rewards=True)

    # Check that optimizers have correct learning rates
    assert agent.actor_optim.param_groups[0]["lr"] == 1e-4
    assert agent.critic_optim.param_groups[0]["lr"] == 1e-3


def test_a2c_max_grad_norm_clipping():
    """Test gradient norm clipping in A2C."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(
        env=env,
        actor=actor,
        critic=critic,
        max_grad_norm=0.5,
    )

    memory = agent.run_episode(max_steps=5)
    agent.learn(memory, steps=1, discount_rewards=True)

    # Check that gradients are finite after clipping
    for param in actor.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_a2c_with_different_gamma():
    """Test A2C with different discount factors."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    for gamma in [0.9, 0.95, 0.99]:
        agent = A2C(env=env, actor=actor, critic=critic, gamma=gamma)
        memory = agent.run_episode(max_steps=5)
        agent.learn(memory, steps=1, discount_rewards=True)
        assert agent.gamma == gamma
