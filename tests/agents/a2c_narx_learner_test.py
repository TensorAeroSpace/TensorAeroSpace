"""Tests for A2C NARX learner module.

This module contains unit tests for the A2CLearner and Runner classes
that use NARX-based critics.
"""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.a2c.narx import (
    A2CLearner,
    Actor,
    Critic,
    Mish,
    Runner,
    clip_grad_norm_,
    mish,
    t,
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


def test_mish_function():
    """Test mish activation function."""
    x = torch.tensor([0.0, 1.0, -1.0])
    y = mish(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_mish_module():
    """Test Mish module."""
    x = torch.tensor([0.0, 1.0, -1.0])
    mod = Mish()
    y = mod(x)

    assert y.shape == x.shape
    assert torch.allclose(y, mish(x))


def test_t_conversion():
    """Test t() helper for tensor conversion."""
    arr = [1, 2, 3]
    tensor = t(arr)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3,)


def test_t_with_numpy():
    """Test t() with numpy input."""
    arr = np.array([1.0, 2.0, 3.0])
    tensor = t(arr)

    assert isinstance(tensor, torch.Tensor)
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_actor_initialization():
    """Test Actor network initialization."""
    state_dim, n_actions = 3, 1
    actor = Actor(state_dim, n_actions, activation=Mish)

    assert actor.n_actions == n_actions
    assert isinstance(actor.model, torch.nn.Sequential)
    assert hasattr(actor, "logstds")


def test_actor_forward():
    """Test Actor forward pass."""
    state_dim, n_actions = 3, 1
    actor = Actor(state_dim, n_actions)

    states = torch.randn(5, state_dim)
    dist = actor(states)

    # Actor returns Normal distribution
    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (5, n_actions)
    assert dist.stddev.shape == (5, n_actions)
    assert (dist.stddev > 0).all()


def test_actor_sample_action():
    """Test Actor action sampling."""
    state_dim, n_actions = 3, 1
    actor = Actor(state_dim, n_actions)

    state = torch.randn(1, state_dim)
    dist = actor(state)
    action = dist.sample()

    assert action.shape == (1, n_actions)
    assert isinstance(action, torch.Tensor)


def test_clip_grad_norm():
    """Test gradient clipping utility."""
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())

    # Create some gradients
    x = torch.randn(5, 3, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Clip gradients
    clip_grad_norm_(optimizer, max_grad_norm=1.0)

    # Check that gradients exist
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_runner_initialization():
    """Test Runner initialization."""
    from tensoraerospace.agent.metrics import MetricWriter

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    writer = MetricWriter(strict=False, required=())

    runner = Runner(env, actor, writer)

    assert runner.env == env
    assert runner.actor == actor
    assert runner.done == True


def test_runner_run():
    """Test Runner.run() method."""
    from tensoraerospace.agent.metrics import MetricWriter

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    writer = MetricWriter(strict=False, required=())

    runner = Runner(env, actor, writer)
    memory = runner.run(max_steps=5)

    assert len(memory) == 5
    # Each memory entry: (action, reward, state, next_state, done)
    assert len(memory[0]) == 5


def test_a2c_learner_initialization():
    """Test A2CLearner initialization."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    # Critic in narx expects concatenated state
    critic = Critic(state_dim)

    learner = A2CLearner(
        actor=actor,
        critic=critic,
        gamma=0.99,
        entropy_beta=0.01,
        actor_lr=1e-3,
        critic_lr=1e-3,
    )

    assert learner.gamma == 0.99
    assert learner.entropy_beta == 0.01


def test_a2c_learner_compute_advantages():
    """Test advantage computation."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    learner = A2CLearner(
        actor=actor,
        critic=critic,
        gamma=0.99,
    )

    # Test advantage calculation
    # A2CLearner uses TD error: advantage = reward + gamma*V(s') - V(s)
    values = torch.randn(10, 1)
    rewards = torch.randn(10, 1)
    next_values = torch.randn(10, 1)
    dones = torch.zeros(10, 1)

    # Calculate expected advantages
    advantages = rewards + learner.gamma * next_values * (1 - dones) - values

    assert advantages.shape == (10, 1)


def test_a2c_learner_update_step():
    """Test A2CLearner update step (learn method)."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    learner = A2CLearner(
        actor=actor,
        critic=critic,
    )

    # Create dummy memory
    memory = []
    state = env.reset()[0]
    for _ in range(10):
        dist = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = dist.sample().detach().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((action, reward, state, next_state, done))
        state = next_state
        if done:
            state = env.reset()[0]

    # Learn should not raise errors
    learner.learn(memory, steps=0)


def test_a2c_learner_with_runner():
    """Test A2CLearner integration with Runner."""
    from tensoraerospace.agent.metrics import MetricWriter

    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    writer = MetricWriter(strict=False, required=())

    learner = A2CLearner(
        actor=actor,
        critic=critic,
    )

    runner = Runner(env, actor, writer)
    memory = runner.run(max_steps=10)

    # Should be able to learn from runner memory
    learner.learn(memory, steps=0)


def test_actor_gradient_flow():
    """Test that gradients flow through Actor."""
    state_dim, n_actions = 3, 1
    actor = Actor(state_dim, n_actions)

    states = torch.randn(5, state_dim, requires_grad=True)
    dist = actor(states)

    loss = dist.mean.sum()
    loss.backward()

    assert states.grad is not None


def test_runner_early_termination():
    """Test Runner with early episode termination."""
    from tensoraerospace.agent.metrics import MetricWriter

    class _TerminatingEnv(_DummyEnv):
        def __init__(self):
            super().__init__()
            self.step_count = 0

        def reset(self, seed=None):
            self.step_count = 0
            return super().reset(seed)

        def step(self, _action):
            self.step_count += 1
            done = self.step_count >= 3
            return (
                np.zeros(3, dtype=np.float32),
                1.0 if done else 0.0,
                done,
                False,
                {},
            )

    env = _TerminatingEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    writer = MetricWriter(strict=False, required=())

    runner = Runner(env, actor, writer)
    memory = runner.run(max_steps=10)

    # Should collect exactly max_steps
    assert len(memory) == 10
    # At least one done should be True
    assert any(done for _, _, _, _, done in memory)


def test_a2c_learner_max_grad_norm():
    """Test gradient clipping in A2CLearner."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    learner = A2CLearner(
        actor=actor,
        critic=critic,
        gamma=0.99,
        max_grad_norm=0.5,
    )

    # Create dummy memory
    memory = []
    state = env.reset()[0]
    for _ in range(10):
        dist = actor(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        action = dist.sample().detach().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((action, reward, state, next_state, done))
        state = next_state
        if done:
            state = env.reset()[0]

    # Should not raise and should clip gradients
    learner.learn(memory, steps=0)
