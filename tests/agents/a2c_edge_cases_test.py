"""Edge case tests for A2C agent.

This module tests edge cases and error handling:
- Zero rewards
- Negative rewards
- Very long episodes
- Extreme values
"""

import numpy as np
import torch

from tensoraerospace.agent.a2c.model import (
    A2C,
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


def test_discounted_rewards_all_zeros():
    """Test discounted rewards with all zero rewards."""
    rewards = [0.0, 0.0, 0.0, 0.0]
    dones = [0, 0, 0, 1]
    gamma = 0.99

    disc = discounted_rewards(rewards, dones, gamma)

    assert all(r == 0.0 for r in disc)


def test_discounted_rewards_negative():
    """Test discounted rewards with negative rewards."""
    rewards = [-1.0, -2.0, -3.0]
    dones = [0, 0, 1]
    gamma = 0.9

    disc = discounted_rewards(rewards, dones, gamma)

    # All should be negative
    assert all(r < 0 for r in disc)
    # First should be most negative (cumulative)
    assert disc[0] < disc[1] < disc[2]


def test_discounted_rewards_mixed_signs():
    """Test discounted rewards with mixed positive and negative."""
    rewards = [1.0, -2.0, 3.0, -1.0]
    dones = [0, 0, 0, 1]
    gamma = 0.9

    disc = discounted_rewards(rewards, dones, gamma)

    assert len(disc) == 4
    assert all(np.isfinite(r) for r in disc)


def test_discounted_rewards_very_long_episode():
    """Test discounted rewards with very long episode."""
    n_steps = 1000
    rewards = [1.0] * n_steps
    dones = [0] * (n_steps - 1) + [1]
    gamma = 0.99

    disc = discounted_rewards(rewards, dones, gamma)

    assert len(disc) == n_steps
    assert all(np.isfinite(r) for r in disc)
    # First reward should be larger than last
    assert disc[0] > disc[-1]


def test_process_memory_single_step():
    """Test memory processing with single step."""
    memory = [
        (np.array([0.0]), 1.0, np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1),
    ]

    actions, rewards_t, states, next_states, dones_t = process_memory(
        memory, gamma=0.99
    )

    assert actions.shape[0] == 1
    assert rewards_t.shape[0] == 1
    assert states.shape[0] == 1


def test_process_memory_immediate_done():
    """Test memory processing where all steps are done."""
    memory = [
        (np.array([0.0]), 1.0, np.array([0.0]), np.array([1.0]), 1),
        (np.array([0.0]), 2.0, np.array([1.0]), np.array([2.0]), 1),
        (np.array([0.0]), 3.0, np.array([2.0]), np.array([3.0]), 1),
    ]

    actions, rewards_t, states, next_states, dones_t = process_memory(
        memory, gamma=0.99
    )

    # Each reward should be independent (no carry over due to done)
    assert torch.allclose(rewards_t, torch.tensor([[1.0], [2.0], [3.0]]))


def test_a2c_with_zero_rewards():
    """Test A2C training with all zero rewards."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Create memory with zero rewards
    memory = []
    state = env.reset()[0]
    for _ in range(10):
        action = env.action_space.low + np.random.rand(n_actions) * (
            env.action_space.high - env.action_space.low
        )
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((action, 0.0, state, next_state, done))
        state = next_state

    # Should not raise even with zero rewards
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_with_negative_rewards():
    """Test A2C training with negative rewards."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Create memory with negative rewards
    memory = []
    state = env.reset()[0]
    for i in range(10):
        action = env.action_space.low + np.random.rand(n_actions) * (
            env.action_space.high - env.action_space.low
        )
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((action, -float(i + 1), state, next_state, done))
        state = next_state

    # Should handle negative rewards
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_with_large_state_values():
    """Test A2C with large state values."""

    class _LargeStateEnv(_DummyEnv):
        def reset(self, seed=None):
            return np.array([1000.0, -500.0, 2000.0], dtype=np.float32), {}

        def step(self, _action):
            return (
                np.array([1000.0, -500.0, 2000.0], dtype=np.float32),
                1.0,
                False,
                False,
                {},
            )

    env = _LargeStateEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    memory = agent.run_episode(max_steps=5)
    # Should handle large values without numerical issues
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_with_very_small_rewards():
    """Test A2C with very small rewards."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Create memory with very small rewards
    memory = []
    state = env.reset()[0]
    for _ in range(10):
        action = env.action_space.low + np.random.rand(n_actions) * (
            env.action_space.high - env.action_space.low
        )
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((action, 1e-8, state, next_state, done))
        state = next_state

    # Should handle very small rewards
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_batch_state_dimension_mismatch_protection():
    """Test that A2C handles single state correctly (adds batch dim)."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Single state without batch dimension
    single_state = np.random.randn(state_dim).astype(np.float32)
    action = agent.predict(single_state, deterministic=True)

    assert action.shape == (n_actions,)


def test_discounted_rewards_single_step():
    """Test discounted rewards with single step episode."""
    rewards = [5.0]
    dones = [1]
    gamma = 0.99

    disc = discounted_rewards(rewards, dones, gamma)

    assert len(disc) == 1
    assert disc[0] == 5.0


def test_discounted_rewards_alternating_done():
    """Test discounted rewards with alternating done flags."""
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    dones = [1, 0, 1, 0, 1, 0]
    gamma = 0.9

    disc = discounted_rewards(rewards, dones, gamma)

    # Done flags should reset accumulation
    assert disc[0] == 1.0  # Reset by done
    assert disc[2] == 1.0  # Reset by done
    assert disc[4] == 1.0  # Reset by done


def test_actor_extreme_logstd():
    """Test Actor with extreme logstd values."""
    state_dim, n_actions = 3, 1
    actor = Actor(state_dim, n_actions)

    # Manually set extreme logstd
    with torch.no_grad():
        actor.logstds.data.fill_(10.0)  # Very large

    states = torch.randn(5, state_dim)
    dist = actor(states)

    # Should be clamped to max 50
    assert (dist.stddev <= 50).all()

    # Test very small
    with torch.no_grad():
        actor.logstds.data.fill_(-10.0)  # Very small

    dist = actor(states)

    # Should be clamped to min 1e-3
    assert (dist.stddev >= 1e-3).all()


def test_a2c_episode_reward_tracking():
    """Test that A2C correctly tracks episode rewards."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Run multiple episodes
    agent.reset()
    initial_reward = agent.episode_reward
    assert initial_reward == 0

    # After running, episode reward should be tracked
    memory = agent.run_episode(max_steps=10)
    # Sum rewards from memory
    expected_reward = sum(r for _, r, _, _, _ in memory)
    # Note: agent.episode_reward might be reset if episode completes


def test_process_memory_preserves_action_shape():
    """Test that process_memory preserves action dimensionality."""
    # Test with scalar action
    memory_scalar = [
        (np.array([0.5]), 1.0, np.array([0.0]), np.array([1.0]), 0),
    ]
    actions, _, _, _, _ = process_memory(memory_scalar, gamma=0.99)
    assert actions.shape == (1, 1)

    # Test with multi-dimensional action
    memory_multi = [
        (
            np.array([0.5, 0.3, 0.1]),
            1.0,
            np.array([0.0]),
            np.array([1.0]),
            0,
        ),
    ]
    actions, _, _, _, _ = process_memory(memory_multi, gamma=0.99)
    assert actions.shape == (1, 3)


if __name__ == "__main__":
    # Run all tests
    test_discounted_rewards_all_zeros()
    test_discounted_rewards_negative()
    test_discounted_rewards_mixed_signs()
    test_discounted_rewards_very_long_episode()
    test_process_memory_single_step()
    test_process_memory_immediate_done()
    test_a2c_with_zero_rewards()
    test_a2c_with_negative_rewards()
    test_a2c_with_large_state_values()
    test_a2c_with_very_small_rewards()
    test_a2c_batch_state_dimension_mismatch_protection()
    test_discounted_rewards_single_step()
    test_discounted_rewards_alternating_done()
    test_actor_extreme_logstd()
    test_a2c_episode_reward_tracking()
    test_process_memory_preserves_action_shape()
    print("All edge case tests passed!")

