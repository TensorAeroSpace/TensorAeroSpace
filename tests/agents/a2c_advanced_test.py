"""Advanced tests for A2C agent covering edge cases and detailed functionality.

This module contains advanced tests for A2C agent covering:
- Multiple episode training
- State normalization
- Different activation functions
- Edge cases
"""

import numpy as np
import torch
import torch.nn as nn

from tensoraerospace.agent.a2c.model import (
    A2C,
    A2CWithNARXCritic,
    Actor,
    Critic,
    Mish,
    discounted_rewards,
    mish,
    process_memory,
    to_tensor,
)


class _DummyEnv:
    """Minimal environment for testing."""

    def __init__(self):
        self.observation_space = type("S", (), {"shape": (4,)})
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -2.0]),
                "high": np.array([1.0, 2.0]),
            },
        )
        self.unwrapped = self
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.random.randn(4).astype(np.float32), {}

    def step(self, action):
        self.step_count += 1
        reward = -np.sum(action**2)  # Simple quadratic reward
        done = self.step_count >= 10
        return (
            np.random.randn(4).astype(np.float32),
            float(reward),
            done,
            False,
            {},
        )


def test_mish_activation():
    """Test Mish activation function."""
    x = torch.randn(10, 5)
    y = mish(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    # Test that Mish is smooth and non-linear
    x_zero = torch.zeros(1)
    y_zero = mish(x_zero)
    assert torch.isclose(y_zero, torch.zeros(1), atol=1e-4)


def test_mish_module():
    """Test Mish as a PyTorch module."""
    mish_layer = Mish()
    x = torch.randn(5, 10)
    y = mish_layer(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_to_tensor_conversion():
    """Test to_tensor helper function."""
    # Test numpy array
    arr = np.array([1.0, 2.0, 3.0])
    tensor = to_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3,)

    # Test list
    lst = [1.0, 2.0, 3.0]
    tensor = to_tensor(lst)
    assert isinstance(tensor, torch.Tensor)

    # Test device placement
    tensor_cpu = to_tensor(arr, device="cpu")
    assert tensor_cpu.device.type == "cpu"


def test_actor_with_different_activations():
    """Test Actor with different activation functions."""
    state_dim, n_actions = 4, 2

    # Test with Tanh
    actor_tanh = Actor(state_dim, n_actions, activation=nn.Tanh)
    states = torch.randn(5, state_dim)
    dist = actor_tanh(states)
    assert isinstance(dist, torch.distributions.Normal)

    # Test with ReLU
    actor_relu = Actor(state_dim, n_actions, activation=nn.ReLU)
    dist = actor_relu(states)
    assert isinstance(dist, torch.distributions.Normal)

    # Test with Mish
    actor_mish = Actor(state_dim, n_actions, activation=Mish)
    dist = actor_mish(states)
    assert isinstance(dist, torch.distributions.Normal)


def test_critic_value_estimation():
    """Test Critic value estimation."""
    state_dim = 4
    critic = Critic(state_dim)

    states = torch.randn(10, state_dim)
    values = critic(states)

    assert values.shape == (10, 1)
    assert torch.isfinite(values).all()


def test_discounted_rewards_no_discount():
    """Test discounted rewards with gamma=1.0."""
    rewards = [1.0, 1.0, 1.0]
    dones = [0, 0, 1]
    gamma = 1.0

    disc = discounted_rewards(rewards, dones, gamma)

    # With gamma=1, cumulative sum
    assert abs(disc[0] - 3.0) < 0.01
    assert abs(disc[1] - 2.0) < 0.01
    assert abs(disc[2] - 1.0) < 0.01


def test_discounted_rewards_zero_gamma():
    """Test discounted rewards with gamma=0.0."""
    rewards = [1.0, 2.0, 3.0]
    dones = [0, 0, 1]
    gamma = 0.0

    disc = discounted_rewards(rewards, dones, gamma)

    # With gamma=0, immediate rewards only
    assert abs(disc[0] - 1.0) < 0.01
    assert abs(disc[1] - 2.0) < 0.01
    assert abs(disc[2] - 3.0) < 0.01


def test_process_memory_without_discount():
    """Test memory processing without reward discounting."""
    memory = [
        (np.array([0.5]), 1.0, np.array([1.0, 2.0]), np.array([2.0, 3.0]), 0),
        (np.array([0.3]), 0.5, np.array([2.0, 3.0]), np.array([3.0, 4.0]), 1),
    ]

    actions, rewards_t, states, next_states, dones_t = process_memory(
        memory, gamma=0.9, discount_rewards=False
    )

    assert actions.shape[0] == 2
    # Without discounting, rewards stay as is
    assert torch.allclose(rewards_t, torch.tensor([[1.0], [0.5]]))


def test_a2c_multi_episode_training():
    """Test A2C training over multiple episodes."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic, gamma=0.99)

    # Train for multiple episodes
    total_rewards = []
    for episode in range(5):
        memory = agent.run_episode(max_steps=20)
        agent.learn(memory, steps=episode, discount_rewards=True)
        episode_reward = sum(r for _, r, _, _, _ in memory)
        total_rewards.append(episode_reward)

    # Check that we completed training
    assert len(total_rewards) == 5


def test_a2c_predict_consistency():
    """Test that A2C predict is consistent in deterministic mode."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    state = np.random.randn(state_dim).astype(np.float32)

    # Deterministic predictions should be the same
    action1 = agent.predict(state, deterministic=True)
    action2 = agent.predict(state, deterministic=True)

    assert np.allclose(action1, action2)


def test_a2c_eval_train_mode():
    """Test A2C eval and train mode switching."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Set to eval mode
    agent.set_eval_mode()
    assert not agent.actor.training
    assert not agent.critic.training

    # Set to train mode
    agent.set_train_mode()
    assert agent.actor.training
    assert agent.critic.training


def test_a2c_with_narx_history_tracking():
    """Test A2CWithNARXCritic history tracking."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    from tensoraerospace.agent.a2c.narx_critic import NARXCritic

    actor = Actor(state_dim, n_actions)
    critic = NARXCritic(state_dim, n_actions, history_length=3)
    agent = A2CWithNARXCritic(
        env=env, actor=actor, critic=critic, history_length=3, gamma=0.99
    )

    # Run episode and check that history is tracked
    memory = agent.run_episode(max_steps=10)

    assert len(memory) == 10
    # Memory should contain full episodes
    for entry in memory:
        assert len(entry) == 5  # (action, reward, state, next_state, done)


def test_a2c_action_clipping():
    """Test that A2C properly clips actions to action space bounds."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    state = np.random.randn(state_dim).astype(np.float32)

    # Get action
    action = agent.predict(state, deterministic=False)

    # Check that action is within bounds
    assert np.all(action >= env.action_space.low)
    assert np.all(action <= env.action_space.high)


def test_a2c_reset_functionality():
    """Test A2C reset method."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Reset
    agent.reset()

    assert agent.state is not None
    assert not agent.done
    assert agent.episode_reward == 0


def test_a2c_learn_with_small_batch():
    """Test A2C learning with very small batch."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Very small batch
    memory = agent.run_episode(max_steps=2)
    agent.learn(memory, steps=1, discount_rewards=True)


def test_a2c_different_state_action_dims():
    """Test A2C with different state and action dimensions."""
    class _CustomEnv(_DummyEnv):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.observation_space = type("S", (), {"shape": (state_dim,)})
            self.action_space = type(
                "A",
                (),
                {
                    "shape": (action_dim,),
                    "low": np.full(action_dim, -1.0),
                    "high": np.full(action_dim, 1.0),
                },
            )

        def reset(self, seed=None):
            return (
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                {},
            )

        def step(self, action):
            return (
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

    # Test with various dimensions
    for state_dim, action_dim in [(2, 1), (5, 3), (10, 5)]:
        env = _CustomEnv(state_dim, action_dim)
        actor = Actor(state_dim, action_dim)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic)

        memory = agent.run_episode(max_steps=3)
        agent.learn(memory, steps=1, discount_rewards=True)


def test_actor_logstd_bounds():
    """Test that Actor logstd is properly bounded."""
    state_dim, n_actions = 4, 2
    actor = Actor(state_dim, n_actions)

    states = torch.randn(10, state_dim)
    dist = actor(states)

    # Check that std is within reasonable bounds (1e-3 to 50)
    assert (dist.stddev >= 1e-3).all()
    assert (dist.stddev <= 50).all()


def test_process_memory_batch_consistency():
    """Test that process_memory produces consistent batch sizes."""
    # Create memory with varying sizes
    memory = []
    for i in range(10):
        action = np.random.randn(2)
        state = np.random.randn(3)
        next_state = np.random.randn(3)
        reward = float(i)
        done = i == 9

        memory.append((action, reward, state, next_state, done))

    actions, rewards_t, states, next_states, dones_t = process_memory(
        memory, gamma=0.99
    )

    # All tensors should have same batch size
    batch_size = len(memory)
    assert actions.shape[0] == batch_size
    assert rewards_t.shape[0] == batch_size
    assert states.shape[0] == batch_size
    assert next_states.shape[0] == batch_size
    assert dones_t.shape[0] == batch_size


if __name__ == "__main__":
    # Run all tests
    test_mish_activation()
    test_mish_module()
    test_to_tensor_conversion()
    test_actor_with_different_activations()
    test_critic_value_estimation()
    test_discounted_rewards_no_discount()
    test_discounted_rewards_zero_gamma()
    test_process_memory_without_discount()
    test_a2c_multi_episode_training()
    test_a2c_predict_consistency()
    test_a2c_eval_train_mode()
    test_a2c_with_narx_history_tracking()
    test_a2c_action_clipping()
    test_a2c_reset_functionality()
    test_a2c_learn_with_small_batch()
    test_a2c_different_state_action_dims()
    test_actor_logstd_bounds()
    test_process_memory_batch_consistency()
    print("All advanced tests passed!")

