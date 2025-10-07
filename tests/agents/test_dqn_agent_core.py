"""Tests for DQN Agent core functionality."""
import numpy as np
import pytest
import torch

from tensoraerospace.agent.dqn import DQNAgent, Model


@pytest.fixture
def dqn_agent(dummy_env, seed_all):
    """Create a DQN agent for testing."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=0.5,
        epsilon_dacay=0.9,
        buffer_size=16,
        batch_size=4,
        train_nums=100,
    )
    return agent


def test_agent_initialization(dummy_env, seed_all):
    """Test DQN Agent initialization."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=0.1,
        buffer_size=16,
        batch_size=4,
    )

    assert agent.epsilon == 0.1
    assert agent.buffer_size == 16
    assert agent.batch_size == 4
    assert agent.model is not None
    assert agent.target_model is not None
    assert agent.optimizer is not None


def test_get_action_exploit(dummy_env, seed_all):
    """Test get_action with epsilon=0 (pure exploitation)."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=0.0,  # No exploration
        buffer_size=16,
        batch_size=4,
    )

    # Initialize model
    obs, _ = dummy_env.reset()
    obs_batch = obs.reshape(1, -1)
    best_action, _ = agent.model.action_value(obs_batch)

    # With epsilon=0, should always return best_action
    for _ in range(10):
        action = agent.get_action(best_action)
        assert action == best_action


def test_get_action_explore(dummy_env, seed_all):  # noqa: ARG001
    """Test get_action with epsilon=1 (pure exploration)."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=1.0,  # Always explore
        buffer_size=16,
        batch_size=4,
    )

    best_action = 0

    # With epsilon=1, should sample from action space
    actions = []
    for _ in range(20):
        action = agent.get_action(best_action)
        actions.append(action)
        assert 0 <= action < dummy_env.action_space.n

    # With enough samples and epsilon=1, we should see some variety
    assert all(0 <= a < dummy_env.action_space.n for a in actions)


def test_e_decay_decreases_epsilon(dqn_agent):
    """Test that e_decay strictly decreases epsilon."""
    initial_epsilon = dqn_agent.epsilon
    epsilons = [initial_epsilon]

    # Decay multiple times
    for _ in range(5):
        dqn_agent.e_decay()
        epsilons.append(dqn_agent.epsilon)

    # Check monotonic decrease
    for i in range(len(epsilons) - 1):
        assert epsilons[i] > epsilons[i + 1], "Epsilon should decrease monotonically"

    # Check epsilon is positive
    assert dqn_agent.epsilon > 0


def test_e_decay_multiple_calls(dqn_agent):
    """Test that e_decay works correctly over multiple calls."""
    epsilon_decay = dqn_agent.epsilon_decay
    initial_epsilon = dqn_agent.epsilon

    # Decay 3 times
    for _ in range(3):
        dqn_agent.e_decay()

    expected_epsilon = initial_epsilon * (epsilon_decay**3)
    assert np.isclose(dqn_agent.epsilon, expected_epsilon)


def test_update_target_model(dqn_agent, seed_all):
    """Test that update_target_model copies parameters correctly."""
    # Initialize models with a forward pass
    obs = np.zeros((1, 4), dtype=np.float32)
    _ = dqn_agent.model.predict(obs)
    _ = dqn_agent.target_model.predict(obs)

    # Modify model weights (simulate training)
    for param in dqn_agent.model.parameters():
        param.data += 0.1

    # Target model should be different before update
    model_params = list(dqn_agent.model.parameters())
    target_params = list(dqn_agent.target_model.parameters())

    # At least one parameter should differ
    params_differ = any(
        not torch.allclose(mp, tp) for mp, tp in zip(model_params, target_params)
    )
    assert params_differ, "Model and target should differ before update"

    # Update target model
    dqn_agent.update_target_model()

    # After update, all parameters should be equal
    model_state = dqn_agent.model.state_dict()
    target_state = dqn_agent.target_model.state_dict()

    for key in model_state:
        assert torch.allclose(
            model_state[key], target_state[key]
        ), f"Parameter {key} not equal after update"


def test_get_target_value_shape(dqn_agent, seed_all):
    """Test that get_target_value returns correct shape."""
    obs = np.random.randn(4, 4).astype(np.float32)
    q_values = dqn_agent.get_target_value(obs)

    assert q_values.shape == (4, dqn_agent.env.action_space.n)
    assert isinstance(q_values, np.ndarray)


def test_store_transition(dqn_agent, seed_all):
    """Test that store_transition adds to replay buffer."""
    obs = np.random.randn(4).astype(np.float32)
    action = 0
    reward = 1.0
    next_obs = np.random.randn(4).astype(np.float32)
    done = False

    initial_idx = dqn_agent.replay_buffer.next_idx
    dqn_agent.store_transition(1.0, obs, action, reward, next_obs, done)

    # next_idx should have advanced
    assert dqn_agent.replay_buffer.next_idx == (initial_idx + 1) % dqn_agent.buffer_size


def test_sum_tree_sample_populates_batch_arrays(dqn_agent, seed_all):
    """Test that sum_tree_sample populates internal batch arrays correctly."""
    # Fill buffer with some transitions
    for i in range(8):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Sample batch
    k = 4
    idxes, is_weights = dqn_agent.sum_tree_sample(k)

    # Check returned values
    assert len(idxes) == k
    assert is_weights.shape == (k, 1)

    # Check batch arrays are populated
    assert dqn_agent.b_obs.shape == (dqn_agent.batch_size, 4)
    assert dqn_agent.b_actions.shape == (dqn_agent.batch_size,)
    assert dqn_agent.b_rewards.shape == (dqn_agent.batch_size,)
    assert dqn_agent.b_next_states.shape == (dqn_agent.batch_size, 4)
    assert dqn_agent.b_dones.shape == (dqn_agent.batch_size,)


def test_sum_tree_sample_importance_weights(dqn_agent, seed_all):
    """Test that importance sampling weights are in valid range."""
    # Fill buffer
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent.store_transition(
            priority=np.random.uniform(0.1, 2.0),
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Sample multiple times
    k = 4
    for _ in range(5):
        idxes, is_weights = dqn_agent.sum_tree_sample(k)

        # IS weights should be in (0, 1] after normalization
        assert np.all(is_weights > 0)
        assert np.all(is_weights <= 1.0)


def test_beta_increases_with_sampling(dqn_agent, seed_all):
    """Test that beta increases and caps at 1.0."""
    # Fill buffer
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    initial_beta = dqn_agent.beta

    # Sample multiple times
    for _ in range(10):
        _ = dqn_agent.sum_tree_sample(4)

    # Beta should have increased
    assert dqn_agent.beta >= initial_beta

    # Beta should not exceed 1.0
    assert dqn_agent.beta <= 1.0

    # Sample many more times to ensure it caps at 1.0
    for _ in range(1000):
        _ = dqn_agent.sum_tree_sample(4)

    assert dqn_agent.beta == 1.0


def test_agent_close(dqn_agent):
    """Test that close() can be called without error."""
    # Should not raise any exception
    dqn_agent.close()

    # Writer should be closed
    # We can't directly check if closed, but calling close again should be safe
    dqn_agent.close()
