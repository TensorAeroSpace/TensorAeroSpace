"""Tests for DQN Agent training and evaluation."""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.dqn import DQNAgent, Model


@pytest.fixture
def dqn_agent_small(dummy_env, seed_all):
    """Create a DQN agent with small buffer for training tests."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=0.1,
        buffer_size=16,
        batch_size=4,
        train_nums=100,
    )
    return agent


def test_train_step_returns_float(dqn_agent_small, seed_all):
    """Test that train_step returns a float loss value."""
    # Fill buffer with enough transitions
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Call train_step
    loss = dqn_agent_small.train_step()

    # Should return a float
    assert isinstance(loss, float)
    assert loss >= 0.0  # Loss should be non-negative


def test_train_step_updates_priorities(dqn_agent_small, seed_all):  # noqa
    """Test that train_step updates leaf priorities in buffer."""
    # Fill buffer
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Get initial leaf priorities
    initial_tree = dqn_agent_small.replay_buffer.tree.copy()

    # Run train_step
    _ = dqn_agent_small.train_step()

    # Some leaf values should have changed
    final_tree = dqn_agent_small.replay_buffer.tree

    # At least one leaf should change (batch_size leaves updated)
    assert not np.allclose(
        initial_tree, final_tree
    ), "Tree priorities should change after train_step"


def test_train_step_updates_model(dqn_agent_small, seed_all):
    """Test that train_step updates model parameters."""
    # Initialize model with forward pass
    obs = np.zeros((1, 4), dtype=np.float32)
    _ = dqn_agent_small.model.predict(obs)

    # Fill buffer
    for i in range(16):
        obs_i = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs_i,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Get initial parameters
    initial_params = [p.clone() for p in dqn_agent_small.model.parameters()]

    # Run train_step
    _ = dqn_agent_small.train_step()

    # Parameters should have changed
    final_params = list(dqn_agent_small.model.parameters())

    params_changed = False
    for init_p, final_p in zip(initial_params, final_params):
        if not torch.allclose(init_p, final_p):
            params_changed = True
            break

    assert params_changed, "Model parameters should update after train_step"


def test_evaluation_returns_float(wrapped_env, seed_all):
    """Test that evaluation returns a float reward."""
    model = Model(num_actions=wrapped_env.env.action_space.n)
    target_model = Model(num_actions=wrapped_env.env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=wrapped_env.env,
        buffer_size=16,
        batch_size=4,
    )

    # Initialize model
    obs, _ = wrapped_env.env.reset()
    _ = agent.model.predict(obs.reshape(1, -1))

    # Run evaluation
    ep_reward = agent.evaluation(wrapped_env, render=False)

    assert isinstance(ep_reward, (float, np.floating))


def test_evaluation_completes_episode(wrapped_env, seed_all):
    """Test that evaluation runs a complete episode."""
    model = Model(num_actions=wrapped_env.env.action_space.n)
    target_model = Model(num_actions=wrapped_env.env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=wrapped_env.env,
        buffer_size=16,
        batch_size=4,
    )

    # Initialize model
    obs, _ = wrapped_env.env.reset()
    _ = agent.model.predict(obs.reshape(1, -1))

    # Run evaluation
    ep_reward = agent.evaluation(wrapped_env, render=False)

    # Should complete without errors and return some reward
    assert isinstance(ep_reward, (float, np.floating))


def test_evaluation_with_render_flag(wrapped_env, seed_all):
    """Test that evaluation works with render=True (no actual rendering needed)."""
    model = Model(num_actions=wrapped_env.env.action_space.n)
    target_model = Model(num_actions=wrapped_env.env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=wrapped_env.env,
        buffer_size=16,
        batch_size=4,
    )

    # Initialize model
    obs, _ = wrapped_env.env.reset()
    _ = agent.model.predict(obs.reshape(1, -1))

    # Run evaluation with render=True
    # (dummy env doesn't actually render, but should not crash)
    ep_reward = agent.evaluation(wrapped_env, render=True)

    assert isinstance(ep_reward, (float, np.floating))


def test_multiple_train_steps(dqn_agent_small, seed_all):
    """Test that multiple train_steps can be run consecutively."""
    # Fill buffer
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    # Run multiple train steps
    losses = []
    for _ in range(5):
        loss = dqn_agent_small.train_step()
        losses.append(loss)

    # All should return valid losses
    assert len(losses) == 5
    assert all(isinstance(loss, float) and loss >= 0 for loss in losses)


def test_train_step_increments_global_step(dqn_agent_small, seed_all):
    """Test that train_step increments global_step counter."""
    # Fill buffer
    for i in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=float(i),
            next_state=next_obs,
            done=False,
        )

    initial_step = dqn_agent_small.global_step

    # Run train step
    _ = dqn_agent_small.train_step()

    assert dqn_agent_small.global_step == initial_step + 1


def test_train_step_with_varying_rewards(dqn_agent_small, seed_all):
    """Test train_step with transitions having varying rewards."""
    # Fill buffer with varying rewards
    rewards = [0.0, 0.5, 1.0, -1.0, 2.0, -0.5, 1.5, 0.3]
    for i, r in enumerate(rewards * 2):  # Fill to 16
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        dqn_agent_small.store_transition(
            priority=1.0,
            obs=obs,
            action=i % 2,
            reward=r,
            next_state=next_obs,
            done=(i % 5 == 0),
        )

    # Should handle varying rewards
    loss = dqn_agent_small.train_step()
    assert isinstance(loss, float)
    assert loss >= 0.0
