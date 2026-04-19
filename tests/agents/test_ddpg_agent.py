"""Comprehensive tests for DDPG agent core functionality.

Tests for DDPG agent initialization, training, and update methods.
"""

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ddpg.model import DDPG


class _DummySpace:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _FakeEnv:
    """Minimal fake environment for testing."""

    def __init__(self, obs_dim=3, act_dim=1):
        self.observation_space = _DummySpace((obs_dim,))
        self.action_space = _DummySpace((act_dim,), low=-2.0, high=2.0)
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        state = np.random.randn(self.observation_space.shape[0]).astype(np.float32)
        return state, {}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)
        next_state = np.random.randn(self.observation_space.shape[0]).astype(np.float32)
        reward = float(1.0 - 0.1 * np.linalg.norm(action))
        terminated = self._step_count >= 10
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info


class TestDDPGInit:
    """Tests for DDPG initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        env = _FakeEnv(obs_dim=4, act_dim=2)
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1000)

        assert agent.env is env
        assert agent.value_lr == 1e-3
        assert agent.policy_lr == 1e-4
        assert agent.replay_buffer_size == 1000
        assert agent.state_dim == 4
        assert agent.action_dim == 2

    def test_init_with_normalization(self):
        """Test initialization with observation normalization enabled."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        assert agent.normalize_observations is True
        assert agent.obs_rms is not None
        assert agent.obs_rms.mean.shape == (env.observation_space.shape[0],)

    def test_init_without_normalization(self):
        """Test initialization without observation normalization."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=False,
        )

        assert agent.normalize_observations is False
        assert agent.obs_rms is None

    def test_networks_created(self):
        """Test that all networks are created."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        assert agent.value_net is not None
        assert agent.policy_net is not None
        assert agent.target_value_net is not None
        assert agent.target_policy_net is not None

    def test_optimizers_created(self):
        """Test that optimizers are created."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=100)

        assert agent.value_optimizer is not None
        assert agent.policy_optimizer is not None

        # Check learning rates
        assert agent.value_optimizer.param_groups[0]["lr"] == 1e-3
        assert agent.policy_optimizer.param_groups[0]["lr"] == 1e-4

    def test_target_networks_initialized(self):
        """Test that target networks match main networks initially."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Target networks should have same weights as main networks
        for target_param, param in zip(
            agent.target_value_net.parameters(), agent.value_net.parameters()
        ):
            assert torch.allclose(target_param.data, param.data)

        for target_param, param in zip(
            agent.target_policy_net.parameters(), agent.policy_net.parameters()
        ):
            assert torch.allclose(target_param.data, param.data)

    def test_action_scaling_setup(self):
        """Test that action scaling is properly set up."""
        env = _FakeEnv(act_dim=2)
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Check that policy network has correct action bounds
        assert hasattr(agent.policy_net, "action_scale")
        assert hasattr(agent.policy_net, "action_bias")


class TestDDPGNormalization:
    """Tests for observation normalization."""

    def test_normalize_observation_enabled(self):
        """Test normalization when enabled."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        # Update statistics
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agent.obs_rms.update(data)

        obs = np.array([2.5, 3.5, 4.5])
        normalized = agent._normalize_observation(obs)

        # Should be different from original
        assert not np.allclose(normalized, obs)

    def test_normalize_observation_disabled(self):
        """Test that normalization returns original when disabled."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=False,
        )

        obs = np.array([1.0, 2.0, 3.0])
        normalized = agent._normalize_observation(obs)

        # Should be identical to original
        assert np.allclose(normalized, obs)


class TestDDPGUpdate:
    """Tests for DDPG update method."""

    def test_ddpg_update_basic(self):
        """Test basic update step."""
        env = _FakeEnv(obs_dim=3, act_dim=1)
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)

        # Fill replay buffer
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        # Should run without error
        agent.ddpg_update(batch_size=16)

    def test_ddpg_update_changes_weights(self):
        """Test that update actually changes network weights."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)

        # Fill buffer
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        # Store initial weights
        initial_value_weight = agent.value_net.linear1.weight.data.clone()
        initial_policy_weight = agent.policy_net.linear1.weight.data.clone()

        # Perform update
        agent.ddpg_update(batch_size=16)

        # Weights should have changed
        assert not torch.allclose(
            initial_value_weight, agent.value_net.linear1.weight.data
        )
        assert not torch.allclose(
            initial_policy_weight, agent.policy_net.linear1.weight.data
        )

    def test_ddpg_update_target_networks(self):
        """Test that target networks are updated with soft update."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)

        # Fill buffer
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        # Store initial target weights
        initial_target_weight = agent.target_value_net.linear1.weight.data.clone()

        # Perform update with soft_tau
        agent.ddpg_update(batch_size=16, soft_tau=0.01)

        # Target weights should have changed slightly
        assert not torch.allclose(
            initial_target_weight, agent.target_value_net.linear1.weight.data
        )

    def test_ddpg_update_with_clipping(self):
        """Test update with Q-value clipping."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)

        # Fill buffer
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 100.0, s, False)  # Large reward

        # Should run without error even with clipping
        agent.ddpg_update(batch_size=16, min_value=-10.0, max_value=10.0)

    def test_ddpg_update_gamma(self):
        """Test update with different gamma values."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)

        # Fill buffer
        for _ in range(32):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        # Test with different gamma values
        for gamma in [0.9, 0.95, 0.99]:
            agent.ddpg_update(batch_size=16, gamma=gamma)


class TestDDPGLearn:
    """Tests for DDPG learn method."""

    def test_learn_smoke(self):
        """Test that learning runs without error."""
        env = _FakeEnv(obs_dim=3, act_dim=1)
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Short training run
        agent.learn(max_frames=50, max_steps=10, batch_size=8, warmup_frames=10)

        # Check that training metrics are recorded
        assert len(agent.rewards) > 0
        assert agent.frame_idx > 0

    def test_learn_warmup(self):
        """Test that warmup period is respected."""
        from tensoraerospace.agent.metrics import MetricWriter

        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)
        # Inject a writer with no mandatory-metric contract: this test only
        # collects warmup transitions, so train/updates and train/lr will never
        # be written and the default contract check would fail.
        agent.writer = MetricWriter(required=())

        initial_weight = agent.value_net.linear1.weight.data.clone()

        # Train only during warmup (no updates should happen)
        agent.learn(
            max_frames=20,
            max_steps=10,
            batch_size=8,
            warmup_frames=100,  # Longer than max_frames
        )

        # Weights should not change during warmup
        assert torch.allclose(initial_weight, agent.value_net.linear1.weight.data)

    def test_learn_with_normalization(self):
        """Test learning with observation normalization."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        initial_count = agent.obs_rms.count

        agent.learn(max_frames=50, max_steps=10, batch_size=8, warmup_frames=10)

        # Normalization statistics should be updated
        assert agent.obs_rms.count > initial_count

    def test_learn_records_rewards(self):
        """Test that episode rewards are recorded."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        agent.learn(max_frames=50, max_steps=10, batch_size=8, warmup_frames=10)

        assert len(agent.rewards) > 0
        assert all(isinstance(r, (int, float)) for r in agent.rewards)

    def test_learn_updates_per_step(self):
        """Test learning with multiple updates per step."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Should run without error
        agent.learn(
            max_frames=50,
            max_steps=10,
            batch_size=8,
            warmup_frames=10,
            updates_per_step=3,
        )

    def test_learn_respects_max_frames(self):
        """Test that training stops at max_frames."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        max_frames = 30
        agent.learn(max_frames=max_frames, max_steps=10, batch_size=8, warmup_frames=5)

        # Should stop at or slightly after max_frames
        assert agent.frame_idx >= max_frames
        assert agent.frame_idx < max_frames + 20  # Allow small overshoot


class TestDDPGCollectGrads:
    """Tests for gradient collection method."""

    def test_collect_grads_basic(self):
        """Test basic gradient collection."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Get device from model to ensure tensors are on the same device
        device = next(agent.value_net.parameters()).device

        # Perform a dummy forward-backward pass
        state = torch.randn(4, 3, device=device)
        action = torch.randn(4, 1, device=device)
        q_values = agent.value_net(state, action)
        loss = q_values.mean()
        loss.backward()

        grads = agent._collect_grads(agent.value_net)

        assert isinstance(grads, dict)
        assert len(grads) > 0
        assert all(isinstance(k, str) for k in grads.keys())

    def test_collect_grads_no_gradients(self):
        """Test collecting gradients when none exist."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # No backward pass, so no gradients
        grads = agent._collect_grads(agent.policy_net)

        assert isinstance(grads, dict)
        # All gradients should be None
        assert all(v is None for v in grads.values())
