"""Comprehensive tests for DDPG auxiliary components.

Tests for RunningMeanStd, ReplayBuffer, and OUNoise classes.
"""

import numpy as np
import pytest

from tensoraerospace.agent.ddpg.model import OUNoise, ReplayBuffer, RunningMeanStd


class TestRunningMeanStd:
    """Tests for RunningMeanStd class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        rms = RunningMeanStd()
        assert rms.mean.shape == ()
        assert rms.var.shape == ()
        assert rms.count > 0

    def test_init_with_shape(self):
        """Test initialization with custom shape."""
        shape = (3,)
        rms = RunningMeanStd(shape=shape)
        assert rms.mean.shape == shape
        assert rms.var.shape == shape
        assert np.allclose(rms.mean, 0.0)
        assert np.allclose(rms.var, 1.0)

    def test_update_single_batch(self):
        """Test updating statistics with a single batch."""
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        rms.update(data)

        expected_mean = np.mean(data, axis=0)
        assert rms.mean.shape == (2,)
        assert np.allclose(rms.mean, expected_mean, atol=0.1)

    def test_update_multiple_batches(self):
        """Test updating statistics with multiple batches."""
        rms = RunningMeanStd(shape=(1,))

        batch1 = np.array([[1.0], [2.0], [3.0]])
        batch2 = np.array([[4.0], [5.0], [6.0]])

        rms.update(batch1)
        rms.update(batch2)

        all_data = np.concatenate([batch1, batch2])
        expected_mean = np.mean(all_data)

        assert np.allclose(rms.mean, expected_mean, atol=0.1)

    def test_normalize(self):
        """Test normalization of data."""
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        rms.update(data)

        normalized = rms.normalize(np.array([3.0, 4.0]))

        # Normalized data should have approximately zero mean
        assert normalized.shape == (2,)
        assert not np.allclose(normalized, [3.0, 4.0])

    def test_normalize_with_epsilon(self):
        """Test normalization with custom epsilon."""
        rms = RunningMeanStd(shape=(1,))
        rms.mean = np.array([5.0])
        rms.var = np.array([0.0])  # Zero variance

        # Should not raise division by zero due to epsilon
        normalized = rms.normalize(np.array([5.0]), epsilon=1e-8)
        assert np.isfinite(normalized).all()

    def test_state_dict(self):
        """Test state dictionary serialization."""
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        rms.update(data)

        state = rms.state_dict()

        assert "mean" in state
        assert "var" in state
        assert "count" in state
        # mean is converted to list for JSON serialization
        assert isinstance(state["mean"], list) and len(state["mean"]) == 2

    def test_load_state_dict(self):
        """Test loading from state dictionary."""
        rms1 = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        rms1.update(data)

        state = rms1.state_dict()

        rms2 = RunningMeanStd(shape=(2,))
        rms2.load_state_dict(state)

        assert np.allclose(rms2.mean, rms1.mean)
        assert np.allclose(rms2.var, rms1.var)
        assert rms2.count == rms1.count

    def test_update_from_moments(self):
        """Test direct update from moments."""
        rms = RunningMeanStd(shape=(1,))

        batch_mean = np.array([2.0])
        batch_var = np.array([1.0])
        batch_count = 10

        rms.update_from_moments(batch_mean, batch_var, batch_count)

        assert rms.count > 10


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_init(self):
        """Test initialization."""
        capacity = 100
        buf = ReplayBuffer(capacity)

        assert buf.capacity == capacity
        assert len(buf) == 0
        assert buf.position == 0

    def test_push_single(self):
        """Test pushing a single transition."""
        buf = ReplayBuffer(10)
        state = np.array([1.0, 2.0])
        action = np.array([0.5])

        buf.push(state, action, 1.0, state, False)

        assert len(buf) == 1

    def test_push_until_full(self):
        """Test pushing until buffer is full."""
        capacity = 5
        buf = ReplayBuffer(capacity)

        for i in range(capacity):
            buf.push(
                np.array([float(i)]),
                np.array([0.0]),
                1.0,
                np.array([float(i + 1)]),
                False,
            )

        assert len(buf) == capacity

    def test_circular_overwrite(self):
        """Test that buffer overwrites old data when full."""
        capacity = 3
        buf = ReplayBuffer(capacity)

        # Fill buffer
        for i in range(capacity):
            buf.push(
                np.array([float(i)]), np.array([0.0]), float(i), np.array([0.0]), False
            )

        # Overwrite first element
        buf.push(np.array([999.0]), np.array([0.0]), 999.0, np.array([0.0]), False)

        assert len(buf) == capacity
        assert buf.position == 1

    def test_sample_basic(self):
        """Test basic sampling."""
        buf = ReplayBuffer(10)

        for i in range(5):
            buf.push(np.array([float(i)]), np.array([0.0]), 1.0, np.array([0.0]), False)

        states, actions, rewards, next_states, dones = buf.sample(3)

        assert states.shape == (3, 1)
        assert actions.shape == (3, 1)
        assert rewards.shape == (3,)
        assert next_states.shape == (3, 1)
        assert dones.shape == (3,)

    def test_sample_error_insufficient_data(self):
        """Test that sampling more than available raises error."""
        buf = ReplayBuffer(10)

        buf.push(np.array([1.0]), np.array([0.0]), 1.0, np.array([0.0]), False)

        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5)

    def test_state_dict_serialization(self):
        """Test state dict serialization."""
        buf = ReplayBuffer(10)

        for i in range(3):
            buf.push(np.array([float(i)]), np.array([0.0]), 1.0, np.array([0.0]), False)

        state = buf.state_dict()

        assert state["capacity"] == 10
        assert len(state["buffer"]) == 3
        assert state["position"] == 3

    def test_load_state_dict(self):
        """Test loading from state dict."""
        buf1 = ReplayBuffer(10)

        for i in range(3):
            buf1.push(
                np.array([float(i)]), np.array([0.0]), 1.0, np.array([0.0]), False
            )

        state = buf1.state_dict()

        buf2 = ReplayBuffer(5)
        buf2.load_state_dict(state)

        assert buf2.capacity == 10
        assert len(buf2) == 3
        assert buf2.position == 3


class TestOUNoise:
    """Tests for OUNoise class."""

    class _DummyActionSpace:
        def __init__(self, shape, low=-1.0, high=1.0):
            self.shape = shape
            self.low = np.full(shape, low, dtype=np.float32)
            self.high = np.full(shape, high, dtype=np.float32)

    def test_init(self):
        """Test initialization."""
        space = self._DummyActionSpace((2,), low=-1.0, high=1.0)
        ou = OUNoise(space)

        assert ou.action_dim == 2
        assert ou.mu == 0.0
        assert ou.theta == 0.15
        assert ou.state.shape == (2,)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        space = self._DummyActionSpace((1,))
        ou = OUNoise(
            space, mu=0.5, theta=0.2, max_sigma=0.5, min_sigma=0.1, decay_period=50000
        )

        assert ou.mu == 0.5
        assert ou.theta == 0.2
        assert ou.max_sigma == 0.5
        assert ou.min_sigma == 0.1
        assert ou.decay_period == 50000

    def test_reset(self):
        """Test state reset."""
        space = self._DummyActionSpace((2,))
        ou = OUNoise(space, mu=0.5)

        # Evolve state
        ou.evolve_state()
        ou.evolve_state()

        # Reset should return to mu
        ou.reset()
        assert np.allclose(ou.state, 0.5)

    def test_evolve_state(self):
        """Test state evolution."""
        space = self._DummyActionSpace((2,))
        ou = OUNoise(space)

        initial_state = ou.state.copy()
        new_state = ou.evolve_state()

        assert new_state.shape == (2,)
        # State should have changed (with very high probability)
        assert not np.allclose(new_state, initial_state)

    def test_get_action_clipping(self):
        """Test that get_action clips to action space bounds."""
        space = self._DummyActionSpace((1,), low=-0.5, high=0.5)
        ou = OUNoise(space, max_sigma=10.0)  # Large noise

        action = np.array([0.0])
        noisy_action = ou.get_action(action, t=0)

        assert np.all(noisy_action >= space.low)
        assert np.all(noisy_action <= space.high)

    def test_sigma_decay(self):
        """Test that sigma decays over time."""
        space = self._DummyActionSpace((1,))
        ou = OUNoise(space, max_sigma=1.0, min_sigma=0.1, decay_period=100)

        action = np.array([0.0])

        # Early timestep
        ou.get_action(action, t=0)
        sigma_early = ou.sigma

        # Late timestep
        ou.get_action(action, t=100)
        sigma_late = ou.sigma

        assert sigma_early > sigma_late
        assert np.isclose(sigma_late, 0.1, atol=0.01)

    def test_state_dict(self):
        """Test state dict serialization."""
        space = self._DummyActionSpace((2,), low=-1.0, high=1.0)
        ou = OUNoise(space, mu=0.5, theta=0.2)

        state = ou.state_dict()

        assert state["mu"] == 0.5
        assert state["theta"] == 0.2
        assert state["action_dim"] == 2
        assert "state" in state

    def test_load_state_dict(self):
        """Test loading from state dict."""
        space = self._DummyActionSpace((2,))
        ou1 = OUNoise(space, mu=0.5)
        ou1.evolve_state()

        state = ou1.state_dict()

        ou2 = OUNoise(space)
        ou2.load_state_dict(state)

        assert ou2.mu == ou1.mu
        assert ou2.theta == ou1.theta
        assert np.allclose(ou2.state, ou1.state)

    def test_deterministic_with_seed(self):
        """Test reproducibility with random seed."""
        np.random.seed(42)
        space = self._DummyActionSpace((2,))
        ou1 = OUNoise(space)
        ou1.reset()
        state1 = ou1.evolve_state()

        np.random.seed(42)
        ou2 = OUNoise(space)
        ou2.reset()
        state2 = ou2.evolve_state()

        assert np.allclose(state1, state2)
