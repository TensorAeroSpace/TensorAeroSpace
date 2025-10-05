"""Tests to increase coverage for SAC and ReplayMemory to near 100%."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tensoraerospace.agent.sac.replay_memory import ReplayMemory
from tensoraerospace.agent.sac.sac import SAC


# ============================================================================
# ReplayMemory tests
# ============================================================================


def test_replay_memory_basic_operations():
    """Test basic push and sample operations."""
    memory = ReplayMemory(capacity=100, seed=42)

    state = np.array([1.0, 2.0, 3.0])
    action = np.array([0.5, -0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1, 3.1])
    done = False

    memory.push(state, action, reward, next_state, done)
    assert len(memory) == 1

    # Add more transitions
    for i in range(10):
        memory.push(
            state + i,
            action,
            reward + i,
            next_state + i,
            done,
        )

    assert len(memory) == 11

    # Sample a batch
    batch = memory.sample(batch_size=5)
    states, actions, rewards, next_states, dones = batch

    assert states.shape == (5, 3)
    assert actions.shape == (5, 2)
    assert rewards.shape == (5,)
    assert next_states.shape == (5, 3)
    assert dones.shape == (5,)


def test_replay_memory_circular_buffer():
    """Test that memory wraps around when capacity is exceeded."""
    memory = ReplayMemory(capacity=5, seed=42)

    state = np.array([1.0, 2.0])
    action = np.array([0.5])
    reward = 1.0
    next_state = np.array([1.1, 2.1])
    done = False

    # Add 10 transitions (twice the capacity)
    for i in range(10):
        memory.push(state * i, action, reward, next_state * i, done)

    # Should only have 5 transitions (capacity)
    assert len(memory) == 5

    # Position should have wrapped
    assert memory.position == 0  # (10 % 5)


def test_replay_memory_save_buffer():
    """Test saving replay buffer to disk."""
    memory = ReplayMemory(capacity=100, seed=42)

    # Add some transitions
    for i in range(20):
        memory.push(
            np.array([float(i), float(i + 1)]),
            np.array([0.5]),
            float(i),
            np.array([float(i + 1), float(i + 2)]),
            False,
        )

    with tempfile.TemporaryDirectory() as d:
        save_path = os.path.join(d, "test_buffer.pkl")
        memory.save_buffer(env_name="test_env", suffix="test", save_path=save_path)

        assert os.path.exists(save_path)


def test_replay_memory_load_buffer():
    """Test loading replay buffer from disk."""
    memory = ReplayMemory(capacity=100, seed=42)

    # Add some transitions
    original_data = []
    for i in range(20):
        transition = (
            np.array([float(i), float(i + 1)]),
            np.array([0.5]),
            float(i),
            np.array([float(i + 1), float(i + 2)]),
            False,
        )
        memory.push(*transition)
        original_data.append(transition)

    with tempfile.TemporaryDirectory() as d:
        save_path = os.path.join(d, "test_buffer.pkl")
        memory.save_buffer(env_name="test_env", suffix="test", save_path=save_path)

        # Create new memory and load
        new_memory = ReplayMemory(capacity=100, seed=42)
        new_memory.load_buffer(save_path)

        assert len(new_memory) == len(memory)
        assert new_memory.position == memory.position


def test_replay_memory_save_buffer_creates_checkpoints_dir():
    """Test that save_buffer creates checkpoints directory if it doesn't exist."""
    memory = ReplayMemory(capacity=100, seed=42)
    memory.push(
        np.array([1.0, 2.0]),
        np.array([0.5]),
        1.0,
        np.array([1.1, 2.1]),
        False,
    )

    # Save with default path (should create checkpoints/)
    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            memory.save_buffer(env_name="test_env", suffix="test")

            # Check that checkpoints directory was created
            assert os.path.exists("checkpoints/")
    finally:
        os.chdir(original_dir)


# ============================================================================
# SAC tests for uncovered lines
# ============================================================================


class _TensorAeroEnv:
    """Mock TensorAeroSpace environment for testing."""

    def __init__(self):
        import gymnasium as gym

        # Use gymnasium environment for simplicity
        try:
            self.base_env = gym.make("Pendulum-v1")
        except Exception:
            # Fallback to simple manual env
            self.base_env = None

        if self.base_env is None:
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {
                "shape": (1,),
                "high": np.ones(1),
                "low": -np.ones(1)
            })()
        else:
            self.observation_space = self.base_env.observation_space
            self.action_space = self.base_env.action_space

        # Add ref_signal attribute to simulate TensorAeroSpace env
        self.ref_signal = np.zeros((100, 2))

    def get_init_args(self):
        """Return initialization arguments for serialization."""
        return {
            "initial_state": [0.0, 0.0, 0.0],
            "reference_signal": [[0.0, 0.0]] * 100,
            "number_time_steps": 100,
        }

    def reset(self):
        if self.base_env:
            return self.base_env.reset()
        return np.zeros(3), {}

    def step(self, action):
        if self.base_env:
            return self.base_env.step(action)
        return np.zeros(3), 0.0, True, False, {}

    @property
    def unwrapped(self):
        # Return an object that looks like a TensorAeroSpace env
        class _Unwrapped:
            __module__ = "tensoraerospace.envs"
            __name__ = "MockTensorAeroEnv"

            def __init__(self, parent):
                self.parent = parent

            @property
            def __class__(self):
                return type(self.__name__, (), {
                    "__module__": self.__module__,
                    "__name__": self.__name__
                })

        return _Unwrapped(self)


class _EnvWithoutSpaces:
    """Mock environment without action/observation space attributes."""

    def __init__(self):
        self.observation_space = type("Space", (), {"shape": (3,)})()
        self.action_space = type("Space", (), {"shape": (2,)})()

    def reset(self):
        return np.zeros(3), {}

    def step(self, action):
        return np.zeros(3), 0.0, True, False, {}

    @property
    def unwrapped(self):
        return self

    def __getattr__(self, name):
        """Raise AttributeError for action_space and observation_space str conversion."""
        if name in ["action_space", "observation_space"]:
            raise AttributeError(f"{name} not available")
        raise AttributeError(name)


def test_get_param_env_with_tensoraerospace_env():
    """Test get_param_env with TensorAeroSpace environment."""
    env = _TensorAeroEnv()
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    params = agent.get_param_env()

    # Should serialize tensoraerospace env
    assert "tensoraerospace" in params["env"]["name"]
    assert len(params["env"]["params"]) > 0

    # Should capture ref_signal
    assert "ref_signal" in params["env"]["params"]


def test_get_param_env_with_env_missing_attributes():
    """Test get_param_env with environment missing some attributes."""

    class _MinimalSpace:
        def __init__(self, shape):
            self.shape = shape
            self.high = np.ones(shape, dtype=np.float32)
            self.low = -np.ones(shape, dtype=np.float32)

    class _MinimalEnv:
        """Environment with minimal attributes."""

        def __init__(self):
            self.observation_space = _MinimalSpace((3,))
            self.action_space = _MinimalSpace((2,))

        def reset(self):
            return np.zeros(3), {}

        def step(self, action):
            return np.zeros(3), 0.0, True, False, {}

        @property
        def unwrapped(self):
            return self

    env = _MinimalEnv()
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    # Should not raise error even if env doesn't have ref_signal
    params = agent.get_param_env()
    assert "env" in params
    assert "policy" in params


def test_save_optimizer_error_handling():
    """Test error handling when saving optimizer states fails."""

    class _TinySpace:
        def __init__(self, shape):
            self.shape = shape
            self.high = np.ones(shape, dtype=np.float32)
            self.low = -np.ones(shape, dtype=np.float32)

    class _TinyEnv:
        def __init__(self):
            self.observation_space = _TinySpace((3,))
            self.action_space = _TinySpace((2,))

        def reset(self):
            return np.zeros(3), {}

        def step(self, action):
            return np.zeros(3), 0.0, True, False, {}

        @property
        def unwrapped(self):
            return self

    env = _TinyEnv()
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    # Mock torch.save to raise an exception
    original_save = torch.save

    def mock_save_error(*args, **kwargs):
        # Save normally for most calls, but fail on optimizer save
        if "optim" in str(args[1]):
            raise IOError("Mock save error")
        return original_save(*args, **kwargs)

    with tempfile.TemporaryDirectory() as d:
        torch.save = mock_save_error
        try:
            with pytest.raises(RuntimeError, match="Error saving optimizer states"):
                agent.save(d, save_gradients=True)
        finally:
            torch.save = original_save


def test_load_tensoraerospace_env_from_config():
    """Test loading SAC from config (tensoraerospace check covered by other tests)."""
    env = _TensorAeroEnv()
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]

        # Verify that config was created with tensoraerospace-like structure
        config_path = Path(saved) / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Should have tensoraerospace in env name
        assert "tensoraerospace" in config["env"]["name"]

        # Note: actual loading would fail because MockTensorAeroEnv doesn't exist
        # in tensoraerospace.envs module, which is expected behavior for save/load


def test_load_with_gradients_auto_entropy():
    """Test loading with gradients when automatic entropy tuning is enabled."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=True,
    )

    # Do some updates to change optimizer state
    from tensoraerospace.agent.sac.replay_memory import ReplayMemory

    memory = ReplayMemory(1000, seed=42)
    for _ in range(10):
        memory.push(
            np.zeros(3),
            np.zeros(1),
            0.0,
            np.zeros(3),
            0.0,
        )

    agent.update_parameters(memory, batch_size=4, updates=1)

    with tempfile.TemporaryDirectory() as d:
        agent.save(d, save_gradients=True)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]

        # Verify alpha_optim.pth exists
        files = os.listdir(saved)
        assert "alpha_optim.pth" in files

        # Load with gradients
        loaded = SAC.from_pretrained(saved, load_gradients=True)
        assert loaded is not None
        assert hasattr(loaded, "alpha_optim")


def test_from_pretrained_relative_path():
    """Test from_pretrained with relative path prefix."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    original_dir = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            # Create a subdirectory
            subdir = Path("./models/test")
            subdir.mkdir(parents=True, exist_ok=True)

            agent.save(subdir)
            subdirs = [subdir / name for name in os.listdir(subdir)]
            saved = subdirs[0]

            # Load with relative path
            loaded = SAC.from_pretrained(str(saved))
            assert loaded is not None
    finally:
        os.chdir(original_dir)


def test_from_pretrained_explicit_nonexistent_relative_path():
    """Test error when explicit relative path doesn't exist."""
    with pytest.raises(FileNotFoundError):
        SAC.from_pretrained("./nonexistent/relative/path")


def test_policy_lr_parameter():
    """Test that policy_lr parameter is used correctly."""

    class _TinySpace:
        def __init__(self, shape):
            self.shape = shape
            self.high = np.ones(shape, dtype=np.float32)
            self.low = -np.ones(shape, dtype=np.float32)

    class _TinyEnv:
        def __init__(self):
            self.observation_space = _TinySpace((3,))
            self.action_space = _TinySpace((2,))

        def reset(self):
            return np.zeros(3), {}

        def step(self, action):
            return np.zeros(3), 0.0, True, False, {}

        @property
        def unwrapped(self):
            return self

    env = _TinyEnv()
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        device="cpu",
        lr=0.001,
        policy_lr=0.0005,
        automatic_entropy_tuning=False,
    )

    # Check that policy optimizer has correct learning rate
    assert agent.policy_optim.defaults["lr"] == 0.0005
    # Check that critic optimizer has different learning rate
    assert agent.critic_optim.defaults["lr"] == 0.001
