"""Shared fixtures for DQN tests."""

import numpy as np
import pytest
import torch


class DummyActionSpace:
    """Dummy action space that mimics Discrete."""

    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        """Return random action."""
        return np.random.randint(0, self.n)


class DummyObservationSpace:
    """Dummy observation space that mimics Box."""

    def __init__(self, shape: tuple):
        self.shape = shape


class DummyEnv:
    """Minimal Gymnasium-compatible environment for testing."""

    def __init__(self, obs_dim: int = 4, num_actions: int = 2, max_steps: int = 10):
        self.observation_space = DummyObservationSpace(shape=(obs_dim,))
        self.action_space = DummyActionSpace(n=num_actions)
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, seed=None, options=None):  # noqa: ARG002
        """Reset environment and return (obs, info)."""
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        info: dict = {}
        return obs, info

    def step(self, action: int):  # noqa: ARG002
        """Step environment and return (obs, reward, terminated, truncated)."""
        self._step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = float(np.random.rand())
        terminated = self._step_count >= self.max_steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        """Dummy render."""

    def close(self):
        """Dummy close."""


class WrappedEnv:
    """Wrapped environment that exposes .env attribute."""

    def __init__(self, obs_dim: int = 4, num_actions: int = 2, max_steps: int = 5):
        self.env = DummyEnv(obs_dim, num_actions, max_steps)

    def capture_frame(self):
        """Dummy frame capture for evaluation with render=True."""

    def close(self):
        """Close wrapped environment."""
        self.env.close()


@pytest.fixture
def seed_all():
    """Seed numpy, torch, and python random for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


@pytest.fixture
def dummy_env():
    """Provide a basic dummy environment."""
    return DummyEnv(obs_dim=4, num_actions=2, max_steps=10)


@pytest.fixture
def wrapped_env():
    """Provide a wrapped environment for evaluation tests."""
    return WrappedEnv(obs_dim=4, num_actions=2, max_steps=5)
