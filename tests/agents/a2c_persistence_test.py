"""Tests for A2C model persistence (save/load functionality).

This module tests:
- Model saving
- Model loading
- Configuration persistence
- Weight preservation
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from tensoraerospace.agent.a2c.model import A2C, Actor, Critic


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


def test_a2c_save_creates_directory():
    """Test that A2C save creates directory with all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic)

        # Save model
        save_path = agent.save(path=tmpdir)

        # Check that directory was created
        assert save_path.exists()
        assert save_path.is_dir()

        # Check that required files exist
        assert (save_path / "config.json").exists()
        assert (save_path / "actor.pth").exists()
        assert (save_path / "critic.pth").exists()


def test_a2c_save_config_content():
    """Test that saved config contains correct information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(
            env=env,
            actor=actor,
            critic=critic,
            gamma=0.95,
            entropy_beta=0.02,
            actor_lr=1e-3,
            critic_lr=2e-3,
        )

        # Save model
        save_path = agent.save(path=tmpdir)

        # Load and check config
        with open(save_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        assert "policy" in config
        assert config["policy"]["params"]["gamma"] == 0.95
        assert config["policy"]["params"]["entropy_beta"] == 0.02
        assert config["policy"]["params"]["actor_lr"] == 1e-3
        assert config["policy"]["params"]["critic_lr"] == 2e-3


def test_a2c_load_preserves_weights():
    """Test that loading preserves model weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic)

        # Train a bit to change weights
        memory = agent.run_episode(max_steps=5)
        agent.learn(memory, steps=1, discount_rewards=True)

        # Save current weights
        actor_weights_before = [p.clone() for p in agent.actor.parameters()]
        critic_weights_before = [p.clone() for p in agent.critic.parameters()]

        # Save model
        save_path = agent.save(path=tmpdir)

        # Load model
        loaded_agent = A2C.from_pretrained(str(save_path))

        # Check that weights match
        for p_before, p_after in zip(
            actor_weights_before, loaded_agent.actor.parameters()
        ):
            assert torch.allclose(p_before, p_after)

        for p_before, p_after in zip(
            critic_weights_before, loaded_agent.critic.parameters()
        ):
            assert torch.allclose(p_before, p_after)


def test_a2c_load_preserves_hyperparameters():
    """Test that loading preserves hyperparameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(
            env=env,
            actor=actor,
            critic=critic,
            gamma=0.98,
            entropy_beta=0.05,
            max_grad_norm=1.0,
        )

        # Save model
        save_path = agent.save(path=tmpdir)

        # Load model
        loaded_agent = A2C.from_pretrained(str(save_path))

        # Check hyperparameters
        assert loaded_agent.gamma == 0.98
        assert loaded_agent.entropy_beta == 0.05
        assert loaded_agent.max_grad_norm == 1.0


def test_a2c_loaded_model_can_predict():
    """Test that loaded model can make predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic)

        # Save model
        save_path = agent.save(path=tmpdir)

        # Load model
        loaded_agent = A2C.from_pretrained(str(save_path))

        # Test prediction
        state = np.random.randn(state_dim).astype(np.float32)
        action = loaded_agent.predict(state, deterministic=True)

        assert action.shape == (n_actions,)
        assert isinstance(action, np.ndarray)


def test_a2c_loaded_model_produces_same_predictions():
    """Test that loaded model produces same predictions as original."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic, seed=42)

        # Test state
        state = np.random.randn(state_dim).astype(np.float32)

        # Get prediction before saving
        action_before = agent.predict(state, deterministic=True)

        # Save model
        save_path = agent.save(path=tmpdir)

        # Load model
        loaded_agent = A2C.from_pretrained(str(save_path))

        # Get prediction after loading
        action_after = loaded_agent.predict(state, deterministic=True)

        # Predictions should be the same (deterministic mode)
        assert np.allclose(action_before, action_after)


def test_a2c_save_multiple_times():
    """Test saving the same agent multiple times creates unique directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _DummyEnv()
        state_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        actor = Actor(state_dim, n_actions)
        critic = Critic(state_dim)
        agent = A2C(env=env, actor=actor, critic=critic)

        # Save twice (with some delay to ensure different timestamps)
        import time

        save_path_1 = agent.save(path=tmpdir)
        time.sleep(1.1)  # Ensure different second
        save_path_2 = agent.save(path=tmpdir)

        # Paths should be different
        assert save_path_1 != save_path_2
        assert save_path_1.exists()
        assert save_path_2.exists()


def test_a2c_load_nonexistent_path():
    """Test that loading from nonexistent path raises appropriate error."""
    try:
        A2C.from_pretrained("/nonexistent/path/to/model")
        assert False, "Should have raised an error"
    except (FileNotFoundError, OSError, ValueError):
        pass  # Expected


def test_a2c_get_param_env():
    """Test get_param_env method returns correct structure."""
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(
        env=env,
        actor=actor,
        critic=critic,
        gamma=0.99,
        entropy_beta=0.01,
    )

    params = agent.get_param_env()

    assert "env" in params
    assert "policy" in params
    assert "name" in params["env"]
    assert "params" in params["policy"]
    assert params["policy"]["params"]["gamma"] == 0.99
    assert params["policy"]["params"]["entropy_beta"] == 0.01


def test_a2c_save_uses_default_path():
    """Test that save uses default path when none provided."""
    # Note: This test creates files in the current directory
    env = _DummyEnv()
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    agent = A2C(env=env, actor=actor, critic=critic)

    # Save without specifying path
    save_path = agent.save()

    try:
        # Should create in checkpoints directory
        assert "checkpoints" in str(save_path)
        assert save_path.exists()
    finally:
        # Clean up
        import shutil

        if save_path.exists():
            shutil.rmtree(save_path)


if __name__ == "__main__":
    # Run all tests
    test_a2c_save_creates_directory()
    test_a2c_save_config_content()
    test_a2c_load_preserves_weights()
    test_a2c_load_preserves_hyperparameters()
    test_a2c_loaded_model_can_predict()
    test_a2c_loaded_model_produces_same_predictions()
    test_a2c_save_multiple_times()
    test_a2c_load_nonexistent_path()
    test_a2c_get_param_env()
    test_a2c_save_uses_default_path()
    print("All persistence tests passed!")
