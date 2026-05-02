"""Tests for DQN Agent save/load functionality."""

import json

import numpy as np
import pytest
import torch

from tensoraerospace.agent.dqn import DQNAgent, Model


@pytest.fixture
def dqn_agent_for_save(dummy_env, seed_all):
    """Create a DQN agent for save/load tests."""
    model = Model(num_actions=dummy_env.action_space.n)
    target_model = Model(num_actions=dummy_env.action_space.n)

    # Initialize models with forward pass
    obs = np.zeros((1, 4), dtype=np.float32)
    _ = model.predict(obs)
    _ = target_model.predict(obs)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=dummy_env,
        epsilon=0.15,
        epsilon_dacay=0.95,
        gamma=0.99,
        buffer_size=16,
        batch_size=4,
        train_nums=100,
        alpha=0.6,
        beta=0.4,
    )
    return agent


def test_save_creates_files(dqn_agent_for_save, tmp_path):
    """Test that save creates all expected files."""
    save_path = tmp_path / "test_agent"

    # Save without gradients
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Check that required files exist
    assert (save_path / "model.pth").exists()
    assert (save_path / "target_model.pth").exists()
    assert (save_path / "config.json").exists()

    # Optimizer should NOT be saved when save_gradients=False
    assert not (save_path / "optimizer.pth").exists()


def test_save_writes_safe_state_dict(dqn_agent_for_save, tmp_path):
    """Saved model files should be loadable with weights_only=True."""
    save_path = tmp_path / "test_agent_state_dict"

    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    state = torch.load(save_path / "model.pth", weights_only=True)
    assert isinstance(state, dict)
    assert "fc1.weight" in state


def test_save_creates_optimizer_file(dqn_agent_for_save, tmp_path):
    """Test that save creates optimizer file when save_gradients=True."""
    save_path = tmp_path / "test_agent_with_grad"

    # Save with gradients
    dqn_agent_for_save.save(path=save_path, save_gradients=True)

    # Check that optimizer file exists
    assert (save_path / "optimizer.pth").exists()


def test_save_config_contains_correct_values(dqn_agent_for_save, tmp_path):
    """Test that config.json contains correct hyperparameters."""
    save_path = tmp_path / "test_agent_config"

    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Read config
    with open(save_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Verify config values
    assert config["epsilon"] == dqn_agent_for_save.epsilon
    assert config["epsilon_decay"] == dqn_agent_for_save.epsilon_decay
    assert config["gamma"] == dqn_agent_for_save.gamma
    assert config["batch_size"] == dqn_agent_for_save.batch_size
    assert config["buffer_size"] == dqn_agent_for_save.buffer_size
    assert config["alpha"] == dqn_agent_for_save.alpha
    assert config["beta"] == dqn_agent_for_save.beta


def test_save_default_path(dqn_agent_for_save, tmp_path, monkeypatch):
    """Test that save uses default path when path=None."""
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    # Save without specifying path
    dqn_agent_for_save.save(path=None, save_gradients=False)

    # Should create "dqn_agent" directory in cwd
    default_path = tmp_path / "dqn_agent"
    assert default_path.exists()
    assert (default_path / "model.pth").exists()
    assert (default_path / "config.json").exists()


def test_load_model_is_usable(dqn_agent_for_save, tmp_path, seed_all):
    """Test that loaded model can be used for forward pass."""
    save_path = tmp_path / "test_load"

    # Save agent
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Load model through the safe state_dict loader
    loaded_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)
    loaded_model = loaded_agent.model

    # Check it's a nn.Module
    assert isinstance(loaded_model, torch.nn.Module)

    # Use for prediction
    obs = np.random.randn(1, 4).astype(np.float32)
    q_values = loaded_model.predict(obs)

    assert q_values.shape == (1, 2)
    assert isinstance(q_values, np.ndarray)


def test_load_model_matches_original(dqn_agent_for_save, tmp_path, seed_all):
    """Test that loaded model produces same output as original."""
    save_path = tmp_path / "test_match"

    # Save agent
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Get prediction from original model
    obs = np.random.randn(2, 4).astype(np.float32)
    original_q = dqn_agent_for_save.model.predict(obs)

    # Load model and get prediction
    loaded_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)
    loaded_q = loaded_agent.model.predict(obs)

    # Should be identical (same parameters)
    assert np.allclose(original_q, loaded_q, rtol=1e-5)


def test_load_target_model(dqn_agent_for_save, tmp_path, seed_all):
    """Test that target model can be loaded."""
    save_path = tmp_path / "test_target"

    # Save agent
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Load target model through the safe state_dict loader
    loaded_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)
    loaded_target = loaded_agent.target_model

    assert isinstance(loaded_target, torch.nn.Module)

    # Use for prediction
    obs = np.random.randn(1, 4).astype(np.float32)
    q_values = loaded_target.predict(obs)

    assert q_values.shape == (1, 2)


def test_load_optimizer_state(dqn_agent_for_save, tmp_path):
    """Test that optimizer state can be loaded when saved."""
    save_path = tmp_path / "test_optimizer"

    # Save with gradients
    dqn_agent_for_save.save(path=save_path, save_gradients=True)

    # Load optimizer state
    optimizer_state = torch.load(save_path / "optimizer.pth", weights_only=True)

    # Should be a dictionary
    assert isinstance(optimizer_state, dict)
    assert "state" in optimizer_state
    assert "param_groups" in optimizer_state


def test_save_creates_directory(dqn_agent_for_save, tmp_path):
    """Test that save creates directory if it doesn't exist."""
    save_path = tmp_path / "nested" / "dir" / "test_agent"

    # Directory doesn't exist yet
    assert not save_path.exists()

    # Save should create it
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    assert save_path.exists()
    assert (save_path / "model.pth").exists()


def test_save_overwrites_existing(dqn_agent_for_save, tmp_path, seed_all):
    """Test that save overwrites existing files."""
    save_path = tmp_path / "test_overwrite"

    # First save
    dqn_agent_for_save.save(path=save_path, save_gradients=False)
    first_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)

    # Modify agent's model
    for param in dqn_agent_for_save.model.parameters():
        param.data += 0.5

    # Second save (should overwrite)
    dqn_agent_for_save.save(path=save_path, save_gradients=False)
    second_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)

    # Get predictions
    obs = np.random.randn(1, 4).astype(np.float32)
    first_q = first_agent.model.predict(obs)
    second_q = second_agent.model.predict(obs)

    # Should be different (second save should have updated weights)
    assert not np.allclose(first_q, second_q, rtol=1e-5)


def test_save_preserves_lazy_layers(dqn_agent_for_save, tmp_path, seed_all):
    """Test that lazy layers are properly saved and loaded."""
    save_path = tmp_path / "test_lazy"

    # Save agent
    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Load model through the safe state_dict loader
    loaded_agent = DQNAgent.load(save_path, env=dqn_agent_for_save.env)
    loaded_model = loaded_agent.model

    # Lazy layer should be initialized after save
    assert hasattr(loaded_model.fc1, "in_features")
    assert loaded_model.fc1.in_features == 4

    # Should work with same input size
    obs = np.random.randn(3, 4).astype(np.float32)
    q_values = loaded_model.predict(obs)
    assert q_values.shape == (3, 2)


def test_config_json_is_valid(dqn_agent_for_save, tmp_path):
    """Test that config.json is valid JSON."""
    save_path = tmp_path / "test_json"

    dqn_agent_for_save.save(path=save_path, save_gradients=False)

    # Should be able to load as JSON
    with open(save_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    # Should have expected keys
    expected_keys = [
        "gamma",
        "epsilon",
        "epsilon_decay",
        "min_epsilon",
        "batch_size",
        "target_update_iter",
        "train_nums",
        "buffer_size",
        "alpha",
        "beta",
    ]
    for key in expected_keys:
        assert key in config
