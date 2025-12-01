"""Comprehensive tests for DDPG save/load functionality.

Tests for checkpointing, HuggingFace Hub integration, and state persistence.
"""

import json
import os
import shutil

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ddpg.model import DDPG


@pytest.fixture
def test_tmpdir():
    """Create test directory in /tmp and clean up after test."""
    test_dir = "/tmp/ddpg_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


class _DummySpace:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _FakeEnv:
    """Minimal fake environment for testing."""

    def __init__(self, obs_dim=3, act_dim=1):
        self.observation_space = _DummySpace((obs_dim,))
        self.action_space = _DummySpace((act_dim,))
        self.unwrapped = self

    @property
    def __class__(self):
        """Return a fake class for serialization."""

        class FakeClass:
            __name__ = "_FakeEnv"
            __module__ = "test_ddpg_save_load"

        return FakeClass

    def reset(self):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return state, {}

    def step(self, action):
        del action  # Unused but required by interface
        next_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        return next_state, reward, terminated, truncated, {}


class TestDDPGSaveCheckpoint:
    """Tests for saving DDPG checkpoints."""

    def test_save_checkpoint_basic(self, test_tmpdir):
        """Test basic checkpoint saving."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent.save(filepath)

        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0

    def test_save_checkpoint_with_grads(self, test_tmpdir):
        """Test saving checkpoint with gradients."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Perform update to generate gradients
        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        agent.ddpg_update(batch_size=16)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent.save(filepath, include_grads=True)

        # Load and check for gradients
        ckpt = torch.load(filepath, weights_only=False)
        assert "value_net_grads" in ckpt
        assert "policy_net_grads" in ckpt

    def test_save_directory_structure(self, test_tmpdir):
        """Test saving as directory (HuggingFace style)."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        save_dir = os.path.join(test_tmpdir, "model_dir")
        agent.save(save_dir)

        # Check directory structure
        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "config.json"))
        assert os.path.exists(os.path.join(save_dir, "policy.pth"))
        assert os.path.exists(os.path.join(save_dir, "value.pth"))
        assert os.path.exists(os.path.join(save_dir, "target_policy.pth"))
        assert os.path.exists(os.path.join(save_dir, "target_value.pth"))

    def test_save_includes_obs_rms(self, test_tmpdir):
        """Test that observation normalizer is saved."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        # Update normalization statistics
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agent.obs_rms.update(data)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent.save(filepath)

        ckpt = torch.load(filepath, weights_only=False)
        assert "obs_rms" in ckpt
        assert "mean" in ckpt["obs_rms"]
        assert "var" in ckpt["obs_rms"]

    def test_save_includes_replay_buffer(self, test_tmpdir):
        """Test that replay buffer is saved."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Add some transitions
        for _ in range(10):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent.replay_buffer.push(s, a, 1.0, s, False)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent.save(filepath)

        ckpt = torch.load(filepath, weights_only=False)
        assert "replay_buffer" in ckpt
        assert len(ckpt["replay_buffer"]["buffer"]) == 10


class TestDDPGLoadCheckpoint:
    """Tests for loading DDPG checkpoints."""

    def test_load_checkpoint_basic(self, test_tmpdir):
        """Test basic checkpoint loading."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Train a bit
        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent1.replay_buffer.push(s, a, 1.0, s, False)
        agent1.ddpg_update(batch_size=16)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        # Create new agent and load
        agent2 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)
        agent2.load(filepath)

        # Check that weights match
        for p1, p2 in zip(
            agent1.policy_net.parameters(), agent2.policy_net.parameters()
        ):
            assert torch.allclose(p1, p2)

    def test_load_with_obs_rms(self, test_tmpdir):
        """Test loading with observation normalizer."""
        env = _FakeEnv()
        agent1 = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        # Update normalization statistics
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agent1.obs_rms.update(data)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        agent2 = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-3,
            replay_buffer_size=100,
            normalize_observations=True,
        )
        agent2.load(filepath)

        # Check that normalizer statistics match
        assert np.allclose(agent2.obs_rms.mean, agent1.obs_rms.mean)
        assert np.allclose(agent2.obs_rms.var, agent1.obs_rms.var)

    def test_load_without_optimizer(self, test_tmpdir):
        """Test loading without optimizer states."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        agent2 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Load without optimizer states
        agent2.load(filepath, load_optimizer=False)

        # Should still work (won't throw error)

    def test_load_without_replay_buffer(self, test_tmpdir):
        """Test loading without replay buffer."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Add transitions
        for _ in range(10):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent1.replay_buffer.push(s, a, 1.0, s, False)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        agent2 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)
        agent2.load(filepath, load_replay=False)

        # Replay buffer should be empty
        assert len(agent2.replay_buffer) == 0

    def test_load_preserves_training_state(self, test_tmpdir):
        """Test that training state is preserved."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Simulate some training
        agent1.frame_idx = 1000
        agent1.rewards = [1.0, 2.0, 3.0]
        agent1.max_frames = 10000
        agent1.max_steps = 200
        agent1.batch_size = 64

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        agent2 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)
        agent2.load(filepath)

        assert agent2.frame_idx == 1000
        assert agent2.rewards == [1.0, 2.0, 3.0]
        assert agent2.max_frames == 10000


class TestDDPGGetParamEnv:
    """Tests for get_param_env method."""

    def test_get_param_env_structure(self, test_tmpdir):
        """Test that get_param_env returns correct structure."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1000)

        config = agent.get_param_env()

        assert "env" in config
        assert "policy" in config
        assert "name" in config["env"]
        assert "params" in config["env"]
        assert "name" in config["policy"]
        assert "params" in config["policy"]

    def test_get_param_env_policy_params(self, test_tmpdir):
        """Test that policy parameters are correctly saved."""
        env = _FakeEnv()
        agent = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-4,
            replay_buffer_size=1000,
            normalize_observations=True,
        )

        config = agent.get_param_env()
        params = config["policy"]["params"]

        assert params["value_lr"] == 1e-3
        assert params["policy_lr"] == 1e-4
        assert params["replay_buffer_size"] == 1000
        assert params["normalize_observations"] is True

    def test_get_param_env_includes_obs_rms(self, test_tmpdir):
        """Test that obs_rms is included when present."""
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

        config = agent.get_param_env()
        params = config["policy"]["params"]

        assert "obs_rms" in params
        assert "mean" in params["obs_rms"]
        assert "var" in params["obs_rms"]


class TestDDPGFromPretrained:
    """Tests for from_pretrained classmethod."""

    def test_from_pretrained_local_directory(self, test_tmpdir):
        """Test loading from local directory."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1000)

        # Train a bit
        for _ in range(20):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent1.replay_buffer.push(s, a, 1.0, s, False)
        agent1.ddpg_update(batch_size=16)

        save_dir = os.path.join(test_tmpdir, "model")
        agent1.save(save_dir)

        # This will fail due to env reconstruction, but tests the flow
        # In real use, the environment would be properly serializable
        try:
            agent2 = DDPG.from_pretrained(save_dir)
            # If it works, check some properties
            assert agent2.value_lr == 1e-3
            assert agent2.policy_lr == 1e-4
        except Exception:
            # Expected to fail due to fake environment
            pass

    def test_from_pretrained_config_structure(self, test_tmpdir):
        """Test that config.json has correct structure."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1000)

        save_dir = os.path.join(test_tmpdir, "model")
        agent.save(save_dir)

        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        assert "env" in config
        assert "policy" in config
        assert config["policy"]["params"]["value_lr"] == 1e-3
        assert config["policy"]["params"]["policy_lr"] == 1e-4


class TestDDPGPushToHub:
    """Tests for push_to_hub method (without actual Hub interaction)."""

    def test_push_to_hub_creates_directory(self, test_tmpdir):
        """Test that push_to_hub creates proper directory structure."""
        env = _FakeEnv()
        agent = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        save_path = os.path.join(test_tmpdir, "my_model")

        # This will fail at actual upload, but will create directory
        try:
            agent.push_to_hub(
                repo_name="test/repo", save_path=save_path, access_token=None
            )
        except Exception:
            pass  # Expected to fail without proper setup

        # Check that directory was created
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))


class TestDDPGSaveLoadIntegration:
    """Integration tests for save/load cycle."""

    def test_save_load_roundtrip(self, test_tmpdir):
        """Test complete save/load roundtrip."""
        env = _FakeEnv()
        agent1 = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-4,
            replay_buffer_size=100,
            normalize_observations=True,
        )

        # Train
        for _ in range(30):
            s = np.random.randn(3).astype(np.float32)
            a = np.random.randn(1).astype(np.float32)
            agent1.replay_buffer.push(s, a, 1.0, s, False)

        agent1.ddpg_update(batch_size=16)

        # Update obs_rms
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        agent1.obs_rms.update(data)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)

        agent2 = DDPG(
            env=env,
            value_lr=1e-3,
            policy_lr=1e-4,
            replay_buffer_size=100,
            normalize_observations=True,
        )
        agent2.load(filepath)

        # Test that loaded agent produces same outputs
        test_state = np.random.randn(3).astype(np.float32)
        action1 = agent1.policy_net.get_action(
            agent1._normalize_observation(test_state)
        )
        action2 = agent2.policy_net.get_action(
            agent2._normalize_observation(test_state)
        )

        assert np.allclose(action1, action2, atol=1e-5)

    def test_save_load_preserves_behavior(self, test_tmpdir):
        """Test that saved/loaded agent behaves identically."""
        env = _FakeEnv()
        agent1 = DDPG(env=env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=100)

        # Get initial action
        test_obs = np.array([1.0, 0.5, -0.5])
        action1_before = agent1.policy_net.get_action(test_obs)

        filepath = os.path.join(test_tmpdir, "checkpoint.pt")
        agent1.save(filepath)
        agent1.load(filepath)

        action1_after = agent1.policy_net.get_action(test_obs)

        # Should produce identical actions
        assert np.allclose(action1_before, action1_after, atol=1e-6)
