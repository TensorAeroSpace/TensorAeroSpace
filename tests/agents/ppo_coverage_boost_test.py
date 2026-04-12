"""Tests targeting uncovered code paths in tensoraerospace/agent/ppo/model.py.

Covers: _to_cpu_detached (dict/list/tuple), _optimizer_state_to_device,
_AsyncBestCheckpointSaver, _act_tensor_batch, _is_vector_env, _to_tensor,
_train_vector, _save_best_checkpoint, eval(), close(), test_reward edge cases,
save/load roundtrip with optimizer + train_state, and from_pretrained local path.
"""

import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tensoraerospace.agent.ppo.model import (
    PPO,
    RunningMeanStd,
    _AsyncBestCheckpointSaver,
    _atomic_np_savez,
    _atomic_torch_save,
    _atomic_write_json,
    _optimizer_state_dict_cpu,
    _optimizer_state_to_device,
    _state_dict_cpu,
    _to_cpu_detached,
)

# ---------------------------------------------------------------------------
# Lightweight dummy environments
# ---------------------------------------------------------------------------


class _DummyEnv:
    """Minimal single-step env for fast unit tests."""

    def __init__(self):
        self.observation_space = type("S", (), {"shape": (4,)})
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )
        self.unwrapped = self
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        self._step_count += 1
        done = self._step_count >= 5
        return (
            np.zeros(4, dtype=np.float32),
            1.0,
            done,
            False,
            {},
        )


class _DummyVectorEnv:
    """Fake vector env that returns batched observations (N, obs_dim)."""

    def __init__(self, n_envs=2):
        self.num_envs = n_envs
        self.observation_space = type("S", (), {"shape": (4,)})
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )
        self.unwrapped = self
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.zeros((self.num_envs, 4), dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        n = self.num_envs
        obs = np.zeros((n, 4), dtype=np.float32)
        reward = np.ones(n, dtype=np.float32)
        terminated = np.array([self._step_count >= 4] * n)
        truncated = np.zeros(n, dtype=bool)
        info = {}
        return obs, reward, terminated, truncated, info


class _DummyVectorEnvOldAPI:
    """Vector env that uses old 4-return step API."""

    def __init__(self, n_envs=2):
        self.num_envs = n_envs
        self.observation_space = type("S", (), {"shape": (4,)})
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0]),
                "high": np.array([1.0, 1.0]),
            },
        )
        self.unwrapped = self
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        return np.zeros((self.num_envs, 4), dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        n = self.num_envs
        obs = np.zeros((n, 4), dtype=np.float32)
        reward = np.ones(n, dtype=np.float32)
        done = np.array([self._step_count >= 4] * n)
        info = {}
        return obs, reward, done, info


class _OldStepEnv(_DummyEnv):
    """Env whose step() returns 4-tuple (no truncated)."""

    def step(self, _action):
        self._step_count += 1
        done = self._step_count >= 5
        return np.zeros(4, dtype=np.float32), 1.0, done, {}


class _OldResetEnv(_DummyEnv):
    """Env whose reset() returns just an obs (no info dict)."""

    def reset(self):
        self._step_count = 0
        return np.zeros(4, dtype=np.float32)


# ===========================================================================
# Unit tests for helper functions
# ===========================================================================


class TestToCpuDetached:
    """Cover _to_cpu_detached for dict, list, tuple, and plain object."""

    def test_dict(self):
        t = torch.tensor([1.0, 2.0], device="cpu")
        result = _to_cpu_detached({"a": t, "b": 42})
        assert isinstance(result, dict)
        assert torch.is_tensor(result["a"])
        assert result["b"] == 42

    def test_list(self):
        t = torch.tensor([3.0])
        result = _to_cpu_detached([t, 7])
        assert isinstance(result, list)
        assert torch.is_tensor(result[0])
        assert result[1] == 7

    def test_tuple(self):
        t = torch.tensor([4.0])
        result = _to_cpu_detached((t, "hello"))
        assert isinstance(result, tuple)
        assert torch.is_tensor(result[0])
        assert result[1] == "hello"

    def test_plain(self):
        assert _to_cpu_detached(99) == 99
        assert _to_cpu_detached("str") == "str"

    def test_nested(self):
        t = torch.tensor([1.0])
        result = _to_cpu_detached({"a": [t, (t,)]})
        assert torch.is_tensor(result["a"][0])
        assert torch.is_tensor(result["a"][1][0])


class TestOptimizerStateToDevice:
    """Cover _optimizer_state_to_device."""

    def test_moves_tensors(self):
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        # Perform one step so that optimizer state is populated
        loss = model(torch.randn(1, 3)).sum()
        loss.backward()
        opt.step()
        # All state tensors should be on cpu
        _optimizer_state_to_device(opt, torch.device("cpu"))
        for state in opt.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    assert v.device == torch.device("cpu")


# ===========================================================================
# _AsyncBestCheckpointSaver
# ===========================================================================


class TestAsyncBestCheckpointSaver:
    """Cover init, submit, flush, close, and worker of the background saver."""

    def test_basic_save(self, tmp_path):
        saver = _AsyncBestCheckpointSaver()
        model_dir = tmp_path / "best"
        # Minimal job
        job = {
            "model_dir": str(model_dir),
            "config": {"policy": {"name": "test"}},
            "actor_state": {"w": torch.tensor([1.0])},
            "critic_state": {"w": torch.tensor([2.0])},
            "actor_opt_state": None,
            "critic_opt_state": None,
            "train_state": None,
            "obs_rms": None,
            "ret_rms": None,
            "meta": None,
        }
        saver.submit(job)
        saver.flush(timeout=10.0)
        assert (model_dir / "config.json").exists()
        assert (model_dir / "actor.pth").exists()
        assert (model_dir / "critic.pth").exists()
        saver.close(timeout=5.0)

    def test_submit_replaces_old(self, tmp_path):
        """Submit twice rapidly; the saver should not block."""
        saver = _AsyncBestCheckpointSaver()
        for i in range(3):
            model_dir = tmp_path / f"best_{i}"
            job = {
                "model_dir": str(model_dir),
                "config": {"v": i},
                "actor_state": {"w": torch.tensor([float(i)])},
                "critic_state": {"w": torch.tensor([float(i)])},
            }
            saver.submit(job)
        saver.flush(timeout=10.0)
        saver.close(timeout=5.0)

    def test_full_job_with_all_fields(self, tmp_path):
        """Submit a job that has optimizer states, normalization, meta, etc."""
        saver = _AsyncBestCheckpointSaver()
        model_dir = tmp_path / "full"
        job = {
            "model_dir": str(model_dir),
            "config": {"policy": {"name": "test"}},
            "actor_state": {"w": torch.tensor([1.0])},
            "critic_state": {"w": torch.tensor([2.0])},
            "actor_opt_state": {"state": {}},
            "critic_opt_state": {"state": {}},
            "train_state": {"best_reward": 42.0},
            "obs_rms": {
                "mean": np.zeros(3),
                "var": np.ones(3),
                "count": 100.0,
            },
            "ret_rms": {
                "mean": np.array(0.0),
                "var": np.array(1.0),
                "count": 50.0,
            },
            "meta": {"eval_reward": 42.0, "episode": 10},
        }
        saver.submit(job)
        saver.flush(timeout=10.0)
        assert (model_dir / "actor_opt.pth").exists()
        assert (model_dir / "critic_opt.pth").exists()
        assert (model_dir / "train_state.json").exists()
        assert (model_dir / "obs_rms.npz").exists()
        assert (model_dir / "ret_rms.npz").exists()
        assert (model_dir / "best_meta.json").exists()
        saver.close(timeout=5.0)

    def test_flush_no_timeout(self, tmp_path):
        saver = _AsyncBestCheckpointSaver()
        # Nothing queued; flush(None) should return instantly
        saver.flush(timeout=None)
        saver.close(timeout=5.0)

    def test_flush_with_short_timeout(self, tmp_path):
        """Flush with very short timeout should not hang."""
        saver = _AsyncBestCheckpointSaver()
        saver.flush(timeout=0.01)
        saver.close(timeout=5.0)

    def test_close_on_empty_queue(self):
        saver = _AsyncBestCheckpointSaver()
        saver.close(timeout=5.0)


# ===========================================================================
# PPO: _is_vector_env, _to_tensor
# ===========================================================================


class TestIsVectorEnv:
    def test_with_num_envs_attribute(self):
        env = _DummyVectorEnv(n_envs=3)
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = np.zeros((3, 4))
        assert agent._is_vector_env(obs) is True

    def test_single_env(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = np.zeros(4)
        assert agent._is_vector_env(obs) is False

    def test_tensor_batched(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = torch.zeros(2, 4)
        assert agent._is_vector_env(obs) is True

    def test_tensor_single(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = torch.zeros(4)
        assert agent._is_vector_env(obs) is False


class TestToTensor:
    def test_numpy(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        result = agent._to_tensor(np.array([1.0, 2.0]))
        assert torch.is_tensor(result)

    def test_torch(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        result = agent._to_tensor(torch.tensor([1.0, 2.0]))
        assert torch.is_tensor(result)


# ===========================================================================
# PPO: act (batched path) and _act_tensor_batch
# ===========================================================================


class TestActBatched:
    def test_batched_numpy(self):
        env = _DummyVectorEnv(n_envs=3)
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = np.zeros((3, 4), dtype=np.float32)
        action, mean, log_prob = agent.act(obs)
        assert action.shape == (3, 2)
        assert mean.shape == (3, 2)
        assert log_prob.shape == (3, 1)

    def test_batched_torch(self):
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = torch.zeros(2, 4, dtype=torch.float32)
        action, mean, log_prob = agent.act(obs)
        assert action.shape == (2, 2)

    def test_batched_deterministic(self):
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        obs = np.zeros((2, 4), dtype=np.float32)
        a1, m1, _ = agent.act(obs, deterministic=True)
        a2, m2, _ = agent.act(obs, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_batched_with_obs_normalization(self):
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(
            env=env,
            normalize_obs=True,
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        # Feed some data into obs_rms
        agent.obs_rms.update(np.random.randn(10, 4))
        obs = np.zeros((2, 4), dtype=np.float32)
        action, mean, log_prob = agent.act(obs)
        assert action.shape == (2, 2)


# ===========================================================================
# PPO: eval() and close()
# ===========================================================================


class TestEvalAndClose:
    def test_eval_returns_self(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        result = agent.eval()
        assert result is agent
        # Networks should be in eval mode
        assert not agent.actor.training
        assert not agent.critic.training

    def test_close_without_saver(self):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        # Should not raise
        agent.close()

    def test_close_with_saver(self, tmp_path):
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        # Manually create a saver
        agent._best_saver = _AsyncBestCheckpointSaver()
        agent.close()
        assert agent._best_saver is None


# ===========================================================================
# PPO: test_reward edge cases
# ===========================================================================


class TestTestReward:
    def test_old_reset_api(self):
        """Env whose reset() returns just obs (no info tuple)."""
        env = _OldResetEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        reward = agent.test_reward()
        assert isinstance(reward, (int, float, np.number))

    def test_old_step_api(self):
        """Env whose step() returns 4-tuple."""
        env = _OldStepEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        reward = agent.test_reward()
        assert reward == 5.0  # 5 steps * 1.0 reward


# ===========================================================================
# PPO: _save_best_checkpoint (sync and async paths)
# ===========================================================================


class TestSaveBestCheckpoint:
    def test_sync_checkpoint(self, tmp_path):
        env = _DummyEnv()
        agent = PPO(
            env=env,
            save_best_model=True,
            save_best_async=False,
            best_model_dir=str(tmp_path / "best"),
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        agent._save_best_checkpoint(eval_reward=10.0, episode=1)
        assert (tmp_path / "best" / "config.json").exists()
        assert (tmp_path / "best" / "actor.pth").exists()
        assert (tmp_path / "best" / "critic.pth").exists()
        assert (tmp_path / "best" / "best_meta.json").exists()

    def test_sync_checkpoint_with_normalization(self, tmp_path):
        env = _DummyEnv()
        agent = PPO(
            env=env,
            save_best_model=True,
            save_best_async=False,
            normalize_obs=True,
            normalize_reward=True,
            best_model_dir=str(tmp_path / "best"),
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        agent._save_best_checkpoint(eval_reward=10.0, episode=1)
        assert (tmp_path / "best" / "obs_rms.npz").exists()
        assert (tmp_path / "best" / "ret_rms.npz").exists()

    def test_async_checkpoint(self, tmp_path):
        env = _DummyEnv()
        agent = PPO(
            env=env,
            save_best_model=True,
            save_best_async=True,
            best_model_dir=str(tmp_path / "best"),
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        agent._save_best_checkpoint(eval_reward=10.0, episode=1)
        # Saver should have been lazy-created
        assert agent._best_saver is not None
        agent._best_saver.flush(timeout=10.0)
        assert (tmp_path / "best" / "config.json").exists()
        agent.close()

    def test_disabled_save(self, tmp_path):
        env = _DummyEnv()
        agent = PPO(
            env=env,
            save_best_model=False,
            best_model_dir=str(tmp_path / "best"),
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        agent._save_best_checkpoint(eval_reward=10.0, episode=1)
        assert not (tmp_path / "best").exists()


# ===========================================================================
# PPO: _train_vector
# ===========================================================================


class TestTrainVector:
    def test_vector_env_5step_training(self, tmp_path):
        """Run _train_vector for 1 episode with a tiny vector env."""
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=8,
            num_epochs=1,
            batch_size=4,
            eval_freq=1,
            save_best_model=True,
            save_best_async=False,
            best_model_dir=str(tmp_path / "best"),
        )
        # _train_vector should be called automatically because env is vector
        agent.train()
        # After training, a best checkpoint should exist (eval_freq=1 triggers eval)
        assert (tmp_path / "best" / "config.json").exists()

    def test_vector_env_old_step_api(self, tmp_path):
        """Vector env with old 4-return step API."""
        env = _DummyVectorEnvOldAPI(n_envs=2)
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=6,
            num_epochs=1,
            batch_size=4,
            eval_freq=100,  # skip eval
            save_best_model=False,
        )
        agent.train()

    def test_vector_env_with_reward_normalization(self, tmp_path):
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=6,
            num_epochs=1,
            batch_size=4,
            normalize_reward=True,
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()

    def test_vector_env_with_target_kl(self, tmp_path):
        env = _DummyVectorEnv(n_envs=2)
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=6,
            num_epochs=3,
            batch_size=4,
            target_kl=0.001,  # Very small -> will trigger early stopping
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()

    def test_vector_env_no_scores_path(self, tmp_path):
        """When no episode finishes during rollout, scores is empty.

        Use very short rollout so no env finishes.
        """
        env = _DummyVectorEnv(n_envs=2)
        # Make env never terminate during short rollout
        env._step_count = -1000
        original_step = env.step

        def never_done_step(action):
            n = env.num_envs
            obs = np.zeros((n, 4), dtype=np.float32)
            reward = np.ones(n, dtype=np.float32)
            terminated = np.zeros(n, dtype=bool)
            truncated = np.zeros(n, dtype=bool)
            return obs, reward, terminated, truncated, {}

        env.step = never_done_step
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=4,
            eval_freq=1,
            save_best_model=False,
        )
        agent.train()


# ===========================================================================
# PPO: train() with single env edge cases
# ===========================================================================


class TestTrainSingleEnv:
    def test_train_old_step_api(self):
        """Single env with old 4-return step()."""
        env = _OldStepEnv()
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=8,
            num_epochs=1,
            batch_size=4,
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()

    def test_train_old_reset_api(self):
        """Single env with old reset() returning only obs."""
        env = _OldResetEnv()
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=8,
            num_epochs=1,
            batch_size=4,
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()

    def test_train_triggers_eval(self, tmp_path):
        """Training triggers evaluation with eval_freq=1."""
        env = _DummyEnv()
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=8,
            num_epochs=1,
            batch_size=4,
            eval_freq=1,
            save_best_model=True,
            save_best_async=False,
            best_model_dir=str(tmp_path / "best"),
        )
        agent.train()
        # Best checkpoint should have been written via eval
        assert (tmp_path / "best" / "config.json").exists()

    def test_train_with_done_mid_rollout(self):
        """Env terminates mid-rollout and resets."""
        env = _DummyEnv()
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=20,  # Longer than episode length (5 steps)
            num_epochs=1,
            batch_size=4,
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()


# ===========================================================================
# PPO: save / load roundtrip with optimizer + train_state + normalization
# ===========================================================================


class TestSaveLoadRoundtrip:
    def test_full_roundtrip(self, tmp_path):
        """Save then load with optimizer states, train_state, and normalization."""
        env = _DummyEnv()
        agent = PPO(
            env=env,
            normalize_obs=True,
            normalize_reward=True,
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=2,
        )
        # Do a learn step to populate optimizer state
        obs_dim = 4
        act_dim = 2
        bs = 4
        states = torch.randn(bs, obs_dim)
        actions = torch.randn(bs, act_dim)
        adv = torch.randn(bs, 1)
        old_probs = torch.randn(bs, 1)
        returns = torch.randn(bs, 1)
        rewards = torch.randn(bs, 1)
        old_values = torch.randn(bs, 1)
        agent.learn(states, actions, adv, old_probs, returns, rewards, old_values)

        # Update normalization
        agent.obs_rms.update(np.random.randn(10, obs_dim))
        agent.ret_rms.update(np.random.randn(20))
        agent.best_reward = 99.0

        agent.save(path=str(tmp_path))

        # Find saved dir
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs
        model_dir = subdirs[0]

        # Verify all files saved
        assert (model_dir / "config.json").exists()
        assert (model_dir / "actor.pth").exists()
        assert (model_dir / "critic.pth").exists()
        assert (model_dir / "actor_opt.pth").exists()
        assert (model_dir / "critic_opt.pth").exists()
        assert (model_dir / "train_state.json").exists()
        assert (model_dir / "obs_rms.npz").exists()
        assert (model_dir / "ret_rms.npz").exists()

        # Load
        loaded = PPO.from_pretrained(str(model_dir))
        assert loaded.gamma == agent.gamma
        assert loaded.best_reward == 99.0
        assert np.allclose(loaded.obs_rms.mean, agent.obs_rms.mean)
        assert np.allclose(loaded.ret_rms.var, agent.ret_rms.var)

        # Actor weights should match
        for p1, p2 in zip(agent.actor.parameters(), loaded.actor.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu(), atol=1e-6)

    def test_save_with_none_path(self, tmp_path, monkeypatch):
        """save(path=None) should use cwd."""
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        monkeypatch.chdir(tmp_path)
        agent.save(path=None)
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs

    def test_load_without_optimizer_files(self, tmp_path):
        """Load should succeed even if optimizer files are missing."""
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        agent.save(path=str(tmp_path))
        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        model_dir = subdirs[0]
        # Remove optional files
        (model_dir / "actor_opt.pth").unlink(missing_ok=True)
        (model_dir / "critic_opt.pth").unlink(missing_ok=True)
        (model_dir / "train_state.json").unlink(missing_ok=True)
        # Should still load
        loaded = PPO.from_pretrained(str(model_dir))
        assert isinstance(loaded, PPO)


# ===========================================================================
# PPO: get_param_env edge cases
# ===========================================================================


class TestGetParamEnv:
    def test_env_with_ref_signal(self):
        env = _DummyEnv()
        env.ref_signal = type("RefSignal", (), {})()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        params = agent.get_param_env()
        assert "ref_signal" in params["env"]["params"]

    def test_env_without_action_space_str(self):
        """Env whose action_space has no __str__ that causes error."""
        env = _DummyEnv()
        # Normal case: should populate action_space / observation_space
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        params = agent.get_param_env()
        assert "action_space" in params["env"]["params"]
        assert "observation_space" in params["env"]["params"]

    def test_env_without_observation_space_attr(self):
        """get_param_env should handle missing observation_space gracefully."""
        env = _DummyEnv()
        agent = PPO(env=env, max_episodes=1, rollout_len=4, num_epochs=1, batch_size=2)
        # Temporarily delete observation_space from env to hit the except branch
        saved = env.observation_space
        del env.observation_space
        try:
            params = agent.get_param_env()
            assert "observation_space" not in params["env"]["params"]
        finally:
            env.observation_space = saved


# ===========================================================================
# PPO: Vector env with action space that lacks .low / .high
# ===========================================================================


class TestVectorEnvFallbackBounds:
    def test_no_low_high(self, tmp_path):
        """Action space without .low/.high should use -1/+1 fallback."""
        env = _DummyVectorEnv(n_envs=2)
        # Remove low/high attributes
        env.action_space = type("A", (), {"shape": (2,)})
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=4,
            eval_freq=100,
            save_best_model=False,
        )
        # The train() will call _train_vector which should hit the except branch
        # for action bounds
        agent.train()


# ===========================================================================
# PPO: Vector env reset returning non-tuple
# ===========================================================================


class TestVectorEnvResetNonTuple:
    def test_reset_returns_only_obs(self, tmp_path):
        env = _DummyVectorEnv(n_envs=2)
        original_reset = env.reset

        def raw_reset():
            obs, _ = original_reset()
            return obs  # Return just obs, no info tuple

        env.reset = raw_reset
        agent = PPO(
            env=env,
            max_episodes=1,
            rollout_len=4,
            num_epochs=1,
            batch_size=4,
            eval_freq=100,
            save_best_model=False,
        )
        agent.train()
