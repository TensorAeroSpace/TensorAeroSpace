"""Extra SAC coverage tests for the static helpers, close(), and the
vectorised-env training loop."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from tensoraerospace.agent.sac import SAC


@pytest.fixture
def sac_agent():
    env = gym.make("Pendulum-v1")
    agent = SAC(env, hidden_size=16, batch_size=4, memory_capacity=500, device="cpu")
    yield agent
    agent.close()
    env.close()


# ---- static helpers -------------------------------------------------------
def test_filter_kwargs_for_init_drops_unexpected():
    class Dummy:
        def __init__(self, a: int, b: str = "x"):  # noqa: D401
            pass

    filtered = SAC._filter_kwargs_for_init(Dummy, {"a": 1, "b": "y", "c": "extra"})
    assert filtered == {"a": 1, "b": "y"}


def test_filter_kwargs_for_init_keeps_everything_when_varkw():
    class Dummy:
        def __init__(self, a: int, **kw):
            pass

    kwargs = {"a": 1, "anything": 2}
    assert SAC._filter_kwargs_for_init(Dummy, kwargs) == kwargs


def test_filter_kwargs_for_init_unknown_signature_is_passthrough():
    # ``type`` has a signature that raises ValueError when introspected in some
    # environments — the helper should just return the kwargs unchanged.
    class Weird:
        __slots__ = ()

    # Use a C-extension-like object that raises on signature inspection.
    kwargs = {"x": 1}
    got = SAC._filter_kwargs_for_init(object, kwargs)
    # object.__init__ is accepted; return type is a dict.
    assert isinstance(got, dict)


# ---- close() is idempotent -----------------------------------------------
def test_close_is_idempotent(sac_agent):
    sac_agent.close()
    # A second close should still be safe.
    sac_agent.close()


# ---- train_vector on a tiny torch-vectorised env -------------------------
class _TinyVecEnv:
    """Minimal torch-based vectorised env to exercise SAC.train_vector."""

    def __init__(self, num_envs: int = 2, obs_dim: int = 3, act_dim: int = 1):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.auto_reset = False

        # gym-spec style action_space has a .shape attribute; SAC.train_vector
        # only reads its first element.
        class _Space:
            def __init__(self, shape):
                self.shape = shape
                self.low = -np.ones(shape[0], dtype=np.float32)
                self.high = np.ones(shape[0], dtype=np.float32)

        self.action_space = _Space((act_dim,))
        self.observation_space = _Space((obs_dim,))

    def reset(self):
        self._t = 0
        obs = torch.zeros((self.num_envs, self.obs_dim), dtype=torch.float32)
        return obs, {}

    def step(self, actions):
        self._t += 1
        n = self.num_envs
        next_obs = torch.randn(n, self.obs_dim)
        reward = torch.ones(n, dtype=torch.float32) * 0.1
        # Terminate one env per call on a rolling schedule.
        terminated = torch.zeros(n, dtype=torch.bool)
        truncated = torch.zeros(n, dtype=torch.bool)
        terminated[self._t % n] = True
        return next_obs, reward, terminated, truncated, {}


def test_train_vector_short_run(tmp_path):
    env = _TinyVecEnv(num_envs=2, obs_dim=3, act_dim=1)
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=200, device="cpu")
    agent.train_vector(
        total_steps=10,
        warmup_steps=2,
        log_every=3,
        reward_window=10,
        save_best=False,
    )
    agent.close()


def test_train_vector_rejects_bad_total_steps():
    env = _TinyVecEnv(num_envs=2)
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=200, device="cpu")
    with pytest.raises(ValueError, match="total_steps"):
        agent.train_vector(total_steps=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        agent.train_vector(total_steps=5, warmup_steps=-1)
    agent.close()


def test_train_vector_rejects_non_tensor_obs():
    class BadEnv(_TinyVecEnv):
        def reset(self):
            return np.zeros((self.num_envs, self.obs_dim)), {}

    env = BadEnv(num_envs=2)
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=200, device="cpu")
    with pytest.raises(TypeError, match="torch Tensor"):
        agent.train_vector(total_steps=5, warmup_steps=1)
    agent.close()


def test_train_vector_rejects_wrong_obs_shape():
    class BadEnv(_TinyVecEnv):
        def reset(self):
            return torch.zeros((self.obs_dim,)), {}  # 1-D, not 2-D.

    env = BadEnv(num_envs=2)
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=200, device="cpu")
    with pytest.raises(ValueError, match="shape"):
        agent.train_vector(total_steps=5, warmup_steps=1)
    agent.close()


def test_train_vector_auto_reset_branch():
    env = _TinyVecEnv(num_envs=2)
    env.auto_reset = True
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=200, device="cpu")
    agent.train_vector(total_steps=8, warmup_steps=1, log_every=2)
    agent.close()
