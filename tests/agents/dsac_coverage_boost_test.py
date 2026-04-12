"""Extra coverage tests for tensoraerospace/agent/dsac/dsac_flight.py.

Targets uncovered lines: select_action(evaluate), update_parameters branches,
save/load roundtrip, train_vector, get_param_env, to_device, eval, close,
push_to_hub guard, risk distortions, hidden_layers validation, and edge cases.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import pytest
import torch

from tensoraerospace.agent.dsac.dsac_flight import (
    DSAC,
    _softplus_stable,
    calculate_huber_loss,
    quantile_huber_loss,
)
from tensoraerospace.agent.dsac.risk_distortions import (
    cpw,
    cvar,
    neutral,
    wang,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinySpace:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)


class _TinyEnv:
    def __init__(self, obs_dim: int = 3, act_dim: int = 2, max_steps: int = 3):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self.max_steps = max_steps

    def reset(self):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self.observation_space.shape[0]).astype(np.float32)
        reward = 0.1
        terminated = self._step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self


class _NoOpWriter:
    def add_scalar(self, *a, **kw):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _make_agent(*, obs_dim=3, act_dim=2, max_steps=3, **kwargs):
    env = _TinyEnv(obs_dim=obs_dim, act_dim=act_dim, max_steps=max_steps)
    defaults = dict(
        batch_size=4,
        hidden_size=8,
        memory_capacity=64,
        num_quantiles=4,
        num_quantiles_exp=4,
        embedding_dim=4,
        learning_starts=0,
        device="cpu",
        automatic_entropy_tuning=True,
        log_every_updates=1,
        warmup_action_scale=0.5,
    )
    defaults.update(kwargs)
    agent = DSAC(env=env, **defaults)
    agent.writer = _NoOpWriter()
    return agent, env


def _fill_memory(agent, n=16):
    obs_dim = agent.env.observation_space.shape[0]
    act_dim = agent.env.action_space.shape[0]
    for _ in range(n):
        s = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.randn(act_dim).astype(np.float32)
        ns = np.random.randn(obs_dim).astype(np.float32)
        agent.memory.push(s, a, float(np.random.randn()), ns, 0.0)


# ---------------------------------------------------------------------------
# Risk distortion coverage (lines 15, 22, 32, 37-39, 44)
# ---------------------------------------------------------------------------


def test_risk_distortion_cvar():
    tau = torch.linspace(0.01, 0.99, 50)
    result = cvar(tau, 0.5)
    assert result.shape == tau.shape
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)


def test_risk_distortion_cpw():
    tau = torch.linspace(0.01, 0.99, 50)
    result = cpw(tau, 0.71)
    assert result.shape == tau.shape
    assert torch.all(torch.isfinite(result))


def test_risk_distortion_wang():
    tau = torch.linspace(0.01, 0.99, 50)
    result = wang(tau, 0.0)
    # wang with xi=0 should be close to identity
    assert torch.allclose(result, tau, atol=1e-4)


def test_risk_distortion_neutral():
    tau = torch.linspace(0.0, 1.0, 10)
    result = neutral(tau, 0.5)
    assert torch.allclose(result, tau)


# ---------------------------------------------------------------------------
# select_action in evaluate mode (line 298-299)
# ---------------------------------------------------------------------------


def test_select_action_evaluate_mode():
    agent, env = _make_agent()
    state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    action = agent.select_action(state, evaluate=True)
    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_select_action_with_exploration_noise():
    agent, env = _make_agent(exploration_noise_std=0.1)
    state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    # Test exploration noise in both eval and non-eval
    action_eval = agent.select_action(state, evaluate=True)
    action_noeval = agent.select_action(state, evaluate=False)
    assert action_eval.shape == env.action_space.shape
    assert action_noeval.shape == env.action_space.shape


# ---------------------------------------------------------------------------
# select_action_batch in evaluate mode (line 325-326)
# ---------------------------------------------------------------------------


def test_select_action_batch_evaluate():
    agent, env = _make_agent()
    states = torch.zeros((5, env.observation_space.shape[0]))
    actions = agent.select_action_batch(states, evaluate=True, return_tensor=True)
    assert actions.shape == (5, env.action_space.shape[0])
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_select_action_batch_3d_input():
    agent, env = _make_agent()
    states = torch.zeros((5, 1, env.observation_space.shape[0]))
    actions = agent.select_action_batch(states, evaluate=False, return_tensor=False)
    assert isinstance(actions, np.ndarray)
    assert actions.shape == (5, env.action_space.shape[0])


def test_select_action_batch_exploration_noise():
    agent, env = _make_agent(exploration_noise_std=0.2)
    states = torch.zeros((3, env.observation_space.shape[0]))
    actions = agent.select_action_batch(states, evaluate=False, return_tensor=True)
    assert actions.shape == (3, env.action_space.shape[0])


# ---------------------------------------------------------------------------
# update_parameters with automatic_entropy_tuning OFF (line 456-457)
# ---------------------------------------------------------------------------


def test_update_parameters_no_auto_entropy():
    agent, env = _make_agent(automatic_entropy_tuning=False, alpha=0.5)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5
    for v in losses:
        assert np.isfinite(v)


def test_update_parameters_with_reward_clip():
    agent, env = _make_agent(reward_clip=1.0)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5


def test_update_parameters_target_update_interval():
    """Cover the soft_update branch via target_update_interval."""
    agent, env = _make_agent(target_update_interval=1)
    _fill_memory(agent)
    # updates=0 => 0 % 1 == 0, triggers soft update
    losses_a = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    # updates=1 => 1 % 1 == 0, also triggers
    losses_b = agent.update_parameters(agent.memory, batch_size=4, updates=1)
    assert len(losses_a) == 5 and len(losses_b) == 5


def test_update_parameters_min_alpha_clamp():
    """Cover the min_alpha > 0 branch (line 450-452)."""
    agent, env = _make_agent(min_alpha=0.01)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert agent.alpha >= 0.01


def test_update_parameters_max_grad_norm():
    """Cover the max_grad_norm branch (not used in dsac_flight update but agent carries it)."""
    agent, env = _make_agent(max_grad_norm=0.5)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5


# ---------------------------------------------------------------------------
# train() single-episode loop (lines 521-525, 543, 559-561)
# ---------------------------------------------------------------------------


def test_train_one_episode():
    agent, env = _make_agent(max_steps=3, learning_starts=0)
    before = len(agent.memory)
    agent.train(num_episodes=1)
    assert len(agent.memory) > before


def test_train_with_save_best():
    agent, env = _make_agent(max_steps=3, learning_starts=0)
    with tempfile.TemporaryDirectory() as d:
        agent.train(num_episodes=2, save_best=True, save_path=d)
        # At least one save should have happened
        assert len(os.listdir(d)) >= 1


# ---------------------------------------------------------------------------
# train_vector (lines 574-742) -- needs tensor-returning env
# ---------------------------------------------------------------------------


class _VecTensorEnv:
    """Minimal vectorized environment returning torch Tensors."""

    def __init__(self, num_envs=2, obs_dim=3, act_dim=2, max_steps=5):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_steps = max_steps
        self._step = 0
        self.auto_reset = True

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self._step = 0
        return torch.zeros((self.num_envs, self.obs_dim)), {}

    def step(self, actions):
        self._step += 1
        obs = torch.randn((self.num_envs, self.obs_dim))
        reward = torch.ones((self.num_envs,)) * 0.1
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool)
        truncated = torch.zeros((self.num_envs,), dtype=torch.bool)
        if self._step >= self.max_steps:
            terminated[:] = True
            self._step = 0
        return obs, reward, terminated, truncated, {}


def test_train_vector_basic():
    env = _VecTensorEnv(num_envs=2, obs_dim=3, act_dim=2, max_steps=4)
    agent = DSAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        memory_capacity=128,
        num_quantiles=4,
        num_quantiles_exp=4,
        embedding_dim=4,
        learning_starts=0,
        warmup_action_scale=0.5,
        device="cpu",
        log_every_updates=1000,
    )
    agent.writer = _NoOpWriter()
    agent.train_vector(
        total_steps=20,
        warmup_steps=5,
        log_every=10,
        reward_window=5,
    )
    assert len(agent.memory) > 0


def test_train_vector_reward_clip():
    env = _VecTensorEnv(num_envs=2, obs_dim=3, act_dim=2, max_steps=4)
    agent = DSAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        memory_capacity=128,
        num_quantiles=4,
        num_quantiles_exp=4,
        embedding_dim=4,
        learning_starts=0,
        reward_clip=0.5,
        device="cpu",
        log_every_updates=1000,
    )
    agent.writer = _NoOpWriter()
    agent.train_vector(total_steps=10, warmup_steps=2, log_every=5)
    assert len(agent.memory) > 0


def test_train_vector_save_best():
    env = _VecTensorEnv(num_envs=2, obs_dim=3, act_dim=2, max_steps=3)
    agent = DSAC(
        env=env,
        batch_size=4,
        hidden_size=8,
        memory_capacity=128,
        num_quantiles=4,
        num_quantiles_exp=4,
        embedding_dim=4,
        learning_starts=0,
        device="cpu",
        log_every_updates=1000,
    )
    agent.writer = _NoOpWriter()
    with tempfile.TemporaryDirectory() as d:
        agent.train_vector(
            total_steps=15,
            warmup_steps=2,
            log_every=5,
            save_best=True,
            save_path=d,
        )


def test_train_vector_rejects_non_tensor():
    env = _TinyEnv(obs_dim=3, act_dim=2, max_steps=3)
    agent, _ = _make_agent()
    with pytest.raises(TypeError):
        agent.train_vector(total_steps=5, warmup_steps=0)


def test_train_vector_rejects_1d_obs():
    class _Bad1DEnv:
        observation_space = _TinySpace((3,))
        action_space = _TinySpace((2,))
        unwrapped = None
        auto_reset = False

        def reset(self):
            return torch.zeros((3,)), {}

        def step(self, a):
            return (
                torch.zeros((3,)),
                torch.tensor(0.0),
                torch.tensor(False),
                torch.tensor(False),
                {},
            )

    agent = DSAC(
        env=_Bad1DEnv(), batch_size=4, hidden_size=8, learning_starts=0, device="cpu"
    )
    agent.writer = _NoOpWriter()
    with pytest.raises(ValueError, match="obs of shape"):
        agent.train_vector(total_steps=2, warmup_steps=0)


# ---------------------------------------------------------------------------
# get_param_env (lines 744-790)
# ---------------------------------------------------------------------------


def test_get_param_env_keys():
    agent, env = _make_agent()
    cfg = agent.get_param_env()
    assert "env" in cfg and "policy" in cfg
    assert "name" in cfg["env"]
    assert "params" in cfg["policy"]
    pp = cfg["policy"]["params"]
    assert "gamma" in pp
    assert "tau" in pp
    assert "risk_distortion" in pp


# ---------------------------------------------------------------------------
# save / load roundtrip (lines 796-835, 905-1033)
# ---------------------------------------------------------------------------


def test_save_load_roundtrip():
    agent, env = _make_agent()
    _fill_memory(agent)
    agent.update_parameters(agent.memory, batch_size=4, updates=0)

    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(path=d, save_gradients=True)
        assert (run_dir / "config.json").exists()
        assert (run_dir / "policy.pth").exists()
        assert (run_dir / "critic.pth").exists()

        loaded = DSAC.from_pretrained(str(run_dir), load_gradients=True)
        assert loaded.device.type == "cpu"

        state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
        a1 = agent.select_action(state, evaluate=True)
        a2 = loaded.select_action(state, evaluate=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


def test_save_without_gradients():
    agent, env = _make_agent()
    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(path=d, save_gradients=False)
        files = set(os.listdir(run_dir))
        assert "policy_optim.pth" not in files


def test_from_pretrained_bad_path_raises():
    with pytest.raises(FileNotFoundError):
        DSAC.from_pretrained("./nonexistent_dir_xyz")


# ---------------------------------------------------------------------------
# to_device (lines 838-867)
# ---------------------------------------------------------------------------


def test_to_device_cpu():
    agent, _ = _make_agent()
    _fill_memory(agent)
    agent.update_parameters(agent.memory, batch_size=4, updates=0)
    agent = agent.to_device("cpu")
    assert agent.device.type == "cpu"
    assert next(agent.policy.parameters()).device.type == "cpu"


# ---------------------------------------------------------------------------
# eval mode (lines 869-876)
# ---------------------------------------------------------------------------


def test_eval_mode():
    agent, _ = _make_agent()
    agent.eval()
    assert not agent.policy.training
    assert not agent.Z1.training
    assert not agent.Z2.training
    assert not agent.Z1_target.training
    assert not agent.Z2_target.training


# ---------------------------------------------------------------------------
# close (lines 1082-1089)
# ---------------------------------------------------------------------------


def test_close():
    agent, _ = _make_agent()
    agent.close()  # should not raise


# ---------------------------------------------------------------------------
# hidden_layers validation (lines 178-183)
# ---------------------------------------------------------------------------


def test_hidden_layers_validation():
    env = _TinyEnv()
    with pytest.raises(ValueError, match="equal hidden units"):
        DSAC(env=env, hidden_layers=[32, 64], device="cpu")


def test_hidden_layers_empty_raises():
    env = _TinyEnv()
    with pytest.raises(ValueError, match="non-empty"):
        DSAC(env=env, hidden_layers=[], device="cpu")


# ---------------------------------------------------------------------------
# NormalPolicyNet.get_std (line 49-51 of flight_actor.py)
# ---------------------------------------------------------------------------


def test_policy_get_std():
    agent, env = _make_agent()
    state = torch.zeros((1, env.observation_space.shape[0]))
    std = agent.policy.get_std(state)
    assert std.shape == (1, env.action_space.shape[0])
    assert torch.all(std > 0.0)


# ---------------------------------------------------------------------------
# _filter_kwargs with **kwargs in init
# ---------------------------------------------------------------------------


def test_filter_kwargs_passthrough_varkw():
    class Open:
        def __init__(self, **kwargs):
            pass

    result = DSAC._filter_kwargs_for_init(Open, {"a": 1, "b": 2})
    assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Different risk_distortion types
# ---------------------------------------------------------------------------


def test_agent_with_cvar_distortion():
    agent, env = _make_agent(risk_distortion="cvar", risk_measure=0.5)
    state = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    action = agent.select_action(state)
    assert action.shape == env.action_space.shape


def test_agent_with_cpw_distortion():
    agent, env = _make_agent(risk_distortion="cpw", risk_measure=0.71)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5


def test_agent_with_wang_distortion():
    agent, env = _make_agent(risk_distortion="wang", risk_measure=0.0)
    _fill_memory(agent)
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5


# ---------------------------------------------------------------------------
# Logging branches in update_parameters (lines 465-489)
# ---------------------------------------------------------------------------


def test_update_parameters_with_logging():
    """Cover the writer.add_scalar calls inside update_parameters."""
    agent, env = _make_agent(log_every_updates=1)
    _fill_memory(agent, n=16)
    # updates=0 => 0 % 1 == 0 triggers logging
    losses = agent.update_parameters(agent.memory, batch_size=4, updates=0)
    assert len(losses) == 5


# ---------------------------------------------------------------------------
# train_vector edge: total_steps < 1, warmup_steps < 0
# ---------------------------------------------------------------------------


def test_train_vector_bad_total_steps():
    env = _VecTensorEnv(num_envs=1, obs_dim=3, act_dim=2)
    agent = DSAC(env=env, batch_size=4, hidden_size=8, learning_starts=0, device="cpu")
    agent.writer = _NoOpWriter()
    with pytest.raises(ValueError, match="total_steps"):
        agent.train_vector(total_steps=0)


def test_train_vector_bad_warmup_steps():
    env = _VecTensorEnv(num_envs=1, obs_dim=3, act_dim=2)
    agent = DSAC(env=env, batch_size=4, hidden_size=8, learning_starts=0, device="cpu")
    agent.writer = _NoOpWriter()
    with pytest.raises(ValueError, match="warmup_steps"):
        agent.train_vector(total_steps=5, warmup_steps=-1)


# ---------------------------------------------------------------------------
# push_to_hub guard (lines 1070-1079)
# ---------------------------------------------------------------------------


def test_push_to_hub_save_creates_dir():
    """Test that push_to_hub calls save() (just test save portion without actual upload)."""
    agent, _ = _make_agent()
    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(path=d, save_gradients=True)
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()
