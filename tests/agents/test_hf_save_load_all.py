"""Round-trip save/load/from_pretrained tests for all agents that support it.

Tests agents that did NOT already have dedicated save/load tests:
  DSAC, A3C, GAIL, MPC, ADP, ADHDP, HDP, PID, NARX, A2C-NARX (A2CLearner)

Each test:
  1. Creates agent with minimal config
  2. Runs one action to populate weights
  3. Saves to temp dir
  4. Loads via from_pretrained with local path
  5. Verifies weights match (or actions match)
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helper envs
# ---------------------------------------------------------------------------


class _ContinuousSpace:
    """Minimal Box-like space for continuous action/obs."""

    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = tuple(shape)
        self.low = np.full(self.shape, low, dtype=np.float32)
        self.high = np.full(self.shape, high, dtype=np.float32)

    def sample(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)


class _TinyContEnv:
    """Minimal continuous env for testing (obs_dim, act_dim configurable)."""

    def __init__(self, obs_dim=4, act_dim=1, max_steps=5):
        self.observation_space = _ContinuousSpace((obs_dim,))
        self.action_space = _ContinuousSpace((act_dim,))
        self._step_count = 0
        self.max_steps = int(max_steps)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        _ = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        obs = np.random.randn(self.observation_space.shape[0]).astype(np.float32) * 0.1
        reward = -0.1
        terminated = self._step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self

    def close(self):
        pass


class _DiscreteSpace:
    """Minimal Discrete-like space."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(0, self.n)


class _TinyDiscreteEnv:
    """Minimal discrete action env for DQN-like agents."""

    def __init__(self, obs_dim=4, num_actions=2, max_steps=5):
        self.observation_space = _ContinuousSpace((obs_dim,))
        self.action_space = _DiscreteSpace(num_actions)
        self.obs_dim = obs_dim
        self._step_count = 0
        self.max_steps = int(max_steps)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        reward = -0.1
        terminated = self._step_count >= self.max_steps
        return obs, reward, terminated, False, {}

    def close(self):
        pass


def _find_saved_subdir(base: Path) -> Path:
    """Return the timestamped subdirectory created by agent.save()."""
    if (base / "config.json").exists():
        return base
    subdirs = sorted(base.iterdir())
    assert len(subdirs) >= 1, f"No subdirectory found in {base}"
    return subdirs[0]


# ---------------------------------------------------------------------------
# DSAC
# ---------------------------------------------------------------------------


def test_dsac_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.dsac.dsac_flight import DSAC

    env = gym.make("Pendulum-v1")
    agent = DSAC(
        env=env,
        hidden_size=16,
        memory_capacity=100,
        device="cpu",
        num_quantiles=4,
        embedding_dim=16,
        learning_starts=0,
    )
    obs, _ = env.reset(seed=42)
    action1 = agent.select_action(obs, evaluate=True)

    save_dir = str(tmp_path / "dsac_test")
    run_dir = agent.save(save_dir)

    loaded = DSAC.from_pretrained(str(run_dir))
    action2 = loaded.select_action(obs, evaluate=True)
    np.testing.assert_array_almost_equal(action1, action2)
    env.close()


# ---------------------------------------------------------------------------
# A3C
# ---------------------------------------------------------------------------


def test_a3c_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.a3c.pytorch import Agent as A3CAgent

    def env_fn(worker_id):
        return _TinyContEnv(obs_dim=3, act_dim=1, max_steps=5)

    agent = A3CAgent(
        env_function=env_fn,
        gamma=0.99,
        lr=1e-4,
        max_episodes=1,
        max_ep_step=5,
        run_in_main=True,
    )

    # Compare network weights directly (choose_action samples stochastically)
    weights_before = {k: v.clone() for k, v in agent.gnet.state_dict().items()}

    save_dir = str(tmp_path / "a3c_test")
    run_dir = agent.save(save_dir)

    loaded = A3CAgent.from_pretrained(str(run_dir), env_function=env_fn)

    for key in weights_before:
        torch.testing.assert_close(
            weights_before[key],
            loaded.gnet.state_dict()[key],
            msg=f"A3C weight mismatch on key={key}",
        )


# ---------------------------------------------------------------------------
# GAIL
# ---------------------------------------------------------------------------


def test_gail_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.gail.model import GAIL

    env = _TinyContEnv(obs_dim=3, act_dim=1)
    # Expert data: random state-action pairs
    expert_data = np.random.randn(10, 3 + 1).astype(np.float32)

    agent = GAIL(
        env=env,
        learning_rate=1e-3,
        max_steps=5,
        mini_batch_size=2,
        epochs=1,
        data=expert_data,
    )

    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        dist1, val1 = agent.model(obs_t.to(agent.device))
        action1 = dist1.mean.cpu().numpy()

    save_dir = str(tmp_path / "gail_test")
    run_dir = agent.save(save_dir)

    loaded = GAIL.from_pretrained(str(run_dir), env=env, data=expert_data)
    with torch.no_grad():
        dist2, val2 = loaded.model(obs_t.to(loaded.device))
        action2 = dist2.mean.cpu().numpy()

    np.testing.assert_array_almost_equal(action1, action2, decimal=5)
    env.close()


# ---------------------------------------------------------------------------
# MPC
# ---------------------------------------------------------------------------


def test_mpc_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.mpc.mpc import MPCAgent

    env = gym.make("Pendulum-v1")
    agent = MPCAgent(
        env=env,
        horizon=5,
        iters=3,
        hidden_layers=(16, 16),
        memory_capacity=100,
        device="cpu",
    )

    # Get model parameter values before save
    params_before = {k: v.clone() for k, v in agent.model.state_dict().items()}

    save_dir = str(tmp_path / "mpc_test")
    run_dir = agent.save(save_dir)

    loaded = MPCAgent.from_pretrained(str(run_dir))

    # Verify model weights match
    for key in params_before:
        torch.testing.assert_close(
            params_before[key],
            loaded.model.state_dict()[key],
            msg=f"MPC model weight mismatch on key={key}",
        )
    env.close()


# ---------------------------------------------------------------------------
# ADP (design="adhdp", default)
# ---------------------------------------------------------------------------


def test_adp_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.adp.adp import ADP

    env = _TinyContEnv(obs_dim=4, act_dim=1)
    agent = ADP(
        env=env,
        hidden_size=16,
        device="cpu",
        exploration_std=0.0,
    )
    obs, _ = env.reset()
    action1 = agent.select_action(obs, evaluate=True)

    save_dir = str(tmp_path / "adp_test")
    run_dir_str = agent.save(save_dir)
    run_dir = Path(run_dir_str)

    loaded = ADP.from_pretrained(str(run_dir))
    action2 = loaded.select_action(obs, evaluate=True)
    np.testing.assert_array_almost_equal(action1, action2)
    env.close()


# ---------------------------------------------------------------------------
# ADHDP
# ---------------------------------------------------------------------------


def test_adhdp_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.adhdp.model import ADHDP

    env = _TinyContEnv(obs_dim=6, act_dim=1)
    agent = ADHDP(
        env=env,
        hidden_size=16,
        device="cpu",
        exploration_std=0.0,
    )
    obs, _ = env.reset()
    action1 = agent.select_action(obs, evaluate=True)

    save_dir = str(tmp_path / "adhdp_test")
    run_dir_str = agent.save(save_dir)
    run_dir = Path(run_dir_str)

    loaded = ADHDP.from_pretrained(str(run_dir))
    action2 = loaded.select_action(obs, evaluate=True)
    np.testing.assert_array_almost_equal(action1, action2)
    env.close()


# ---------------------------------------------------------------------------
# HDP (subclass of ADP with design="hdp")
# HDP requires env.reference_signal, env.initial_state, env.model.filt_A/B
# ---------------------------------------------------------------------------


class _HDPModel:
    """Minimal model stub with linearized dynamics matrices."""

    def __init__(self, state_dim=4, act_dim=1):
        self.filt_A = np.eye(state_dim, dtype=np.float32) * 0.99
        self.filt_B = np.ones((state_dim, act_dim), dtype=np.float32) * 0.01


class _HDPEnv(_TinyContEnv):
    """TinyContEnv with reference_signal and model attributes required by HDP."""

    def __init__(self, obs_dim=6, act_dim=1, max_steps=5, n_steps=100):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim, max_steps=max_steps)
        self.initial_state = np.zeros(4, dtype=np.float32)
        self.reference_signal = np.zeros((1, n_steps), dtype=np.float32)
        self.model = _HDPModel(state_dim=4, act_dim=act_dim)
        self.dt = 0.1


def test_hdp_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.hdp.model import HDP

    env = _HDPEnv(obs_dim=6, act_dim=1)
    agent = HDP(
        env=env,
        hidden_size=16,
        device="cpu",
        exploration_std=0.0,
    )
    obs, _ = env.reset()
    action1 = agent.select_action(obs, evaluate=True)

    save_dir = str(tmp_path / "hdp_test")
    run_dir_str = agent.save(save_dir)
    run_dir = Path(run_dir_str)

    loaded = HDP.from_pretrained(str(run_dir))
    action2 = loaded.select_action(obs, evaluate=True)
    np.testing.assert_array_almost_equal(action1, action2)
    env.close()


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------


def test_pid_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.pid import PID

    env = _TinyContEnv(obs_dim=4, act_dim=1)
    agent = PID(env=env, kp=1.5, ki=0.3, kd=0.1, dt=0.01)

    # PID select_action takes (setpoint, measurement)
    action1 = agent.select_action(1.0, 0.5)

    save_dir = str(tmp_path / "pid_test")
    run_dir = agent.save(save_dir)

    loaded = PID.from_pretrained(str(run_dir))
    # Reset prev state for fair comparison
    loaded.integral = 0
    loaded.prev_error = 0
    loaded.prev_measurement = 0
    action2 = loaded.select_action(1.0, 0.5)

    assert abs(action1 - action2) < 1e-6, f"PID actions differ: {action1} vs {action2}"
    assert loaded.kp == agent.kp
    assert loaded.ki == agent.ki
    assert loaded.kd == agent.kd
    assert loaded.dt == agent.dt
    env.close()


# ---------------------------------------------------------------------------
# NARX
# ---------------------------------------------------------------------------


def test_narx_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.narx.model import NARX

    model = NARX(input_size=3, hidden_size=16, output_size=1)

    # Forward pass to verify weights
    inp = torch.randn(3)
    last_out = torch.zeros(1)
    with torch.no_grad():
        out1 = model(inp, last_out).numpy()

    save_dir = str(tmp_path / "narx_test")
    run_dir = model.save(save_dir)

    loaded = NARX.from_pretrained(str(run_dir))
    with torch.no_grad():
        out2 = loaded(inp, last_out).numpy()

    np.testing.assert_array_almost_equal(out1, out2)


# ---------------------------------------------------------------------------
# A2C-NARX (A2CLearner from a2c/narx.py)
# ---------------------------------------------------------------------------


def test_a2c_narx_save_load_roundtrip(tmp_path):
    from tensoraerospace.agent.a2c.narx import A2CLearner, Actor, Critic

    state_dim = 4
    n_actions = 1
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    agent = A2CLearner(
        actor=actor,
        critic=critic,
        gamma=0.9,
        entropy_beta=0.01,
        actor_lr=4e-4,
        critic_lr=4e-3,
        device="cpu",
    )

    # Get action distribution mean
    obs = torch.randn(1, state_dim)
    with torch.no_grad():
        dist1 = agent.actor(obs)
        action1 = dist1.mean.numpy()

    save_dir = str(tmp_path / "a2c_narx_test")
    run_dir = agent.save(save_dir)

    loaded = A2CLearner.from_pretrained(str(run_dir))
    with torch.no_grad():
        dist2 = loaded.actor(obs)
        action2 = dist2.mean.numpy()

    np.testing.assert_array_almost_equal(action1, action2)
