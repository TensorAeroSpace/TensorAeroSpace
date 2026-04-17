"""Small targeted tests filling the last uncovered branches across several
agents: ADHDP ``_load_from_dir(load_gradients=True)``, ``from_pretrained``
local-path-not-found handling, and similar.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# --- ADHDP -----------------------------------------------------------------
class _TinyEnv:
    """Minimal env that looks enough like a Gym env for ADHDP/ADP training."""

    def __init__(self, obs_dim: int = 4, act_dim: int = 1):
        self.observation_space = _Space((obs_dim,))
        self.action_space = _Space((act_dim,))
        self._obs_dim = obs_dim
        self._step_count = 0
        # ADHDP's ``get_param_env`` reads ``env.unwrapped.__class__`` for its
        # serialised env name, so we expose ``unwrapped`` as self.
        self.unwrapped = self

    def reset(self, seed=None, options=None):
        self._step_count = 0
        return np.zeros(self._obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self._obs_dim).astype(np.float32) * 0.01
        reward = -float(np.sum(np.square(action)))
        term = self._step_count > 50
        return obs, reward, term, False, {}


class _Space:
    def __init__(self, shape):
        self.shape = shape
        self.low = -np.ones(shape[0], dtype=np.float32)
        self.high = np.ones(shape[0], dtype=np.float32)


def test_adhdp_load_with_gradients_path(tmp_path):
    from tensoraerospace.agent.adhdp.model import ADHDP

    env = _TinyEnv(obs_dim=4, act_dim=1)
    agent = ADHDP(env=env, device="cpu")
    run_dir = agent.save(path=tmp_path, save_gradients=True)
    assert (Path(run_dir) / "actor_optim.pth").exists()

    # Exercise _load_from_dir with load_gradients=True — restores optim states.
    loaded = ADHDP._load_from_dir(run_dir, load_gradients=True)
    assert loaded is not None

    # from_pretrained with a local directory returns the same agent.
    loaded2 = ADHDP.from_pretrained(str(run_dir))
    assert loaded2 is not None


def test_adhdp_from_pretrained_missing_local_path_raises(tmp_path):
    from tensoraerospace.agent.adhdp.model import ADHDP

    bad = tmp_path / "does-not-exist-for-sure"
    # Path starts with '/', so it's treated as a file-system path and must exist.
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        ADHDP.from_pretrained(str(bad))


# --- DDPG ------------------------------------------------------------------
def test_ddpg_from_pretrained_missing_local_path_raises(tmp_path):
    from tensoraerospace.agent.ddpg.model import DDPG

    bad = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        DDPG.from_pretrained(str(bad))


# --- GAIL ------------------------------------------------------------------
def test_gail_from_pretrained_missing_local_path_raises(tmp_path):
    from tensoraerospace.agent.gail.model import GAIL

    with pytest.raises(FileNotFoundError):
        GAIL.from_pretrained(str(tmp_path / "missing-path"))


# --- PPO -------------------------------------------------------------------
# (PPO.from_pretrained has a different shape — it delegates to HuggingFace
# rather than short-circuiting on local paths, so the corresponding error
# path is covered by other tests.)


# --- IMGDHP ----------------------------------------------------------------
def test_imgdhp_agent_reset_clears_rolling_history():
    from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig

    cfg = IMGDHPConfig(
        actor_hidden=(8, 8), critic_hidden=(8, 8),
        actor_lr=1e-3, critic_lr=1e-3,
        track_Q=[1.0], u_max=1.0, seed=0,
        warmup_steps=1,
    )
    agent = IMGDHPAgent(
        n_obs=2, n_action=1, reference_size=1, tracking_indices=[0],
        config=cfg,
    )
    # A predict+learn cycle should populate the rolling buffer; reset clears it.
    ref = np.zeros((1, 10))
    _ = agent.predict(np.array([0.1, 0.0]), ref, 0)
    agent.learn(np.array([0.11, 0.0]), ref, 0)
    # After learn(), _y_tm1 must be set.
    assert agent._y_tm1 is not None
    agent.reset()
    assert agent._y_tm1 is None
    assert agent._u_tm1 is None


# --- ET-DHP event trigger edge cases --------------------------------------
def test_et_dhp_event_trigger_threshold_floor_honoured():
    from tensoraerospace.agent.et_dhp.event_trigger import EventTrigger

    et = EventTrigger(rho=0.2, trigger_first_step=False, min_floor=0.5)
    # First evaluation: norm=0 (state all zeros) is below the floor, so no trigger.
    got = et.should_trigger(np.zeros(3), step=0)
    assert got is False
    # A state with norm > floor triggers.
    got = et.should_trigger(np.array([1.0, 0.0, 0.0]), step=1)
    assert got is True


def test_et_dhp_event_trigger_rho_validated():
    from tensoraerospace.agent.et_dhp.event_trigger import EventTrigger

    with pytest.raises(ValueError, match="rho"):
        EventTrigger(rho=0.0)
    with pytest.raises(ValueError, match="rho"):
        EventTrigger(rho=0.5)
    with pytest.raises(ValueError, match="rho"):
        EventTrigger(rho=0.7)


def test_et_dhp_event_trigger_reset_rearms():
    from tensoraerospace.agent.et_dhp.event_trigger import EventTrigger

    et = EventTrigger(rho=0.2, trigger_first_step=True)
    assert et.should_trigger(np.array([1.0]), 0) is True
    et.reset()
    # After reset the first call triggers again.
    assert et.should_trigger(np.array([1.0]), 0) is True


# --- SAC save/load missing branches ---------------------------------------
def test_sac_close_before_train_is_safe():
    import gymnasium as gym
    from tensoraerospace.agent.sac import SAC

    env = gym.make("Pendulum-v1")
    agent = SAC(env, hidden_size=8, batch_size=4, memory_capacity=100, device="cpu")
    agent.close()
    env.close()
