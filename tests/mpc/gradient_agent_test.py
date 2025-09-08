import json
import os
import shutil
import tempfile

import numpy as np
import torch

from tensoraerospace.agent.mpc.gradient import MPCOptimizationAgent, Net


class _DummyEnv:
    def __init__(self):
        self._step = 0
        # minimal gymnasium-like API
        self.action_space = self
        self.observation_space = self

    def reset(self):
        self._step = 0
        return np.array([0.0, 0.0]), {}

    def step(self, action):
        self._step += 1
        obs = np.array([float(self._step), 0.0])
        reward = 1.0
        terminated = self._step >= 1
        truncated = False
        return obs, reward, terminated, truncated, {}

    # action_space helpers
    def sample(self):
        return np.array([[0.0]])

    @property
    def unwrapped(self):
        return self

    def __class__(self):  # just to be safe
        return self.__class__


def _cost_fn(next_state, action, *args):
    # simple L2
    return (next_state**2).sum() + (action**2).sum()


def test_mpc_agent_choose_action_and_save_load(tmp_path: str = None):
    env = _DummyEnv()
    model = Net()
    agent = MPCOptimizationAgent(
        gamma=0.99,
        action_dim=1,
        observation_dim=2,
        model=model,
        cost_function=_cost_fn,
        env=env,
        lr=1e-3,
        optimization_lr=0.5,
    )

    state = np.array([0.0, 0.0])
    action = agent.choose_action(state, rollout=2, horizon=1)
    assert isinstance(action, np.ndarray)

    # get params for env/config
    config = agent.get_param_env()
    assert "env" in config and "policy" in config

    # save and load
    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        # find the created folder
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        assert subdirs, "No save directory created"
        load_dir = subdirs[0]
        agent.load(load_dir)
