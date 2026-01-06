import os
import tempfile
from typing import Tuple

import numpy as np
import torch

from tensoraerospace.agent.dsac import DSAC


class _TinySpace:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


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
        del action
        self._step_count += 1
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = 0.1
        terminated = self._step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _make_agent():
    env = _TinyEnv(obs_dim=3, act_dim=2, max_steps=3)
    agent = DSAC(
        env=env,
        batch_size=2,
        hidden_size=8,
        memory_capacity=32,
        num_quantiles=4,
        num_quantiles_exp=4,
        embedding_dim=4,
        learning_starts=0,
        device="cpu",
        automatic_entropy_tuning=True,
        log_every_updates=1000,
    )
    agent.writer = _NoOpWriter()
    return agent, env


def test_dsac_save_creates_expected_files():
    agent, _env = _make_agent()
    with tempfile.TemporaryDirectory() as d:
        run_dir = agent.save(d)
        files = set(os.listdir(run_dir))
        expected = {"config.json", "policy.pth", "critic.pth", "critic_target.pth"}
        assert expected.issubset(files)
        if agent.automatic_entropy_tuning:
            assert "log_alpha.pth" in files


def test_dsac_select_action_shapes_and_bounds():
    agent, env = _make_agent()
    state = np.zeros(env.observation_space.shape[0], dtype=np.float32)

    action = agent.select_action(state, evaluate=False)
    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0) and np.all(action <= 1.0)

    batch_states = torch.zeros((3, env.observation_space.shape[0]))
    batch_actions = agent.select_action_batch(
        batch_states, evaluate=False, return_tensor=True
    )
    assert batch_actions.shape == (3, env.action_space.shape[0])
    assert torch.all(batch_actions >= -1.0)
    assert torch.all(batch_actions <= 1.0)


def test_dsac_update_parameters_returns_losses():
    agent, env = _make_agent()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    for _ in range(4):
        state = np.zeros((obs_dim,), dtype=np.float32)
        action = np.zeros((act_dim,), dtype=np.float32)
        reward = 0.0
        next_state = np.ones((obs_dim,), dtype=np.float32)
        done = 0.0
        agent.memory.push(state, action, reward, next_state, done)

    losses = agent.update_parameters(agent.memory, batch_size=2, updates=0)
    for value in losses:
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_dsac_train_one_episode_adds_experience():
    agent, _env = _make_agent()
    before = len(agent.memory)
    agent.train(num_episodes=1)
    after = len(agent.memory)
    assert after > before


def test_dsac_eval_and_to_device_cpu_noop():
    agent, _env = _make_agent()
    agent = agent.to_device("cpu")
    assert agent.device.type == "cpu"
    # Ensure parameters live on CPU
    assert next(agent.policy.parameters()).device.type == "cpu"
    agent.eval()
    assert not agent.policy.training
    assert not agent.Z1.training

