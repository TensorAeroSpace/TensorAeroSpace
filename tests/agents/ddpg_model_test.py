import numpy as np
import torch

from tensoraerospace.agent.ddpg.model import (
    DDPG,
    OUNoise,
    PolicyNetwork,
    ReplayBuffer,
    ValueNetwork,
)


class _DummySpace:
    def __init__(self, shape, low=-1.0, high=1.0):
        self.shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)


class _FakeEnv:
    def __init__(self, obs_dim=3, act_dim=1):
        self.observation_space = _DummySpace((obs_dim,))
        self.action_space = _DummySpace((act_dim,))

    def reset(self):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return state, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        next_state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = float(1.0 - np.linalg.norm(action))
        terminated = False
        truncated = False
        info: dict = {}
        return next_state, reward, terminated, truncated, info


def test_replay_buffer_push_and_sample():
    buf = ReplayBuffer(capacity=10)
    s = np.array([0.0], dtype=np.float32)
    a = np.array([0.1], dtype=np.float32)
    for _ in range(5):
        buf.push(s, a, 1.0, s, False)
    assert len(buf) == 5
    state, _, _, _, _ = buf.sample(batch_size=2)
    assert state.shape[0] == 2


def test_ou_noise_shapes_and_clip():
    space = _DummySpace(shape=(1,), low=-0.5, high=0.5)
    ou = OUNoise(space, max_sigma=0.2, min_sigma=0.1, decay_period=10)
    ou.reset()
    base_action = np.array([0.6], dtype=np.float32)
    noisy = ou.get_action(base_action, t=5)
    assert noisy.shape == (1,)
    assert np.all(noisy <= space.high) and np.all(noisy >= -space.high)


def test_networks_forward_shapes():
    obs_dim, act_dim, hidden = 3, 1, 8
    value = ValueNetwork(obs_dim, act_dim, hidden)
    policy = PolicyNetwork(obs_dim, act_dim, hidden)
    s = torch.randn(4, obs_dim)
    a = torch.randn(4, act_dim)
    q = value(s, a)
    assert q.shape == (4, 1)
    act = policy(s)
    assert act.shape == (4, act_dim)


def test_ddpg_update_smoke():
    env = _FakeEnv(obs_dim=3, act_dim=1)
    agent = DDPG(env, value_lr=1e-3, policy_lr=1e-3, replay_buffer_size=64)
    # Fill replay buffer to enable update
    s = np.zeros(env.observation_space.shape[0], dtype=np.float32)
    a = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(32):
        agent.replay_buffer.push(s, a, 0.0, s, False)
    # Should run one update without errors
    agent.ddpg_update(batch_size=16)
