import numpy as np
import torch

from tensoraerospace.agent.sac.sac import SAC


class _TinySpace:
    def __init__(self, shape):
        import numpy as _np

        self.shape = shape
        self.high = _np.ones(shape, dtype=_np.float32)
        self.low = -_np.ones(shape, dtype=_np.float32)


class _TinyEnv:
    def __init__(self, obs_dim=3, act_dim=2):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))

    def reset(self):
        return np.zeros(self.observation_space.shape[0]), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape[0])
        reward = 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}


class _StubMemory:
    def __init__(self, obs_dim=3, act_dim=2, batch=4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch = batch

    def sample(self, batch_size):
        b = self.batch
        s = np.zeros((b, self.obs_dim), dtype=np.float32)
        a = np.zeros((b, self.act_dim), dtype=np.float32)
        r = np.zeros((b,), dtype=np.float32)
        ns = np.zeros((b, self.obs_dim), dtype=np.float32)
        m = np.ones((b,), dtype=np.float32)
        return s, a, r, ns, m


def test_sac_select_and_update_smoke():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    agent = SAC(
        env=env,
        batch_size=4,
        hidden_size=16,
        device="cpu",
        automatic_entropy_tuning=False,
    )

    act = agent.select_action(np.zeros(3), evaluate=False)
    assert act.shape == (2,)

    # single update step with stub memory
    m = _StubMemory(obs_dim=3, act_dim=2, batch=4)
    q1, q2, pi, aloss, at = agent.update_parameters(m, batch_size=4, updates=1)
    assert isinstance(q1, float) and isinstance(pi, float)
