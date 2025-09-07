import os
import tempfile

import numpy as np

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

    @property
    def unwrapped(self):
        return self


def test_sac_from_pretrained_local_roundtrip():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    agent = SAC(
        env=env,
        batch_size=2,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )
    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        subdirs = [os.path.join(d, name) for name in os.listdir(d)]
        saved = subdirs[0]
        # load back via classmethod
        loaded = SAC.from_pretrained(saved)
        # simple smoke: select_action works
        act = loaded.select_action(np.zeros(3), evaluate=True)
        assert act.shape == (2,)
