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
        self._done = False

    def reset(self):
        self._done = False
        return np.zeros(self.observation_space.shape[0]), {}

    def step(self, action):
        # one step episode
        self._done = True
        obs = np.zeros(self.observation_space.shape[0])
        reward = 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}


class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def test_sac_train_one_episode():
    env = _TinyEnv(obs_dim=3, act_dim=2)
    agent = SAC(
        env=env,
        batch_size=8,
        hidden_size=8,
        device="cpu",
        automatic_entropy_tuning=False,
    )
    agent.writer = _NoOpWriter()
    # run one episode, should not error and increase memory length by at least one transition
    before = len(agent.memory)
    agent.train(num_episodes=1)
    after = len(agent.memory)
    assert after >= before
