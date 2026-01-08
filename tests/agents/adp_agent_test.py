import numpy as np

from tensoraerospace.agent import ADP


class _TinySpace:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.low = -np.ones(self.shape, dtype=np.float32)
        self.high = np.ones(self.shape, dtype=np.float32)


class _TinyEnv:
    def __init__(self, obs_dim: int = 4, act_dim: int = 1, max_steps: int = 4):
        self.observation_space = _TinySpace((obs_dim,))
        self.action_space = _TinySpace((act_dim,))
        self._step_count = 0
        self.max_steps = int(max_steps)

    def reset(self):
        self._step_count = 0
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # Make sure action is used (shape should match act_dim)
        _ = np.asarray(action, dtype=np.float32).reshape(-1)
        self._step_count += 1
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        reward = 0.1
        terminated = self._step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    @property
    def unwrapped(self):
        return self


def test_adp_select_action_shape_and_bounds():
    env = _TinyEnv(obs_dim=3, act_dim=2, max_steps=3)
    agent = ADP(env=env, device="cpu", exploration_std=0.0)
    obs, _ = env.reset()
    act = agent.select_action(obs, evaluate=True)
    assert act.shape == env.action_space.shape
    assert np.all(act >= env.action_space.low)
    assert np.all(act <= env.action_space.high)


def test_adp_train_smoke_online():
    env = _TinyEnv(obs_dim=4, act_dim=1, max_steps=3)
    agent = ADP(env=env, device="cpu", use_replay=False, log_every_updates=10)
    agent.train(num_episodes=2, max_steps=3)
    # At least some updates happened
    assert getattr(agent, "_updates", 0) > 0


