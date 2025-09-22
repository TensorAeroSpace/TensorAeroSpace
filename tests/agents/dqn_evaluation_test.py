import numpy as np

from tensoraerospace.agent.dqn.model import Model, PERAgent


class _WrappedEnv:
    def __init__(self):
        self.env = self
        self.observation_space = type("Obs", (), {"shape": (4,)})()
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros(4), {}

    def step(self, action):
        self._step += 1
        obs = np.ones(4) * self._step
        reward = 1.0
        done = self._step >= 1
        info = {}
        terminated = False
        return obs, reward, done, info, terminated

    def render(self):
        pass

    def close(self):
        pass

    def capture_frame(self):
        pass


def test_dqn_evaluation_smoke():
    env = _WrappedEnv()
    model = Model(num_actions=2)
    target = Model(num_actions=2)
    agent = PERAgent(
        model=model, target_model=target, env=env, buffer_size=8, batch_size=4
    )
    # mock action_value to avoid TF run
    agent.model.action_value = lambda obs: (0, np.zeros(2, dtype=np.float32))
    rew = agent.evaluation(env, render=False)
    assert isinstance(rew, float)
