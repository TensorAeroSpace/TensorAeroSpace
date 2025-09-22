import numpy as np

from tensoraerospace.agent.dqn.model import Model, PERAgent


class _DummyEnv:
    def __init__(self):
        self.observation_space = type("Obs", (), {"shape": (4,)})()
        self._step = 0
        self.action_space = self

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

    def sample(self):
        return 0


def test_dqn_train_step_with_mocks():
    env = _DummyEnv()
    model = Model(num_actions=2)
    target = Model(num_actions=2)
    agent = PERAgent(
        model=model, target_model=target, env=env, buffer_size=8, batch_size=4
    )

    # fill buffer
    obs, _ = env.reset()
    for i in range(8):
        action = 0
        next_obs, reward, done, info, terminated = env.step(action)
        agent.store_transition(
            1.0, obs, action, reward, next_obs.reshape([1, -1]), done
        )
        obs = next_obs

    # mock network methods to avoid TF compute
    agent.model.predict = lambda x: np.zeros((x.shape[0], 2), dtype=np.float32)
    agent.target_model.predict = lambda x: np.zeros((x.shape[0], 2), dtype=np.float32)
    agent.model.action_value = lambda obs: (
        np.zeros((obs.shape[0],), dtype=int),
        np.zeros(2, dtype=np.float32),
    )
    agent.model.train_on_batch = lambda a, b: 0.0

    losses = agent.train_step()
    assert isinstance(losses, float) or losses == 0.0
