import numpy as np
import tensorflow as tf

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
        return 1


def test_dqn_epsilon_greedy_and_target_update():
    env = _DummyEnv()
    model = Model(num_actions=2)
    target = Model(num_actions=2)

    agent = PERAgent(
        model=model, target_model=target, env=env, buffer_size=8, batch_size=4
    )

    # epsilon greedy
    agent.epsilon = 0.0
    assert agent.get_action(1) == 1
    agent.epsilon = 1.0
    a = agent.get_action(0)
    assert a in [0, 1]

    # build models by a forward pass
    _ = model.predict(np.zeros((1, 4), dtype=np.float32))
    _ = target.predict(np.zeros((1, 4), dtype=np.float32))
    # update target weights and verify equality with model
    agent.update_target_model()
    for w_m, w_t in zip(model.get_weights(), target.get_weights()):
        assert np.allclose(w_m, w_t)


def test_dqn_get_target_value_shape():
    env = _DummyEnv()
    model = Model(num_actions=2)
    target = Model(num_actions=2)
    agent = PERAgent(
        model=model, target_model=target, env=env, buffer_size=8, batch_size=4
    )

    obs = np.zeros((4, 4))
    q = agent.get_target_value(obs)
    assert q.shape == (4, 2)
