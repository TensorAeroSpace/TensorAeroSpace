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


def test_dqn_e_decay_and_sum_tree_sample():
    env = _DummyEnv()
    model = Model(num_actions=2)
    target = Model(num_actions=2)
    agent = PERAgent(
        model=model, target_model=target, env=env, buffer_size=8, batch_size=4
    )

    e0 = agent.epsilon
    agent.e_decay()
    assert agent.epsilon < e0

    # fill buffer with minimal transitions
    obs, _ = env.reset()
    for i in range(8):
        input_obs = obs.reshape([1, -1])
        best_action, q_values = model.action_value(input_obs)
        action = agent.get_action(int(best_action))
        next_obs, reward, done, info, terminated = env.step(action)
        p = 1.0
        agent.store_transition(p, obs, action, reward, next_obs.reshape([1, -1]), done)
        obs = next_obs
    idxes, is_w = agent.sum_tree_sample(4)
    assert len(idxes) == 4 and is_w.shape == (4, 1)
