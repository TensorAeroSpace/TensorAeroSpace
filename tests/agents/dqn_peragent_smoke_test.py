import numpy as np

from tensoraerospace.agent.dqn.model import PERAgent


class _DummySpace:
    def __init__(self, shape, n=None):
        self.shape = shape
        self.n = n

    def sample(self):
        if self.n is not None:
            return 0
        return np.zeros(self.shape, dtype=np.float32)


class _DummyEnv:
    def __init__(self):
        self.observation_space = _DummySpace((4,))
        self.action_space = _DummySpace(shape=None, n=2)

    def reset(self):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, _action):
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


class _StubModel:
    def compile(self, optimizer=None, loss=None):
        return None

    def predict(self, inputs):
        # return q-values
        batch = inputs.shape[0]
        return np.zeros((batch, 2), dtype=np.float32)

    def train_on_batch(self, obs, target_q):
        return 0.0

    def action_value(self, obs):
        q = self.predict(obs)
        best_action = np.argmax(q, axis=-1)
        return best_action, q

    def get_weights(self):
        return []

    def set_weights(self, _w):
        return None


def test_peragent_core_methods_smoke():
    env = _DummyEnv()
    model = _StubModel()
    target_model = _StubModel()
    agent = PERAgent(
        model=model,
        target_model=target_model,
        env=env,
        buffer_size=16,
        batch_size=4,
        train_nums=10,
    )

    # store_transition and sampling
    agent.store_transition(1.0, np.zeros(4), 0, 0.0, np.zeros(4), False)
    agent.store_transition(1.0, np.zeros(4), 0, 0.0, np.zeros(4), False)
    idxes, isw = agent.sum_tree_sample(2)
    assert len(idxes) == 2
    assert isw.shape == (2, 1)

    # target update and helpers
    agent.update_target_model()
    val = agent.get_target_value(np.zeros((2, 4)))
    assert val.shape == (2, 2)
    agent.e_decay()
