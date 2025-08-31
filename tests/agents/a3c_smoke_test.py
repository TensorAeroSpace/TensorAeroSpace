import numpy as np


class _DummySpace:
    def __init__(self, shape, high_val=1.0):
        self.shape = shape
        self.high = np.array([high_val] * shape[0])


class _DummyEnv:
    def __init__(self):
        self.observation_space = _DummySpace((3,))
        self.action_space = _DummySpace((2,), high_val=1.0)

    def reset(self):
        return np.zeros(3, dtype=np.float64)

    def step(self, _action):
        return np.zeros(3, dtype=np.float64), 0.0, True, {}

    def close(self):
        pass


def _env_fn(_idx):
    return _DummyEnv()


def test_a3c_worker_sync_smoke(monkeypatch):
    # Avoid heavy TF ops by mocking parts of Actor/Critic to no-op
    import tensoraerospace.agent.a3c.model as a3c

    class _NoOpModel:
        def __init__(self, *a, **k):
            self.model = self

        def get_weights(self):
            return []

        def set_weights(self, _w):
            return None

        def predict(self, state):
            mu = np.zeros((1, 2), dtype=np.float64)
            std = np.ones((1, 2), dtype=np.float64) * 0.1
            return mu, std

    monkeypatch.setattr(a3c, "Actor", _NoOpModel)
    monkeypatch.setattr(a3c, "Critic", _NoOpModel)

    # Build worker and perform sync
    worker = a3c.Worker(
        _DummyEnv(), gamma=0.99, global_actor=_NoOpModel(), global_critic=_NoOpModel()
    )
    worker.sync_with_global()
    # Get one action
    action = worker.get_action(np.zeros(3))
    assert action.shape == (2,)
