import numpy as np


def _env_fn(_idx):
    class _DummySpace:
        def __init__(self, shape, high_val=1.0):
            self.shape = shape
            self.high = np.array([high_val] * shape[0])

    class _DummyEnv:
        def __init__(self):
            self.observation_space = _DummySpace((3,))
            self.action_space = _DummySpace((2,))

        def reset(self):
            return np.zeros(3, dtype=np.float64)

        def close(self):
            pass

    return _DummyEnv()


def test_a3c_agent_train_smoke(monkeypatch):
    import types

    import tensoraerospace.agent.a3c.model as a3c

    class _NoOpWorker:
        def __init__(self, env, gamma, global_actor, global_critic):
            self.env = env

        def start(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(a3c, "Worker", _NoOpWorker)

    class _NoOpModel:
        def __init__(self, *a, **k):
            self.model = self

        def get_weights(self):
            return []

        def set_weights(self, _w):
            return None

    monkeypatch.setattr(a3c, "Actor", _NoOpModel)
    monkeypatch.setattr(a3c, "Critic", _NoOpModel)

    # Stub out TF summary writer to avoid filesystem writes
    class _DummyWriter:
        def as_default(self):
            class _Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Ctx()

    def _create_writer(_path):
        return _DummyWriter()

    monkeypatch.setattr(
        a3c.tf.summary, "create_file_writer", _create_writer, raising=True
    )

    ag = a3c.Agent(env_function=_env_fn, gamma=0.99)
    # Ensure creating workers doesn't crash
    ag.train()
