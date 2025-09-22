import types

import numpy as np

from tensoraerospace.agent.base import (
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)


class _DummyEnv:
    def __init__(self):
        self.initial_state = np.array([0.0, 1.0], dtype=np.float32)
        self.reference_signal = np.zeros((2, 10), dtype=np.float32)
        self.alpha_states = np.array([1.0, 2.0], dtype=np.float32)

    def get_init_args(self):
        return {
            "initial_state": self.initial_state,
            "reference_signal": self.reference_signal,
            "alpha_states": self.alpha_states,
            "misc": {"a": [1, 2, 3]},
        }


def test_get_class_from_string_builtin():
    cls = get_class_from_string("types.SimpleNamespace")
    assert cls is types.SimpleNamespace


def test_serialize_and_deserialize_env():
    env = _DummyEnv()
    data = serialize_env(env)
    # numpy arrays converted to lists
    assert isinstance(data["initial_state"], list)
    out = deserialize_env_params(data)
    # lists converted back to numpy for known keys
    assert isinstance(out["initial_state"], np.ndarray)
