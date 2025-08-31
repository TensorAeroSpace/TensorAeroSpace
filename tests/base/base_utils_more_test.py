import types

import numpy as np

from tensoraerospace.agent.base import (
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)
from tensoraerospace.agent.base import (
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)


class _DummyEnv:
    def __init__(self):
        self.a = np.array([1, 2, 3])
        self.b = {"x": np.array([[1.0, 2.0]]), "y": [np.array([4, 5])]}

    def get_init_args(self):
        return {"reference_signal": self.a, "nested": self.b, "alpha_states": [1, 2]}


def test_get_class_from_string_builtin():
    cls = get_class_from_string("builtins.str")
    assert cls is str


def test_serialize_deserialize_env():
    env = _DummyEnv()
    data = serialize_env(env)
    # numpy arrays converted to lists
    assert isinstance(data["reference_signal"], list)
    assert isinstance(data["nested"]["x"], list)

    restored = deserialize_env_params(data)
    assert isinstance(restored["reference_signal"], np.ndarray)
    assert isinstance(restored["alpha_states"], np.ndarray)
