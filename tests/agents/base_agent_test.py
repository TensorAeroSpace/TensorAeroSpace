"""Tests for tensoraerospace.agent.base module."""

import pytest

from tensoraerospace.agent.base import (
    BaseRLModel,
    TheEnvironmentDoesNotMatch,
    deserialize_env_params,
    get_class_from_string,
    serialize_env,
)


class _ConcreteRL(BaseRLModel):
    """Concrete subclass to instantiate for testing."""

    pass


def test_baserlmodel_methods_fail_loudly():
    """Base interface methods should not silently no-op."""
    model = _ConcreteRL()
    assert model.train() == {}
    for method_name in (
        "get_env",
        "action_probability",
        "save",
        "load",
        "predict",
        "get_param_env",
    ):
        with pytest.raises(NotImplementedError, match=method_name):
            getattr(model, method_name)()


def test_get_class_from_string():
    """Cover get_class_from_string utility."""
    cls = get_class_from_string("tensoraerospace.agent.base.BaseRLModel")
    assert cls is BaseRLModel


def test_serialize_and_deserialize_env():
    """Cover serialize_env and deserialize_env_params."""
    import numpy as np

    class _FakeEnv:
        def get_init_args(self):
            return {"initial_state": np.array([1.0, 2.0]), "dt": 0.01}

    env = _FakeEnv()
    serialized = serialize_env(env)
    assert isinstance(serialized["initial_state"], list)
    assert serialized["dt"] == 0.01

    deserialized = deserialize_env_params(serialized)
    assert isinstance(deserialized["initial_state"], np.ndarray)


def test_exception_message():
    """Cover TheEnvironmentDoesNotMatch exception."""
    exc = TheEnvironmentDoesNotMatch()
    assert "does not match" in exc.message
