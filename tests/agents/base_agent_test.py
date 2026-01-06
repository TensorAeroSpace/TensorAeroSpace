"""Tests for tensoraerospace.agent.base module."""

from unittest.mock import MagicMock, patch

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


def test_baserlmodel_methods_return_none():
    """Cover the placeholder methods that just 'pass'."""
    model = _ConcreteRL()
    assert model.get_env() is None
    model.train()  # no return, just ensure it runs
    assert model.action_probability() is None
    model.save()
    model.load()
    assert model.predict() is None
    assert model.get_param_env() is None


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
