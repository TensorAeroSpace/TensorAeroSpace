"""Behaviour of tensoraerospace.agent.metrics.writer._WandbSink."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tensoraerospace.agent.metrics import writer as writer_mod


@pytest.fixture
def mock_wandb(monkeypatch):
    """Provide a fresh mock for the `wandb` module attribute on writer module."""
    mock = MagicMock()
    mock.Histogram = MagicMock(side_effect=lambda v: ("HISTOGRAM", v))
    mock.Settings = MagicMock(return_value="SETTINGS_OBJ")
    mock_run = MagicMock()
    mock.init = MagicMock(return_value=mock_run)
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock, mock_run


def test_init_calls_wandb_init_with_kwargs(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    _WandbSink(
        project="proj-1",
        entity="ent-1",
        run_name="run-1",
        tags=["sac", "pendulum"],
        config={"lr": 1e-3},
    )

    mock.login.assert_not_called()
    mock.init.assert_called_once()
    kwargs = mock.init.call_args.kwargs
    assert kwargs["project"] == "proj-1"
    assert kwargs["entity"] == "ent-1"
    assert kwargs["name"] == "run-1"
    assert kwargs["tags"] == ["sac", "pendulum"]
    assert kwargs["config"] == {"lr": 1e-3}
    assert kwargs["reinit"] is True
    assert kwargs["settings"] == "SETTINGS_OBJ"


def test_init_calls_login_when_api_key_missing(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, _ = mock_wandb
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    mock.login.assert_called_once()
    mock.init.assert_called_once()


def test_add_scalar_calls_wandb_log(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)
    mock_run.log.reset_mock()

    sink.add_scalar("loss/actor", 0.123, env_step=10)

    mock_run.log.assert_called_once_with({"loss/actor": 0.123}, step=10)


def test_add_histogram_wraps_in_wandb_histogram(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)
    mock_run.log.reset_mock()

    values = np.zeros(8)
    sink.add_histogram("weights/actor/fc1", values, env_step=5)

    mock.Histogram.assert_called_once_with(values)
    mock_run.log.assert_called_once_with(
        {"weights/actor/fc1": ("HISTOGRAM", values)}, step=5
    )


def test_close_calls_finish_once(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    sink.close()
    sink.close()  # second call must be a no-op

    mock_run.finish.assert_called_once()


def test_flush_is_noop(mock_wandb, monkeypatch):
    from tensoraerospace.agent.metrics.writer import _WandbSink
    mock, _ = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    # Should not raise; nothing to assert on the mock besides no exception.
    sink.flush()


def test_two_sinks_log_to_their_own_runs(monkeypatch):
    """Two coexisting _WandbSink instances must not cross-write to each other."""
    from unittest.mock import MagicMock
    from tensoraerospace.agent.metrics import writer as writer_mod
    from tensoraerospace.agent.metrics.writer import _WandbSink

    mock = MagicMock()
    mock.Histogram = MagicMock(side_effect=lambda v: ("H", v))
    mock.Settings = MagicMock(return_value="S")
    run_a = MagicMock(name="run_a")
    run_b = MagicMock(name="run_b")
    mock.init = MagicMock(side_effect=[run_a, run_b])
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    sink_a = _WandbSink(project="A", entity=None, run_name=None, tags=None, config=None)
    sink_b = _WandbSink(project="B", entity=None, run_name=None, tags=None, config=None)

    sink_a.add_scalar("loss/actor", 1.0, env_step=1)
    sink_b.add_scalar("loss/critic", 2.0, env_step=1)

    run_a.log.assert_called_once_with({"loss/actor": 1.0}, step=1)
    run_b.log.assert_called_once_with({"loss/critic": 2.0}, step=1)
    # No cross-contamination
    assert run_a.log.call_count == 1
    assert run_b.log.call_count == 1
