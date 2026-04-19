"""Multi-worker guard: workers must not initialize wandb."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tensoraerospace.agent.metrics import create_metric_writer
from tensoraerospace.agent.metrics import writer as writer_mod


@pytest.fixture
def mock_wandb(monkeypatch):
    mock = MagicMock()
    mock.init = MagicMock(return_value=MagicMock())
    mock.Settings = MagicMock(return_value="S")
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock


def test_in_main_process_wandb_init_runs(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(algo="a3c")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_called_once()


def test_in_worker_process_wandb_skipped(monkeypatch, mock_wandb):
    """When not running as MainProcess, wandb sink is omitted."""
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    fake_proc = MagicMock()
    fake_proc.name = "Worker-1"
    with patch("multiprocessing.current_process", return_value=fake_proc):
        w = create_metric_writer(algo="a3c")

    assert len(w._sinks) == 0
    mock_wandb.init.assert_not_called()


def test_in_worker_process_explicit_wandb_project_still_skipped(
    monkeypatch, mock_wandb
):
    """Even an explicit wandb_project doesn't override the worker guard."""
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    fake_proc = MagicMock()
    fake_proc.name = "Worker-2"
    with patch("multiprocessing.current_process", return_value=fake_proc):
        w = create_metric_writer(wandb_project="exp", algo="a3c")
    assert len(w._sinks) == 0
    mock_wandb.init.assert_not_called()
