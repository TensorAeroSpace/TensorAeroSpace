"""Auto-activation behaviour of create_metric_writer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensoraerospace.agent.metrics import create_metric_writer
from tensoraerospace.agent.metrics import writer as writer_mod


@pytest.fixture
def mock_wandb(monkeypatch):
    mock = MagicMock()
    mock.init = MagicMock(return_value=MagicMock())
    mock.Settings = MagicMock(return_value="S")
    mock.Histogram = MagicMock(side_effect=lambda v: ("H", v))
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock


def test_tb_only_when_only_tb_log_dir(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = create_metric_writer(tb_log_dir=str(tmp_path / "tb"), algo="sac")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_not_called()


def test_wandb_only_when_only_api_key(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(algo="sac")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_called_once()
    assert mock_wandb.init.call_args.kwargs["project"] == "sac"


def test_both_when_tb_log_dir_and_api_key(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(tb_log_dir=str(tmp_path / "tb"), algo="ppo")
    assert len(w._sinks) == 2
    mock_wandb.init.assert_called_once()


def test_zero_sinks_when_nothing(monkeypatch, mock_wandb):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = create_metric_writer(algo="sac")
    assert len(w._sinks) == 0
    with pytest.raises(ValueError):
        w.add_scalar("not/in/schema", 1.0, env_step=1)


def test_explicit_wandb_project_calls_login_when_api_key_missing(
    monkeypatch, mock_wandb
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    create_metric_writer(wandb_project="my-exp", algo="sac")
    mock_wandb.login.assert_called_once()
    mock_wandb.init.assert_called_once()
    assert mock_wandb.init.call_args.kwargs["project"] == "my-exp"


def test_default_project_uses_algo_when_only_api_key(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="ddpg")
    assert mock_wandb.init.call_args.kwargs["project"] == "ddpg"


def test_default_project_falls_back_when_no_algo(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer()
    assert mock_wandb.init.call_args.kwargs["project"] == "tensoraerospace"


def test_default_run_name_includes_algo_and_timestamp(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="sac")
    name = mock_wandb.init.call_args.kwargs["name"]
    assert name.startswith("sac-")
    assert len(name) > len("sac-")


def test_default_tags_contain_algo(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="dqn")
    tags = mock_wandb.init.call_args.kwargs["tags"]
    assert "dqn" in tags


def test_explicit_kwargs_override_defaults(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(
        algo="sac",
        wandb_project="p",
        wandb_entity="e",
        wandb_run_name="r",
        wandb_tags=["t1", "t2"],
        wandb_config={"k": "v"},
    )
    kw = mock_wandb.init.call_args.kwargs
    assert kw["project"] == "p"
    assert kw["entity"] == "e"
    assert kw["name"] == "r"
    assert kw["tags"] == ["t1", "t2"]
    assert kw["config"] == {"k": "v"}
