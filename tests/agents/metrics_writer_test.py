"""Behaviour of tensoraerospace.agent.metrics.writer.MetricWriter."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoraerospace.agent.metrics import (
    MANDATORY_METRICS,
    MetricsContractError,
    MetricWriter,
    schema,
)


@pytest.fixture
def writer(tmp_path: Path) -> MetricWriter:
    return MetricWriter(tb_log_dir=str(tmp_path / "tb"), algo="test")


def test_add_scalar_accepts_registered_tag(writer: MetricWriter):
    writer.add_scalar(schema.LOSS_ACTOR, 0.123, env_step=10)


def test_add_scalar_rejects_unregistered_tag(writer: MetricWriter):
    with pytest.raises(ValueError, match="Unknown metric tag 'Performance/Reward'"):
        writer.add_scalar("Performance/Reward", 1.0, env_step=1)


def test_add_scalar_accepts_worker_suffix(writer: MetricWriter):
    writer.add_scalar(f"{schema.LOSS_ACTOR}/worker_2", 0.5, env_step=5)


def test_add_scalar_rejects_bad_worker_suffix(writer: MetricWriter):
    with pytest.raises(ValueError):
        writer.add_scalar(f"{schema.LOSS_ACTOR}/wrkr_2", 0.5, env_step=5)


def test_add_scalar_strict_false_allows_anything(tmp_path: Path):
    w = MetricWriter(tb_log_dir=str(tmp_path / "tb"), strict=False)
    w.add_scalar("anything/at/all", 1.0, env_step=1)


def test_add_scalar_requires_env_step(writer: MetricWriter):
    with pytest.raises(TypeError):
        writer.add_scalar(schema.LOSS_ACTOR, 0.5)  # type: ignore[call-arg]


def test_add_histogram_accepts_valid_prefix(writer: MetricWriter):
    import numpy as np

    writer.add_histogram("weights/actor/fc1.weight", np.zeros(8), env_step=1)


def test_add_histogram_rejects_invalid_prefix(writer: MetricWriter):
    import numpy as np

    with pytest.raises(ValueError, match="Unknown histogram tag"):
        writer.add_histogram("parameters/actor", np.zeros(8), env_step=1)


def test_log_episode_writes_mandatory_rollout_tier(writer: MetricWriter):
    writer.log_episode(
        reward=12.5,
        length=200,
        env_step=200,
        terminated=False,
        truncated=True,
    )
    written = writer._written  # internal but stable for tests
    assert schema.ROLLOUT_EPISODE_REWARD in written
    assert schema.ROLLOUT_EPISODE_LENGTH in written
    assert schema.ROLLOUT_TOTAL_STEPS in written
    assert schema.DIAG_TRUNCATED_COUNT in written
    assert schema.DIAG_TERMINATED_COUNT in written


def test_assert_contract_raises_when_missing(writer: MetricWriter):
    # Only log rollout tier, miss train/updates and train/lr
    writer.log_episode(reward=1.0, length=10, env_step=10)
    with pytest.raises(MetricsContractError, match="train/updates"):
        writer.assert_contract_satisfied()


def test_assert_contract_passes_when_complete(writer: MetricWriter):
    writer.log_episode(reward=1.0, length=10, env_step=10)
    writer.add_scalar(schema.TRAIN_UPDATES, 5, env_step=10)
    writer.add_scalar(schema.TRAIN_LR, 1e-3, env_step=10)
    writer.assert_contract_satisfied()


def test_close_does_not_raise(writer: MetricWriter):
    writer.close()


def test_metricwriter_with_wandb_project_creates_two_sinks(
    tmp_path: Path, monkeypatch
):
    """When both tb_log_dir and wandb_project are passed, both sinks are active."""
    from unittest.mock import MagicMock
    from tensoraerospace.agent.metrics import writer as writer_mod

    mock_wandb = MagicMock()
    mock_wandb.init = MagicMock(return_value=MagicMock())
    mock_wandb.Settings = MagicMock(return_value="S")
    monkeypatch.setattr(writer_mod, "wandb", mock_wandb, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    w = MetricWriter(
        tb_log_dir=str(tmp_path / "tb"),
        wandb_project="exp",
        algo="test",
    )
    assert len(w._sinks) == 2  # noqa: SLF001 — internal but stable for tests


def test_metricwriter_no_kwargs_creates_zero_sinks(monkeypatch):
    """No tb_log_dir and no wandb activation => zero sinks (writer no-ops)."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = MetricWriter(algo="test")
    assert len(w._sinks) == 0
    # whitelist still enforced
    w.add_scalar(schema.LOSS_ACTOR, 0.5, env_step=1)
    with pytest.raises(ValueError):
        w.add_scalar("not/in/schema", 0.5, env_step=1)
