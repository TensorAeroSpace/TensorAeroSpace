"""Dual-sink end-to-end: SAC.train() writes to both TB and (mocked) wandb."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensoraerospace.agent.metrics import writer as writer_mod
from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present


REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.TRAIN_REPLAY_SIZE,
    schema.SAC.LOSS_Q1,
    schema.LOSS_POLICY,
}


@pytest.mark.timeout(120)
def test_sac_writes_to_both_tb_and_wandb(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym
    from tensoraerospace.agent.sac.sac import SAC

    # Mock wandb so the test doesn't hit the network
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init = MagicMock(return_value=mock_run)
    mock_wandb.Settings = MagicMock(return_value="S")
    mock_wandb.Histogram = MagicMock(side_effect=lambda v: ("H", v))
    monkeypatch.setattr(writer_mod, "wandb", mock_wandb, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"

    agent = SAC(
        env=env,
        batch_size=16,
        memory_capacity=2_000,
        log_dir=str(log_dir),
        wandb_project="dual-sink-test",
        device="cpu",
    )
    agent.train(num_episodes=1, max_steps=64, verbose=False)
    agent.writer.flush()

    # TB side: tags appear in event files
    assert_tags_present(str(log_dir), REQUIRED)

    # Wandb side: the same tags appear in mocked run.log calls
    logged_keys = set()
    for call in mock_run.log.call_args_list:
        payload = call.args[0] if call.args else call.kwargs.get("data", {})
        logged_keys.update(payload.keys())
    missing = REQUIRED - logged_keys
    assert not missing, (
        f"Tags missing from run.log calls: {sorted(missing)}; "
        f"got {sorted(logged_keys)}"
    )
