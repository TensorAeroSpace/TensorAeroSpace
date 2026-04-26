"""Smoke test: DDPG.train() writes canonical TensorBoard tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoraerospace.agent.metrics import (
    create_metric_writer,
    schema,
)
from tests.agents.metrics_contract_smoke_test import assert_tags_present

REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.TRAIN_REPLAY_SIZE,
    schema.LOSS_POLICY,
    schema.LOSS_VALUE,
    schema.POLICY_ACTION_ABS_MEAN,
}


@pytest.mark.timeout(120)
def test_ddpg_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.ddpg.model import DDPG

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"

    agent = DDPG(
        env=env,
        value_lr=1e-3,
        policy_lr=1e-4,
        replay_buffer_size=2_000,
        normalize_observations=False,
        device="cpu",
    )
    agent.writer = create_metric_writer(str(log_dir), algo="ddpg")

    # short, fast run with low warmup so updates actually fire
    agent.train(
        num_episodes=2,
        max_steps=64,
        max_frames=128,
        batch_size=32,
        warmup_frames=32,
    )

    agent.writer.flush()
    assert_tags_present(str(log_dir), REQUIRED)
