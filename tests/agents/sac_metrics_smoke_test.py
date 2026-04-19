"""Smoke test: SAC.train() writes canonical TensorBoard tags."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    schema.SAC.LOSS_Q2,
    schema.LOSS_POLICY,
    schema.SAC.ALPHA_VALUE,
}


@pytest.mark.timeout(120)
def test_sac_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.sac.sac import SAC

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"

    agent = SAC(
        env=env,
        batch_size=16,
        memory_capacity=2_000,
        log_dir=str(log_dir),
        device="cpu",
    )
    # Tiny run: needs enough steps for memory > batch_size + at least one update
    agent.train(num_episodes=1, max_steps=64, verbose=False)

    agent.writer.flush()
    assert_tags_present(str(log_dir), REQUIRED)
