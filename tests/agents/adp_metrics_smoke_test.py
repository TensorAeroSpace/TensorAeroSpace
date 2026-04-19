"""Smoke test: ADP.train() writes canonical TensorBoard tags."""

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
    schema.LOSS_ACTOR,
    schema.LOSS_CRITIC,
}


@pytest.mark.timeout(120)
def test_adp_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.adp.adp import ADP

    env_inner = gym.make("Pendulum-v1")
    env = gym.wrappers.TimeLimit(env_inner, max_episode_steps=8)
    log_dir = tmp_path / "tb"

    agent = ADP(
        env=env,
        device="cpu",
        design="adhdp",
        exploration_std=0.0,
        log_dir=str(log_dir),
        log_every_updates=1,
    )
    agent.train(num_episodes=1, max_steps=8, verbose=False)
    agent.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED)
