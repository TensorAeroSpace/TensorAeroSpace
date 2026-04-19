"""Smoke test: ADHDP.train() writes canonical TensorBoard tags."""

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
    schema.ADHDP.DO_CRITIC,
    schema.ADHDP.DO_ACTOR,
    schema.POLICY_ACTION_ABS_MEAN,
}


@pytest.mark.timeout(120)
def test_adhdp_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.adhdp.model import ADHDP

    env_inner = gym.make("Pendulum-v1")
    env = gym.wrappers.TimeLimit(env_inner, max_episode_steps=16)
    log_dir = tmp_path / "tb"

    agent = ADHDP(
        env=env,
        device="cpu",
        exploration_std=0.0,
        log_dir=str(log_dir),
        log_every_updates=2,
    )
    agent.train(num_episodes=1, max_steps=16, show_progress=False)
    agent.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED)
