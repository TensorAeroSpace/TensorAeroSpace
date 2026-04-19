"""Smoke test: PPO training writes canonical TensorBoard tags."""

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
    schema.PPO.APPROX_KL,
    schema.PPO.CLIP_FRACTION,
    schema.POLICY_ENTROPY,
}


@pytest.mark.timeout(180)
def test_ppo_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.ppo.model import PPO

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"

    # Smallest config that exercises the per-epoch logging block at least once.
    agent = PPO(
        env=env,
        max_episodes=1,
        rollout_len=64,
        num_epochs=1,
        batch_size=16,
        eval_freq=1,
        normalize_obs=False,
        normalize_reward=False,
        log_dir=str(log_dir),
        device="cpu",
        save_best_model=False,
    )
    agent.train(num_episodes=1, max_steps=64)

    agent.writer.flush()
    assert_tags_present(str(log_dir), REQUIRED)
