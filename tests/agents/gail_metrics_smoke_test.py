"""Smoke test: GAIL writes canonical TensorBoard tags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    schema.POLICY_ENTROPY,
    schema.GAIL.LOSS_DISCRIMINATOR,
    schema.GAIL.EXPERT_ACCURACY,
    schema.GAIL.POLICY_ACCURACY,
}


@pytest.mark.timeout(180)
def test_gail_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym
    from tensoraerospace.agent.gail.model import GAIL

    env_inner = gym.make("Pendulum-v1")
    env = gym.wrappers.TimeLimit(env_inner, max_episode_steps=8)
    log_dir = tmp_path / "tb"

    # Generate fake "expert" data: random states+actions in valid range.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    expert_data = np.random.randn(64, state_dim + action_dim).astype(np.float32)

    agent = GAIL(
        env=env,
        learning_rate=1e-3,
        max_steps=8,
        mini_batch_size=4,
        epochs=1,
        data=expert_data,
        log_dir=str(log_dir),
    )
    # Run enough frames to trigger:
    #   - at least one episode end (every 8 steps)
    #   - at least one discriminator update (every rollout = 8 frames)
    #   - at least one PPO update (every 3rd rollout, so 24 frames)
    agent.learn(max_frames=32, max_reward=float("inf"))
    agent.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED)
