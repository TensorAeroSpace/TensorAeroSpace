"""Smoke test: DQN.train() writes canonical TensorBoard tags."""

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
    schema.DQN.LOSS_Q,
    schema.DQN.EPSILON,
    schema.VALUE_TD_ERROR_MEAN,
}


@pytest.mark.timeout(120)
def test_dqn_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.dqn.model import DQNAgent, Model

    env = gym.make("CartPole-v1")
    log_dir = tmp_path / "tb"

    model = Model(num_actions=env.action_space.n)
    target_model = Model(num_actions=env.action_space.n)

    agent = DQNAgent(
        model=model,
        target_model=target_model,
        env=env,
        train_nums=200,        # large enough for warmup + a few train_steps
        buffer_size=64,        # small so warmup ends quickly
        batch_size=16,
        learning_rate=1e-3,
        epsilon=0.5,
        epsilon_dacay=0.99,
        min_epsilon=0.1,
        target_update_iter=20,
        log_dir=str(log_dir),
    )

    agent.train()
    agent.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED)
