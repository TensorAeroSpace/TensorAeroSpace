"""Smoke test: A2C.train() writes canonical TensorBoard tags."""

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
    schema.LOSS_ENTROPY,
    schema.A2C.ADVANTAGE_MEAN,
    schema.POLICY_ACTION_STD,
}


@pytest.mark.timeout(120)
def test_a2c_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.a2c.model import A2C, Actor, Critic

    # Wrap Pendulum in a TimeLimit short enough to guarantee episode
    # termination within the smoke run. Default Pendulum-v1 truncates at 200,
    # which would not trigger inside the tiny rollout below.
    env = gym.wrappers.TimeLimit(gym.make("Pendulum-v1").unwrapped, max_episode_steps=16)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    log_dir = tmp_path / "tb"
    agent = A2C(
        env=env,
        actor=actor,
        critic=critic,
        log_dir=str(log_dir),
        device="cpu",
    )
    # Tiny run that produces at least one rollout and several episode ends
    # (each TimeLimit-wrapped Pendulum episode caps at 16 steps).
    agent.train(
        episodes=2,
        steps_on_memory=32,
        episode_length=64,
        log_freq=1,
    )
    agent.writer.flush()
    assert_tags_present(str(log_dir), REQUIRED)
