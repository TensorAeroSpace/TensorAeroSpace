"""Smoke test: A2C-NARX (A2CLearner + Runner) writes canonical TB tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present


REQUIRED_LEARNER = {
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.LOSS_ACTOR,
    schema.LOSS_CRITIC,
    schema.LOSS_ENTROPY,
    schema.A2C.ADVANTAGE_MEAN,
    schema.A2C.ENTROPY_BETA,
}

REQUIRED_RUNNER = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
}


@pytest.mark.timeout(120)
def test_a2c_narx_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.a2c.narx import A2CLearner, Actor, Critic, Runner

    env_inner = gym.make("Pendulum-v1")
    env = gym.wrappers.TimeLimit(env_inner, max_episode_steps=16)

    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)

    log_dir = tmp_path / "tb"
    learner = A2CLearner(
        actor=actor,
        critic=critic,
        log_dir=str(log_dir),
        device="cpu",
    )

    # Use the same writer for both learner and runner so all tags share one log_dir
    # without the contract failing on a runner-only writer.
    runner = Runner(env=env, actor=actor, writer=learner.writer)

    # Collect transitions, then run one learn() pass
    memory = runner.run(max_steps=64)
    learner.learn(memory, steps=runner.steps)

    learner.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED_LEARNER | REQUIRED_RUNNER)
