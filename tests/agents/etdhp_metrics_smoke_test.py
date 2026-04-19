"""Smoke test: ETDHPAgent.train() writes canonical TensorBoard tags."""

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
def test_etdhp_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym

    from tensoraerospace.agent.et_dhp.model import ETDHPAgent, ETDHPConfig

    env_inner = gym.make("Pendulum-v1")
    env = gym.wrappers.TimeLimit(env_inner, max_episode_steps=16)
    log_dir = tmp_path / "tb"

    n_state = env.observation_space.shape[0]
    n_control = env.action_space.shape[0]

    cfg = ETDHPConfig(
        Q=[1.0] * n_state,
        R=[1.0] * n_control,
        # Smallest valid rho keeps the Lipschitz threshold tiny so the
        # event trigger fires frequently in a 16-step budget; combined
        # with min_floor=0 and Pendulum-v1's volatile observations, every
        # step after the initial one easily clears the bound.
        rho=0.001,
        trigger_floor=0.0,
        num_epochs_per_trigger=1,
        device="cpu",
        seed=0,
    )
    agent = ETDHPAgent(
        n_state=n_state,
        n_control=n_control,
        config=cfg,
        log_dir=str(log_dir),
    )
    agent.train(env=env, num_episodes=1, max_steps=16)
    agent.writer.flush()

    assert_tags_present(str(log_dir), REQUIRED)
