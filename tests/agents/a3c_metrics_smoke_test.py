"""Smoke test: A3C training writes canonical TensorBoard tags with /worker_<N> suffix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present


# A3C Worker writes per-worker scalars with /worker_<id> suffix and a
# small set of shared train/* scalars without a suffix. The smoke test
# uses a single worker (worker_id == 0) running in the main process so
# event files are flushed in the same process that owns the writer.
REQUIRED = {
    f"{schema.ROLLOUT_EPISODE_REWARD}/worker_0",
    f"{schema.ROLLOUT_EPISODE_LENGTH}/worker_0",
    f"{schema.ROLLOUT_TOTAL_STEPS}/worker_0",
    f"{schema.LOSS_ACTOR}/worker_0",
    f"{schema.LOSS_VALUE}/worker_0",
    f"{schema.LOSS_ENTROPY}/worker_0",
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
}


class _SmokeEnv:
    """Tiny continuous-action env that always terminates within max_ep_step."""

    def __init__(self) -> None:
        self.observation_space = type("S", (), {"shape": (4,)})()
        self.action_space = type(
            "A",
            (),
            {
                "shape": (2,),
                "low": np.array([-1.0, -1.0], dtype=np.float32),
                "high": np.array([1.0, 1.0], dtype=np.float32),
            },
        )()
        self.step_count = 0

    def reset(self, seed=None):
        self.step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count >= 4
        truncated = False
        return (
            np.random.randn(4).astype(np.float32),
            float(-np.sum(np.asarray(action) ** 2) * 0.01),
            bool(terminated),
            bool(truncated),
            {},
        )

    def close(self) -> None:
        pass


def _smoke_env_fn(worker_id):
    return _SmokeEnv()


@pytest.mark.timeout(180)
def test_a3c_train_writes_canonical_tags_with_worker_suffix(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("tensorboard")

    from tensoraerospace.agent.a3c.pytorch import Agent

    log_dir = tmp_path / "tb"

    # run_in_main=True keeps the writer in the same process so event
    # files are flushed and visible to the EventAccumulator below.
    agent = Agent(
        env_function=_smoke_env_fn,
        gamma=0.99,
        n_workers=1,
        lr=1e-3,
        max_episodes=2,
        max_ep_step=4,
        update_global_iter=2,
        render=False,
        run_in_main=True,
        log_dir=str(log_dir),
    )
    agent.train()
    if agent.writer is not None:
        agent.writer.flush()
        agent.writer.close()

    assert_tags_present(str(log_dir), REQUIRED)
