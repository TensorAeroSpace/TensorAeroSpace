"""Mandatory metric contract for RL training agents."""

from __future__ import annotations

from typing import Iterable, Tuple

from . import schema

MANDATORY_METRICS: Tuple[str, ...] = (
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
)


class MetricsContractError(AssertionError):
    """Raised when an agent's training run did not log the mandatory minimum."""


def check_contract(written_tags: Iterable[str], required: Iterable[str]) -> None:
    """Raise MetricsContractError if any required tag was never written."""
    written = set(written_tags)
    missing = [t for t in required if t not in written]
    if missing:
        raise MetricsContractError(
            f"Mandatory metrics never written: {missing}. "
            "Make sure the agent's train() loop calls writer.log_episode(...) "
            "and writes train/updates and train/lr."
        )
