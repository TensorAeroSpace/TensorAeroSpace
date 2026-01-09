"""Utilities for consistent TensorBoard metric naming across agents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except Exception:  # pragma: no cover - tensorboard optional at runtime

    class TorchSummaryWriter:  # type: ignore
        """Fallback SummaryWriter when tensorboard is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_histogram(self, *args, **kwargs) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass


_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")

# Normalize common group/name variations to a shared schema
_GROUP_ALIASES = {
    "losses": "loss",
}

_METRIC_ALIASES = {
    # Reward-style metrics
    "performance/reward": "performance/episode_reward",
    "performance/best_reward": "performance/best_episode_reward",
    "performance/reward_median": "performance/reward_median",
    "performance/reward_p10": "performance/reward_p10",
    "performance/reward_p90": "performance/reward_p90",
    "evaluation/reward": "evaluation/episode_reward",
    # Training progress
    "train/totalsteps": "train/total_steps",
    "train/replaysize": "train/replay_size",
    # Loss aliases
    "losses/actor": "loss/actor",
    "losses/critic": "loss/critic",
    "losses/entropy": "loss/entropy",
    "losses/advantage": "advantage/mean",
    "loss/qf1": "loss/q1",
    "loss/qf2": "loss/q2",
    "loss/z1": "loss/q1",
    "loss/z2": "loss/q2",
    "loss/log_probs": "loss/log_prob",
    "advantage/raw_std": "advantage/std",
}


def _normalize_token(token: str) -> str:
    """Convert camel-case/space/dash tokens to snake_case lowercase."""
    token = token.replace(" ", "_").replace("-", "_")
    token = _CAMEL_CASE_BOUNDARY.sub("_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_").lower()


def normalize_tag(tag: str) -> str:
    """Normalize a TensorBoard tag to a consistent group/name schema."""
    if not isinstance(tag, str):
        return tag

    parts = [p for p in tag.replace("\\", "/").split("/") if p]
    if not parts:
        return tag

    normalized_parts = []
    for idx, raw in enumerate(parts):
        token = _normalize_token(raw)
        if idx == 0:
            token = _GROUP_ALIASES.get(token, token)
        normalized_parts.append(token)

    normalized_tag = "/".join(normalized_parts)
    return _METRIC_ALIASES.get(normalized_tag, normalized_tag)


class MetricWriter:
    """SummaryWriter wrapper that normalizes metric names before logging."""

    def __init__(self, writer: TorchSummaryWriter) -> None:
        self._writer = writer

    def add_scalar(self, tag, scalar_value, global_step=None, *args, **kwargs):
        normalized_tag = normalize_tag(tag)
        return self._writer.add_scalar(
            normalized_tag, scalar_value, global_step, *args, **kwargs
        )

    def add_histogram(self, tag, values, global_step=None, *args, **kwargs):
        normalized_tag = normalize_tag(tag)
        return self._writer.add_histogram(
            normalized_tag, values, global_step, *args, **kwargs
        )

    def flush(self) -> None:
        return self._writer.flush()

    def close(self) -> None:
        return self._writer.close()

    def __getattr__(self, name):
        # Delegate other attributes to the underlying writer (e.g., add_scalars).
        return getattr(self._writer, name)


def create_metric_writer(log_dir: Optional[Union[str, Path]] = None) -> MetricWriter:
    """Create a MetricWriter with optional log directory."""
    log_path = str(log_dir) if log_dir is not None else None
    base_writer = (
        TorchSummaryWriter(log_dir=log_path)
        if log_path is not None
        else TorchSummaryWriter()
    )
    return MetricWriter(base_writer)


def ensure_metric_writer(
    writer: Optional[TorchSummaryWriter],
) -> Optional[MetricWriter]:
    """Wrap an existing writer if provided, otherwise return None."""
    if writer is None:
        return None
    if isinstance(writer, MetricWriter):
        return writer
    return MetricWriter(writer)
