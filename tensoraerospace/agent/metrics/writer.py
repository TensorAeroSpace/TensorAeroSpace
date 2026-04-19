"""MetricWriter — strict-whitelist wrapper around torch.utils.tensorboard.SummaryWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Set, Union

from . import schema
from .contract import MANDATORY_METRICS, check_contract


class _FallbackSummaryWriter:
    """No-op SummaryWriter used when tensorboard is unavailable."""

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


def _get_summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter
    except Exception:
        return _FallbackSummaryWriter


class _LazyTorchSummaryWriter:
    def __call__(self, *args, **kwargs):
        cls = _get_summary_writer_class()
        return cls(*args, **kwargs)


TorchSummaryWriter = _LazyTorchSummaryWriter()


class MetricWriter:
    """SummaryWriter wrapper that enforces the canonical metric schema.

    Parameters
    ----------
    log_dir
        TensorBoard log directory.
    strict
        If True, ``add_scalar``/``add_histogram`` raise ``ValueError`` for tags
        not in ``schema.REGISTRY`` (after stripping multi-worker suffix) or not
        matching the histogram prefix rule.
    required
        Tuple of tags that must be written at least once during the writer's
        lifetime. Checked by ``assert_contract_satisfied``.
    algo
        Optional algorithm label, included in error messages.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        *,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: Optional[str] = None,
    ) -> None:
        log_path = str(log_dir) if log_dir is not None else None
        self._writer = (
            TorchSummaryWriter(log_dir=log_path)
            if log_path is not None
            else TorchSummaryWriter()
        )
        self._strict = strict
        self._required = tuple(required)
        self._algo = algo
        self._written: Set[str] = set()

    # -- core api ----------------------------------------------------------

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        if self._strict and not schema.is_registered_scalar(tag):
            raise ValueError(
                f"Unknown metric tag {tag!r}"
                + (f" (algo={self._algo})" if self._algo else "")
                + ". Register it in tensoraerospace.agent.metrics.schema "
                "or construct MetricWriter(strict=False)."
            )
        self._written.add(schema.strip_worker_suffix(tag))
        self._writer.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        if self._strict and not schema.is_registered_histogram(tag):
            raise ValueError(
                f"Unknown histogram tag {tag!r}. "
                "Histograms must match weights/<group>/<param> or "
                "grads/<group>/<param> with <group> in "
                f"{sorted(schema.HISTOGRAM_SUBGROUPS)}."
            )
        self._writer.add_histogram(tag, values, env_step)

    # -- sugar -------------------------------------------------------------

    def log_episode(
        self,
        *,
        reward: float,
        length: int,
        env_step: int,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        """Write the mandatory rollout/* tier for one finished episode."""
        self.add_scalar(schema.ROLLOUT_EPISODE_REWARD, float(reward), env_step)
        self.add_scalar(schema.ROLLOUT_EPISODE_LENGTH, int(length), env_step)
        self.add_scalar(schema.ROLLOUT_TOTAL_STEPS, int(env_step), env_step)
        if terminated is not None:
            self.add_scalar(
                schema.DIAG_TERMINATED_COUNT, int(bool(terminated)), env_step
            )
        if truncated is not None:
            self.add_scalar(schema.DIAG_TRUNCATED_COUNT, int(bool(truncated)), env_step)

    def assert_contract_satisfied(self) -> None:
        """Raise MetricsContractError if any required metric was never written."""
        check_contract(self._written, self._required)

    # -- lifecycle ---------------------------------------------------------

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def create_metric_writer(
    log_dir: Optional[Union[str, Path]] = None,
    *,
    strict: bool = True,
    algo: Optional[str] = None,
) -> MetricWriter:
    """Factory used by agents — keeps call sites short."""
    return MetricWriter(log_dir=log_dir, strict=strict, algo=algo)
