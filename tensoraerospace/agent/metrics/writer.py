"""MetricWriter — strict-whitelist wrapper around torch.utils.tensorboard.SummaryWriter."""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Union

import wandb  # type: ignore[import-untyped]

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


class _TensorBoardSink:
    """Sink that forwards metrics to a torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: Optional[Union[str, Path]]) -> None:
        log_path = str(log_dir) if log_dir is not None else None
        self._writer = (
            TorchSummaryWriter(log_dir=log_path)
            if log_path is not None
            else TorchSummaryWriter()
        )

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        self._writer.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        self._writer.add_histogram(tag, values, env_step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class _WandbSink:
    """Sink that forwards metrics to Weights & Biases."""

    def __init__(
        self,
        *,
        project: str,
        entity: Optional[str],
        run_name: Optional[str],
        tags: Optional[Sequence[str]],
        config: Optional[Mapping[str, Any]],
    ) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            wandb.login()
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags) if tags else None,
            config=dict(config) if config else None,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        wandb.log({tag: float(value)}, step=int(env_step))

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        wandb.log({tag: wandb.Histogram(values)}, step=int(env_step))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


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
        tb_log_dir: Optional[Union[str, Path]] = None,
        *,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: Optional[str] = None,
    ) -> None:
        self._sinks: list = []
        if tb_log_dir is not None:
            self._sinks.append(_TensorBoardSink(tb_log_dir))
        if wandb_project is not None:
            self._sinks.append(
                _WandbSink(
                    project=wandb_project,
                    entity=wandb_entity,
                    run_name=wandb_run_name,
                    tags=wandb_tags,
                    config=wandb_config,
                )
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
        for sink in self._sinks:
            sink.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        if self._strict and not schema.is_registered_histogram(tag):
            raise ValueError(
                f"Unknown histogram tag {tag!r}. "
                "Histograms must match weights/<group>/<param> or "
                "grads/<group>/<param> with <group> in "
                f"{sorted(schema.HISTOGRAM_SUBGROUPS)}."
            )
        for sink in self._sinks:
            sink.add_histogram(tag, values, env_step)

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
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def create_metric_writer(
    tb_log_dir: Optional[Union[str, Path]] = None,
    *,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_config: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    algo: Optional[str] = None,
) -> MetricWriter:
    """Construct a MetricWriter with auto-detection of the wandb backend.

    Activation rules:
      * tb_log_dir set                          -> TB sink active.
      * wandb_project set                       -> wandb sink active (calls
                                                   wandb.login() if no key).
      * WANDB_API_KEY in env, no wandb_project  -> wandb sink active with
                                                   project = algo (or
                                                   "tensoraerospace").

    In multi-worker (forked) processes, wandb is skipped.
    """
    import multiprocessing

    is_main_process = multiprocessing.current_process().name == "MainProcess"

    wandb_enabled = is_main_process and (
        wandb_project is not None or os.environ.get("WANDB_API_KEY") is not None
    )

    if wandb_enabled:
        resolved_project = wandb_project or algo or "tensoraerospace"
        if wandb_run_name is None:
            ts = _dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            wandb_run_name = f"{algo or 'run'}-{ts}"
        if wandb_tags is None and algo is not None:
            wandb_tags = [algo]
    else:
        resolved_project = None

    return MetricWriter(
        tb_log_dir=tb_log_dir,
        wandb_project=resolved_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_config=wandb_config,
        strict=strict,
        algo=algo,
    )
