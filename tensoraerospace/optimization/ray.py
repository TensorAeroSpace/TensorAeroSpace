"""Ray Tune-based hyperparameter optimization backend.

This module integrates `ray.tune` with TensorAeroSpace's optimization interface.
It provides a thin wrapper to run a search and extract the best configuration.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ray import tune

from .base import HyperParamOptimizationBase


class HyperParamOptimizationRay(HyperParamOptimizationBase):
    """Hyperparameter optimization using Ray Tune."""

    def __init__(self, direction: str, metric: Optional[str] = None) -> None:
        """Create a Ray Tune optimizer wrapper.

        Args:
            direction: Optimization direction. One of
                ``'minimize'``, ``'maximize'``, ``'min'``, ``'max'``.
            metric: Metric name used to choose the best result when Ray Tune
                supports it.

        Raises:
            ValueError: If ``direction`` is not supported.
        """
        super().__init__()
        if direction in ("minimize", "min"):
            self.mode = "min"
        elif direction in ("maximize", "max"):
            self.mode = "max"
        else:
            raise ValueError("direction must be one of: minimize/maximize (or min/max)")

        self.metric = metric
        self.tuner: Any = None
        self.results: Any = None

    def run_optimization(
        self,
        func: Callable,
        param_space: Any,
        tune_config: tune.TuneConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Run a Ray Tune search.

        Args:
            func: Trainable (callable) to execute. See Ray Tune docs.
            param_space: Search space definition passed to Ray Tune.
            tune_config: Optional Ray Tune configuration. Defaults to
                ``tune.TuneConfig(num_samples=5)``.
            **kwargs: Additional keyword arguments forwarded to ``tune.Tuner``.
        """
        if tune_config is None:
            tune_config = tune.TuneConfig(num_samples=5)
        self.tuner = tune.Tuner(
            func, param_space=param_space, tune_config=tune_config, **kwargs
        )
        self.results = self.tuner.fit()

    def get_best_param(self) -> dict:
        """Return the best configuration from the latest Ray Tune run.

        Returns:
            dict: Best configuration dictionary.

        Raises:
            RuntimeError: If optimization has not been run yet or best result
                cannot be determined.
        """
        if self.results is None:
            raise RuntimeError(
                "Optimization has not been run yet. Call run_optimization() first."
            )

        grid = self.results
        best = None

        # Prefer Ray Tune API when available
        if hasattr(grid, "get_best_result"):
            try:
                if self.metric:
                    best = grid.get_best_result(metric=self.metric, mode=self.mode)
                else:
                    best = grid.get_best_result()
            except Exception:
                best = None

        # Fallback: first result in iterable grid
        if best is None:
            try:
                best = next(iter(grid))
            except Exception as e:
                raise RuntimeError(
                    "Unable to determine best result from Ray Tune results."
                ) from e

        cfg = getattr(best, "config", None)
        if isinstance(cfg, dict):
            return cfg

        # Last resort: try params attribute or empty dict
        params = getattr(best, "params", {})
        return dict(params) if isinstance(params, dict) else {}

    def plot_parms(self):
        """Plot optimization history (not implemented yet).

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError()
