from __future__ import annotations

from typing import Any, Callable, Optional

from ray import tune

from .base import HyperParamOptimizationBase


class HyperParamOptimizationRay(HyperParamOptimizationBase):
    """
    Поиск гиперпараметров модели
    """

    def __init__(self, direction: str, metric: Optional[str] = None) -> None:
        """Инициализация поиска гиперпараметров

        Args:
            direction (str): Направление поиска. Ex. minimize|maximize (или min|max)
            metric (str, optional): Метрика для выбора лучшего результата (Ray Tune).
        """
        super().__init__()
        if direction in ("minimize", "min"):
            self.mode = "min"
        elif direction in ("maximize", "max"):
            self.mode = "max"
        else:
            raise ValueError("Выберите один из вариантов minimize/maximize (или min/max)")

        self.metric = metric
        self.tuner: Any = None
        self.results: Any = None

    def run_optimization(
        self,
        func: Callable,
        param_space,
        tune_config=None,
        **kwargs,
    ):
        """Запуск поиска гиперпараметров

        Args:
            func (Callable): Функция поиска параметров
            param_space (_type_): Переменные для поиска
            tune_config (_type_, optional): Параметры оптимизации. Defaults to tune.TuneConfig(num_samples=5).
        """
        if tune_config is None:
            tune_config = tune.TuneConfig(num_samples=5)
        self.tuner = tune.Tuner(
            func, param_space=param_space, tune_config=tune_config, **kwargs
        )
        self.results = self.tuner.fit()

    def get_best_param(self) -> dict:
        """Получить лучшие гиперпараметры

        Returns:
            dict: Словарь с лучшими гиперпараметрами
        """
        if self.results is None:
            raise RuntimeError("Optimization has not been run yet. Call run_optimization() first.")

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
                raise RuntimeError("Unable to determine best result from Ray Tune results.") from e

        cfg = getattr(best, "config", None)
        if isinstance(cfg, dict):
            return cfg

        # Last resort: try params attribute or empty dict
        params = getattr(best, "params", {})
        return dict(params) if isinstance(params, dict) else {}

    def plot_parms(self):
        """Построить график поиска гиперпараметров (WIP)

        Raises:
            NotImplementedError:  (WIP)
        """
        raise NotImplementedError()
