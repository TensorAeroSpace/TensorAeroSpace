"""Base interfaces for hyperparameter optimization backends.

This module defines abstract interfaces and concrete implementations for
hyperparameter search used across TensorAeroSpace (e.g., Optuna, Ray Tune).

The intent is to provide a small, consistent API for running searches and
extracting the best found configuration.
"""

from abc import ABC
from typing import Callable

import matplotlib.pyplot as plt
import optuna


class HyperParamOptimizationBase(ABC):
    """Base interface for hyperparameter search backends."""

    def __init__(self) -> None:
        """Initialize the optimization backend."""
        pass

    def run_optimization(self):
        """Run the optimization procedure."""
        pass

    def get_best_param(self) -> dict:
        """Return the best found hyperparameters.

        Returns:
            dict: Best parameters found by the backend.
        """
        pass

    def plot_parms(self, fig_size: tuple[float, float]) -> None:
        """Plot optimization history.

        Args:
            fig_size: Figure size passed to the underlying plotting backend.
        """
        pass


class HyperParamOptimizationOptuna(HyperParamOptimizationBase):
    """Hyperparameter optimization using Optuna."""

    def __init__(self, direction: str) -> None:
        """Create an Optuna study for hyperparameter optimization.

        Args:
            direction: Optimization direction. One of ``'minimize'`` or
                ``'maximize'``.

        Raises:
            ValueError: If ``direction`` is not supported.
        """
        super().__init__()
        if direction not in ["minimize", "maximize"]:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        self.study = optuna.create_study(direction=direction)

    def run_optimization(self, func: Callable, n_trials: int):
        """Run hyperparameter search.

        Args:
            func: Objective function to optimize.
            n_trials: Number of trials to run.
        """
        self.study.optimize(func, n_trials=n_trials)

    def get_best_param(self) -> dict:
        """Return the best hyperparameters found by Optuna.

        Returns:
            dict: Best hyperparameters.
        """
        return self.study.best_trial.params

    def plot_parms(self, figsize: tuple[float, float] = (15.0, 5.0)) -> None:
        """Plot trial values over the optimization history.

        Args:
            figsize: Matplotlib figure size. Defaults to ``(15, 5)``.
        """
        x = []
        x_labels = []
        for trial in self.study.trials:
            x.append(trial.value)
            x_labels.append(
                "".join([f"{key}={trial.params[key]}\n" for key in trial.params.keys()])
            )

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(self.study.trials)), x)
        ax.set_xticks(range(len(self.study.trials)))
        ax.set_xticklabels(x_labels, rotation=90, multialignment="left")
        ax.set_title("Hyperparameter search history")
        ax.set_ylabel("Значение функции", fontsize=15)
        ax.set_xlabel("Итерации", fontsize=15)
