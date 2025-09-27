"""Module for control algorithm hyperparameter optimization.

This module provides tools for automatic hyperparameter optimization
of reinforcement learning algorithms and other aerospace system control methods.
Supports various optimization frameworks, including Optuna and Ray Tune.

Main components:
    - HyperParamOptimizationOptuna: Hyperparameter optimization using Optuna
    - HyperParamOptimizationRay: Hyperparameter optimization using Ray Tune
"""

from .base import HyperParamOptimizationOptuna as HyperParamOptimizationOptuna
from .ray import HyperParamOptimizationRay as HyperParamOptimizationRay
