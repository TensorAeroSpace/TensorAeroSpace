"""
Модуль для оптимизации гиперпараметров алгоритмов управления.

Этот модуль предоставляет инструменты для автоматической оптимизации гиперпараметров
алгоритмов обучения с подкреплением и других методов управления аэрокосмическими
системами. Поддерживает различные фреймворки оптимизации, включая Optuna и Ray Tune.

Основные компоненты:
- HyperParamOptimizationOptuna: Оптимизация гиперпараметров с использованием Optuna
- HyperParamOptimizationRay: Оптимизация гиперпараметров с использованием Ray Tune
"""

from .base import HyperParamOptimizationOptuna as HyperParamOptimizationOptuna
from .ray import HyperParamOptimizationRay as HyperParamOptimizationRay
