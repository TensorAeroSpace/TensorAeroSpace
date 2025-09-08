"""
Модуль генерации сигналов для тестирования и обучения.

Этот модуль предоставляет различные типы сигналов для использования в симуляциях
и тестировании систем управления, включая случайные сигналы, синусоидальные
сигналы и единичные ступенчатые функции.
"""

from .random import full_random_signal as full_random_signal
from .standart import sinusoid as sinusoid
from .standart import unit_step as unit_step
