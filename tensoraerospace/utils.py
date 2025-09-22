"""
Утилиты общего назначения для библиотеки TensorAeroSpace.

Этот модуль содержит вспомогательные функции общего назначения, используемые
в различных частях библиотеки. Включает функции для работы с временными
интервалами, преобразования данных и другие полезные утилиты.

Основные функции:
- generate_time_period: Генерация временного промежутка с заданной частотой дискретизации
- convert_tp_to_sec_tp: Конвертирование временного интервала в секунды
"""

import numpy as np


def generate_time_period(tn: int = 20, dt: float = 0.01):
    """Генерация временного промежутка с частотой дискретизации dt

    Args:
        tn (int): Время моделирования. Defaults to 20.
        dt (float): Частота дискретизации. Defaults to 0.01.

    Returns:
        np.array: Временной промежуток с частотой дискретизации dt
    """
    t0 = 0
    number_time_steps = int(((tn - t0) / dt) + 1)  # Количество шагов моделирования
    time = list(np.arange(0, number_time_steps * dt, dt))  # Массив с шагом dt
    return np.linspace(-0, len(time), len(time))


def convert_tp_to_sec_tp(tp: np.array, dt: float = 0.01) -> list:
    """Конвертирование временного интервала tp с частотой дискретизации в массив в секундах

    Args:
        tp (np.array): Временной промежуток с частотой дискретизации dt
        dt (float, optional): Частота дискретизации. Defaults to 0.01.

    Returns:
        np.array: Временной промежуток в секундах
    """
    return list(np.arange(0, len(tp) * dt, dt))
