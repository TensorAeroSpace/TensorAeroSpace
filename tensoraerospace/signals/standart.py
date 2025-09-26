"""Standard signals module for control system testing.

This module contains functions for generating standard test signals,
including step functions, sinusoidal signals, constants and signals
with vertical shift, used for control system analysis and testing.
"""

import numpy as np


def unit_step(
    tp: np.array, degree: int, time_step: int = 10, dt: float = 0.01, output_rad=False
) -> np.array:
    """Generate step signal.

    Args:
        degree (int): Deflection angle.
        tp (np.array): Time period.
        time_step (int): Step time, defaults to 10.
        dt (float): Discretization frequency, defaults to 0.01.
        output_rad (bool): Signal output in radians, defaults to False.

    Returns:
        np.array: Step signal.
    """
    if output_rad:
        return np.deg2rad(degree) * (tp >= time_step)
    else:
        return degree * (tp >= time_step)


def sinusoid(tp: np.ndarray, frequency: float, amplitude: int) -> np.ndarray:
    """Sinusoidal signal.

    Args:
        tp (np.array): Time period.
        amplitude: Amplitude.
        frequency: Frequency.

    Returns:
        np.ndarray: Sinusoidal signal.
    """
    return np.sin(tp * amplitude) * frequency


def constant_line(tp: np.ndarray, value_state: float = 2) -> np.ndarray:
    """Straight line signal.

    Args:
        tp (np.ndarray): Time period.
        value_state (float): Value to be returned at each time step, defaults to 2.

    Returns:
        np.ndarray: Array of values equal to value_state at each time step.
    """
    return np.full_like(tp, value_state)


def sinusoid_vertical_shift(
    tp: np.ndarray, frequency: float, amplitude: float, vertical_shift: float = 0.0
) -> np.ndarray:
    """Синусоидальный сигнал с вертикальным сдвигом

    Args:
        tp (np.ndarray): Временной промежуток
        frequency (float): Частота волны
        amplitude (float): Амплитуда волны
        vertical_shift (float): Вертикальный сдвиг волны по умолчанию равен 0.0

    Returns:
        np.ndarray: Синусоидальный сигнал, колеблющийся между значениями (vertical_shift + amplitude) и (vertical_shift - amplitude)
    """
    return amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift
