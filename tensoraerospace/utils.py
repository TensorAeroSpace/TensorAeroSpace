"""General purpose utilities for TensorAeroSpace library.

This module contains general purpose helper functions used in various parts
of the library. Includes functions for working with time intervals, data
conversion and other useful utilities.

Main functions:
    - generate_time_period: Generate time period with specified sampling frequency
    - convert_tp_to_sec_tp: Convert time interval to seconds
"""

import numpy as np


def generate_time_period(tn: int = 20, dt: float = 0.01):
    """Generate time period with sampling frequency dt.

    Args:
        tn (int): Simulation time. Defaults to 20.
        dt (float): Sampling frequency. Defaults to 0.01.

    Returns:
        np.array: Time period with sampling frequency dt.
    """
    t0 = 0
    number_time_steps = int(((tn - t0) / dt) + 1)  # Количество шагов моделирования
    time = list(np.arange(0, number_time_steps * dt, dt))  # Массив с шагом dt
    return np.linspace(-0, len(time), len(time))


def convert_tp_to_sec_tp(tp: np.array, dt: float = 0.01) -> list:
    """Convert time interval tp with sampling frequency to array in seconds.

    Args:
        tp (np.array): Time period with sampling frequency dt.
        dt (float, optional): Sampling frequency. Defaults to 0.01.

    Returns:
        list: Time period in seconds.
    """
    return list(np.arange(0, len(tp) * dt, dt))
