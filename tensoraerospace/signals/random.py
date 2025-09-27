"""Random signal generation module.

This module contains functions for creating random signals of various types,
used for control system testing and agent training.
"""

import numpy as np


def full_random_signal(
    t0: float, dt: float, tn: float, sd: tuple, sv: tuple
) -> np.ndarray:
    """Random signal with variable frequency and amplitude.

    Args:
        t0: Initial time.
        dt: Discretization step.
        tn: Signal duration.
        sd: Signal duration constraints (min, max).
        sv: Signal value constraints (min, max).

    Returns:
        np.ndarray: Array with random signal by frequency and amplitude.
    """
    sd_min, sd_max = sd
    sv_min, sv_max = sv
    n = int(np.floor((tn - t0) / dt) + 1)
    signal = [0 for _ in range(n)]
    step_start_time = t0
    step_duration = np.random.uniform(sd_min, sd_max)
    step_value = np.random.uniform(sv_min, sv_max)
    for i in range(n):
        t = t0 + i * dt
        signal[i] = step_value
        if t >= step_start_time + step_duration:
            step_start_time = t
            step_duration = np.random.uniform(sd_min, sd_max)
            step_value = np.random.uniform(sv_min, sv_max)

    return np.array(signal)
