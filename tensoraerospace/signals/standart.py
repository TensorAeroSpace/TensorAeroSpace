"""Standard signals module for control system testing.

This module contains functions for generating standard test signals,
including step functions, sinusoidal signals, constants and signals
with vertical shift, used for control system analysis and testing.
"""

import numpy as np


def unit_step(
    tp: np.ndarray,
    degree: int,
    time_step: int = 10,
    dt: float = 0.01,
    output_rad: bool = False,
) -> np.ndarray:
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
        value_state (float): Value to be returned at each time step,
            defaults to 2.

    Returns:
        np.ndarray: Array of values equal to value_state at each time
            step.
    """
    return np.full_like(tp, value_state)


def sinusoid_vertical_shift(
    tp: np.ndarray,
    frequency: float,
    amplitude: float,
    vertical_shift: float = 0.0,
) -> np.ndarray:
    """Sinusoidal signal with vertical shift.

    Args:
        tp (np.ndarray): Time period.
        frequency (float): Wave frequency.
        amplitude (float): Wave amplitude.
        vertical_shift (float): Vertical wave shift, defaults to 0.0.

    Returns:
        np.ndarray: Sinusoidal signal oscillating between
            (vertical_shift + amplitude) and
            (vertical_shift - amplitude).
    """
    return amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift


def ramp(
    tp: np.ndarray, slope: float = 1.0, time_start: float = 0.0
) -> np.ndarray:
    """Generate ramp signal.

    Args:
        tp (np.ndarray): Time period.
        slope (float): Slope of the ramp, defaults to 1.0.
        time_start (float): Time when ramp starts, defaults to 0.0.

    Returns:
        np.ndarray: Ramp signal.
    """
    return slope * np.maximum(tp - time_start, 0)


def pulse(
    tp: np.ndarray,
    amplitude: float = 1.0,
    time_start: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """Generate pulse signal.

    Args:
        tp (np.ndarray): Time period.
        amplitude (float): Pulse amplitude, defaults to 1.0.
        time_start (float): Pulse start time, defaults to 0.0.
        width (float): Pulse width (duration), defaults to 1.0.

    Returns:
        np.ndarray: Pulse signal.
    """
    return amplitude * ((tp >= time_start) & (tp < time_start + width))


def square_wave(
    tp: np.ndarray,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
) -> np.ndarray:
    """Generate square wave signal.

    Args:
        tp (np.ndarray): Time period.
        frequency (float): Wave frequency in Hz, defaults to 1.0.
        amplitude (float): Wave amplitude, defaults to 1.0.
        duty_cycle (float): Duty cycle (fraction of period signal is high),
            defaults to 0.5.

    Returns:
        np.ndarray: Square wave signal.
    """
    phase = (tp * frequency) % 1.0
    return amplitude * (phase < duty_cycle).astype(float)


def sawtooth(
    tp: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    """Generate sawtooth signal.

    Args:
        tp (np.ndarray): Time period.
        frequency (float): Wave frequency in Hz, defaults to 1.0.
        amplitude (float): Wave amplitude, defaults to 1.0.

    Returns:
        np.ndarray: Sawtooth signal.
    """
    phase = (tp * frequency) % 1.0
    return amplitude * (2 * phase - 1)


def triangular_wave(
    tp: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    """Generate triangular wave signal.

    Args:
        tp (np.ndarray): Time period.
        frequency (float): Wave frequency in Hz, defaults to 1.0.
        amplitude (float): Wave amplitude, defaults to 1.0.

    Returns:
        np.ndarray: Triangular wave signal.
    """
    phase = (tp * frequency) % 1.0
    return amplitude * (2 * np.abs(2 * phase - 1) - 1)


def chirp(
    tp: np.ndarray,
    f0: float = 0.1,
    f1: float = 1.0,
    amplitude: float = 1.0,
    method: str = "linear",
) -> np.ndarray:
    """Generate chirp signal (swept-frequency sinusoid).

    Args:
        tp (np.ndarray): Time period.
        f0 (float): Starting frequency in Hz, defaults to 0.1.
        f1 (float): Ending frequency in Hz, defaults to 1.0.
        amplitude (float): Signal amplitude, defaults to 1.0.
        method (str): Frequency sweep method ('linear' or 'exponential'),
            defaults to 'linear'.

    Returns:
        np.ndarray: Chirp signal.
    """
    if len(tp) == 0:
        return tp

    t_max = tp[-1] if tp[-1] > tp[0] else 1.0

    if method == "linear":
        # Linear frequency sweep
        phase = 2 * np.pi * (f0 * tp + (f1 - f0) * tp**2 / (2 * t_max))
    elif method == "exponential":
        # Exponential frequency sweep
        k = (f1 / f0) ** (1.0 / t_max)
        phase = 2 * np.pi * f0 * (k**tp - 1) / np.log(k)
    else:
        raise ValueError("method must be 'linear' or 'exponential'")

    return amplitude * np.sin(phase)


def doublet(
    tp: np.ndarray,
    amplitude: float = 1.0,
    time_start: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """Generate doublet signal (positive then negative pulse).

    Args:
        tp (np.ndarray): Time period.
        amplitude (float): Doublet amplitude, defaults to 1.0.
        time_start (float): Doublet start time, defaults to 0.0.
        width (float): Width of each pulse, defaults to 1.0.

    Returns:
        np.ndarray: Doublet signal.
    """
    positive_pulse = amplitude * (
        (tp >= time_start) & (tp < time_start + width)
    )
    negative_pulse = -amplitude * (
        (tp >= time_start + width) & (tp < time_start + 2 * width)
    )
    return positive_pulse + negative_pulse


def multi_step(
    tp: np.ndarray, step_times: list, step_values: list
) -> np.ndarray:
    """Generate multi-step signal.

    Args:
        tp (np.ndarray): Time period.
        step_times (list): List of times when steps occur.
        step_values (list): List of values for each step.

    Returns:
        np.ndarray: Multi-step signal.
    """
    if len(step_times) != len(step_values):
        raise ValueError(
            "step_times and step_values must have the same length"
        )

    signal = np.zeros_like(tp)
    for time, value in zip(step_times, step_values):
        signal += value * (tp >= time)

    return signal


def exponential(
    tp: np.ndarray,
    amplitude: float = 1.0,
    time_constant: float = 1.0,
    time_start: float = 0.0,
) -> np.ndarray:
    """Generate exponential signal.

    Args:
        tp (np.ndarray): Time period.
        amplitude (float): Signal amplitude, defaults to 1.0.
        time_constant (float): Time constant, defaults to 1.0.
        time_start (float): Signal start time, defaults to 0.0.

    Returns:
        np.ndarray: Exponential signal.
    """
    t_shifted = np.maximum(tp - time_start, 0)
    return (
        amplitude
        * (1 - np.exp(-t_shifted / time_constant))
        * (tp >= time_start)
    )


def gaussian_pulse(
    tp: np.ndarray,
    amplitude: float = 1.0,
    center: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """Generate Gaussian pulse signal.

    Args:
        tp (np.ndarray): Time period.
        amplitude (float): Pulse amplitude, defaults to 1.0.
        center (float): Center time of the pulse, defaults to 0.0.
        width (float): Standard deviation (width) of the Gaussian, defaults to 1.0.

    Returns:
        np.ndarray: Gaussian pulse signal.
    """
    return amplitude * np.exp(-((tp - center) ** 2) / (2 * width**2))


def multisine(
    tp: np.ndarray,
    frequencies: list,
    amplitudes: list,
    phases: list | None = None,
) -> np.ndarray:
    """Generate multi-sine signal (sum of multiple sinusoids).

    Args:
        tp (np.ndarray): Time period.
        frequencies (list): List of frequencies for each sinusoid.
        amplitudes (list): List of amplitudes for each sinusoid.
        phases (list | None): List of phase shifts (in radians) for each
            sinusoid, defaults to None (all zeros).

    Returns:
        np.ndarray: Multi-sine signal.
    """
    if len(frequencies) != len(amplitudes):
        raise ValueError(
            "frequencies and amplitudes must have the same length"
        )

    if phases is None:
        phases = [0.0] * len(frequencies)
    elif len(phases) != len(frequencies):
        raise ValueError("phases must have the same length as frequencies")

    signal = np.zeros_like(tp)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += amp * np.sin(2 * np.pi * freq * tp + phase)

    return signal


def damped_sinusoid(
    tp: np.ndarray,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    damping: float = 0.1,
    time_start: float = 0.0,
) -> np.ndarray:
    """Generate damped sinusoidal signal.

    Args:
        tp (np.ndarray): Time period.
        frequency (float): Oscillation frequency in Hz, defaults to 1.0.
        amplitude (float): Initial amplitude, defaults to 1.0.
        damping (float): Damping coefficient, defaults to 0.1.
        time_start (float): Signal start time, defaults to 0.0.

    Returns:
        np.ndarray: Damped sinusoidal signal.
    """
    t_shifted = np.maximum(tp - time_start, 0)
    envelope = np.exp(-damping * t_shifted)
    return (
        amplitude
        * envelope
        * np.sin(2 * np.pi * frequency * t_shifted)
        * (tp >= time_start)
    )
