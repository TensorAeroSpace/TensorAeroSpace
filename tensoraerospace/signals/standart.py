"""Standard signals module for control system testing.

This module contains functions for generating standard test signals,
including step functions, sinusoidal signals, constants and signals
with vertical shift, used for control system analysis and testing.
"""

from typing import Optional

import numpy as np


def unit_step(
    tp: np.ndarray,
    degree: int,
    time_step: int = 10,
    dt: float = 0.01,
    output_rad: bool = False,
) -> np.ndarray:
    """Generate unit step signal for control system testing.

    Creates a step function that transitions from 0 to a specified amplitude
    at a given time. Commonly used for analyzing system transient response
    and step response characteristics.

    Args:
        tp: Time array in seconds.
        degree: Step amplitude (deflection angle in degrees if
            output_rad=False, or will be converted to radians if
            output_rad=True).
        time_step: Time at which the step occurs, in seconds. Defaults to 10.
        dt: Discretization time step, in seconds. Defaults to 0.01.
            (Note: This parameter is not currently used in the function).
        output_rad: If True, converts degree to radians. Defaults to False.

    Returns:
        Step signal array of the same shape as tp, with values of 0 before
        time_step and the specified amplitude after time_step.

    Examples:
        >>> t = np.linspace(0, 20, 2001)
        >>> step = unit_step(t, degree=5, time_step=5.0, output_rad=False)
        >>> # Creates a step of amplitude 5 starting at t=5s
    """
    if output_rad:
        return np.deg2rad(degree) * (tp >= time_step)
    else:
        return degree * (tp >= time_step)


def sinusoid(tp: np.ndarray, frequency: float, amplitude: int) -> np.ndarray:
    """Generate sinusoidal signal for frequency analysis.

    Creates a sinusoidal signal commonly used for frequency response analysis
    and harmonic testing of control systems.

    Note:
        Current implementation uses np.sin(tp * amplitude) * frequency.
        For standard sinusoidal behavior, consider using:
        amplitude * np.sin(2 * np.pi * frequency * tp)

    Args:
        tp: Time array in seconds.
        frequency: Signal frequency multiplier (note: not in Hz).
        amplitude: Signal amplitude multiplier (note: affects phase).

    Returns:
        Sinusoidal signal array of the same shape as tp.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> sine = sinusoid(t, frequency=1.0, amplitude=1.0)
    """
    return np.sin(tp * amplitude) * frequency


def constant_line(tp: np.ndarray, value_state: float = 2) -> np.ndarray:
    """Generate constant reference signal.

    Creates a constant-valued signal useful for setpoint tracking tests
    and steady-state analysis of control systems.

    Args:
        tp: Time array in seconds.
        value_state: Constant value to maintain throughout the signal.
            Defaults to 2.

    Returns:
        Array of constant values with the same shape as tp, where all
        elements equal value_state.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> const = constant_line(t, value_state=5.0)
        >>> # Creates a signal with constant value 5.0
    """
    return np.full_like(tp, value_state)


def sinusoid_vertical_shift(
    tp: np.ndarray,
    frequency: float,
    amplitude: float,
    vertical_shift: float = 0.0,
) -> np.ndarray:
    """Generate sinusoidal signal with DC offset.

    Creates a sinusoidal signal with a vertical offset, useful for testing
    systems with non-zero operating points or bias conditions.

    Args:
        tp: Time array in seconds.
        frequency: Oscillation frequency in Hz.
        amplitude: Peak amplitude of oscillation.
        vertical_shift: DC offset (vertical shift) of the signal.
            Defaults to 0.0.

    Returns:
        Sinusoidal signal array oscillating between
        (vertical_shift - amplitude) and (vertical_shift + amplitude).

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> signal = sinusoid_vertical_shift(
        ...     t, frequency=0.5, amplitude=2.0, vertical_shift=5.0)
        >>> # Signal oscillates between 3.0 and 7.0
    """
    return amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift


def ramp(tp: np.ndarray, slope: float = 1.0, time_start: float = 0.0) -> np.ndarray:
    """Generate ramp (linearly increasing) signal.

    Creates a ramp signal for testing the tracking capability of control
    systems to linearly varying reference trajectories.

    Args:
        tp: Time array in seconds.
        slope: Rate of increase (units per second). Defaults to 1.0.
        time_start: Time at which the ramp begins, in seconds.
            Before this time, the signal is zero. Defaults to 0.0.

    Returns:
        Ramp signal array that increases linearly with the specified slope
        after time_start, and is zero before time_start.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> ramp_sig = ramp(t, slope=0.5, time_start=2.0)
        >>> # Creates a ramp starting at t=2s with slope 0.5
    """
    return slope * np.maximum(tp - time_start, 0)


def pulse(
    tp: np.ndarray,
    amplitude: float = 1.0,
    time_start: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """Generate rectangular pulse signal.

    Creates a pulse signal for analyzing impulse response and transient
    behavior of control systems.

    Args:
        tp: Time array in seconds.
        amplitude: Pulse amplitude (height). Defaults to 1.0.
        time_start: Time at which pulse begins, in seconds. Defaults to 0.0.
        width: Duration of the pulse, in seconds. Defaults to 1.0.

    Returns:
        Pulse signal array with value 'amplitude' from time_start to
        (time_start + width), and zero elsewhere.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> pulse_sig = pulse(t, amplitude=5.0, time_start=3.0, width=2.0)
        >>> # Creates a pulse from t=3s to t=5s with amplitude 5.0
    """
    return amplitude * ((tp >= time_start) & (tp < time_start + width))


def square_wave(
    tp: np.ndarray,
    frequency: float = 1.0,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
) -> np.ndarray:
    """Generate square wave signal.

    Creates a periodic square wave useful for switching control systems,
    relay-based control, and testing system response to periodic ON/OFF inputs.

    Args:
        tp: Time array in seconds.
        frequency: Oscillation frequency in Hz. Defaults to 1.0.
        amplitude: Signal amplitude (high level). Low level is always 0.
            Defaults to 1.0.
        duty_cycle: Fraction of each period where signal is high (0 to 1).
            For example, 0.5 means signal is high for 50% of each period.
            Defaults to 0.5.

    Returns:
        Square wave signal array alternating between 0 and amplitude at the
        specified frequency and duty cycle.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> sq = square_wave(t, frequency=0.5, amplitude=3.0, duty_cycle=0.3)
        >>> # Square wave at 0.5 Hz, high for 30% of each cycle
    """
    phase = (tp * frequency) % 1.0
    return amplitude * (phase < duty_cycle).astype(float)


def sawtooth(
    tp: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    """Generate sawtooth wave signal.

    Creates a periodic sawtooth wave with linear increase from negative
    to positive amplitude. Useful for sweep generators and linear ramp
    testing.

    Args:
        tp: Time array in seconds.
        frequency: Oscillation frequency in Hz. Defaults to 1.0.
        amplitude: Peak amplitude. Signal ranges from -amplitude to +amplitude.
            Defaults to 1.0.

    Returns:
        Sawtooth wave signal array with values linearly increasing from
        -amplitude to +amplitude within each period.

    Examples:
        >>> t = np.linspace(0, 5, 1000)
        >>> saw = sawtooth(t, frequency=0.5, amplitude=2.0)
        >>> # Sawtooth wave oscillating between -2.0 and +2.0
    """
    phase = (tp * frequency) % 1.0
    return amplitude * (2 * phase - 1)


def triangular_wave(
    tp: np.ndarray, frequency: float = 1.0, amplitude: float = 1.0
) -> np.ndarray:
    """Generate triangular wave signal.

    Creates a periodic triangular wave with symmetric rise and fall times.
    Useful for smooth periodic reference signals in control system testing.

    Args:
        tp: Time array in seconds.
        frequency: Oscillation frequency in Hz. Defaults to 1.0.
        amplitude: Peak amplitude. Signal ranges from -amplitude to +amplitude.
            Defaults to 1.0.

    Returns:
        Triangular wave signal array with values varying linearly between
        -amplitude and +amplitude with equal rise and fall times.

    Examples:
        >>> t = np.linspace(0, 5, 1000)
        >>> tri = triangular_wave(t, frequency=0.5, amplitude=3.0)
        >>> # Triangular wave oscillating between -3.0 and +3.0
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
    """Generate chirp signal with frequency sweep.

    Creates a swept-frequency sinusoid (chirp) for system identification
    and frequency response analysis. The instantaneous frequency increases
    from f0 to f1 over the duration of the signal.

    Args:
        tp: Time array in seconds.
        f0: Starting frequency in Hz. Defaults to 0.1.
        f1: Ending frequency in Hz. Defaults to 1.0.
        amplitude: Signal amplitude (peak value). Defaults to 1.0.
        method: Frequency sweep method. Options are:
            - 'linear': Linear frequency increase (constant rate)
            - 'exponential': Exponential frequency increase (logarithmic)
            Defaults to 'linear'.

    Returns:
        Chirp signal array with frequency varying from f0 to f1.

    Raises:
        ValueError: If method is not 'linear' or 'exponential'.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> chirp_sig = chirp(t, f0=0.1, f1=5.0, amplitude=2.0,
        ...                   method='linear')
        >>> # Linear frequency sweep from 0.1 Hz to 5.0 Hz
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
    """Generate doublet signal for stability analysis.

    Creates a doublet signal consisting of a positive pulse followed
    immediately by a negative pulse of equal magnitude and duration.
    Commonly used in aerospace for stability and control analysis,
    particularly for aircraft handling qualities testing.

    Args:
        tp: Time array in seconds.
        amplitude: Amplitude of both positive and negative pulses.
            Defaults to 1.0.
        time_start: Time at which the doublet begins, in seconds.
            Defaults to 0.0.
        width: Duration of each pulse (positive and negative), in seconds.
            Total doublet duration is 2*width. Defaults to 1.0.

    Returns:
        Doublet signal array with a positive pulse from time_start to
        (time_start + width), followed by a negative pulse from
        (time_start + width) to (time_start + 2*width), and zero elsewhere.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> doublet_sig = doublet(t, amplitude=np.deg2rad(5),
        ...                       time_start=3.0, width=1.0)
        >>> # 5-degree doublet maneuver starting at t=3s
    """
    positive_pulse = amplitude * ((tp >= time_start) & (tp < time_start + width))
    negative_pulse = -amplitude * (
        (tp >= time_start + width) & (tp < time_start + 2 * width)
    )
    return positive_pulse + negative_pulse


def multi_step(tp: np.ndarray, step_times: list, step_values: list) -> np.ndarray:
    """Generate multi-step signal with multiple setpoint changes.

    Creates a signal with multiple step changes at specified times, useful
    for testing tracking performance and setpoint response of control systems.
    Each step adds cumulatively to create a staircase pattern.

    Args:
        tp: Time array in seconds.
        step_times: List of times (in seconds) when each step occurs.
            Must have the same length as step_values.
        step_values: List of step magnitudes. Each value is added to the
            cumulative signal at the corresponding step_time.
            Must have the same length as step_times.

    Returns:
        Multi-step signal array where the signal value at any time is the
        cumulative sum of all step_values that have occurred up to that time.

    Raises:
        ValueError: If step_times and step_values have different lengths.

    Examples:
        >>> t = np.linspace(0, 20, 2000)
        >>> steps = multi_step(t, step_times=[2, 5, 10, 15],
        ...                    step_values=[1, 2, -1, 3])
        >>> # Signal: 0→1 at t=2, 1→3 at t=5, 3→2 at t=10, 2→5 at t=15
    """
    if len(step_times) != len(step_values):
        raise ValueError("step_times and step_values must have the same length")

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
    """Generate exponential approach signal.

    Creates a signal that exponentially approaches a final value, modeling
    first-order system response or smooth transitions in control systems.
    The signal follows the form: amplitude * (1 - e^(-t/τ)).

    Args:
        tp: Time array in seconds.
        amplitude: Final asymptotic value that the signal approaches.
            Defaults to 1.0.
        time_constant: Time constant τ (tau) in seconds. The signal reaches
            approximately 63.2% of the final value after one time constant.
            Defaults to 1.0.
        time_start: Time at which the exponential rise begins, in seconds.
            Signal is zero before this time. Defaults to 0.0.

    Returns:
        Exponential signal array that rises from 0 to approach amplitude
        with the specified time constant, starting at time_start.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> exp_sig = exponential(t, amplitude=10.0, time_constant=2.0,
        ...                       time_start=1.0)
        >>> # Exponential rise from 0 to 10 with τ=2s, starting at t=1s
    """
    t_shifted = np.maximum(tp - time_start, 0)
    return amplitude * (1 - np.exp(-t_shifted / time_constant)) * (tp >= time_start)


def gaussian_pulse(
    tp: np.ndarray,
    amplitude: float = 1.0,
    center: float = 0.0,
    width: float = 1.0,
) -> np.ndarray:
    """Generate Gaussian-shaped pulse signal.

    Creates a smooth, bell-shaped pulse based on the Gaussian (normal)
    distribution. Useful for modeling smooth disturbances and analyzing
    system response to band-limited excitations.

    Args:
        tp: Time array in seconds.
        amplitude: Peak amplitude of the pulse at its center. Defaults to 1.0.
        center: Time at which the pulse reaches its peak, in seconds.
            Defaults to 0.0.
        width: Standard deviation (σ) of the Gaussian distribution, in seconds.
            Controls the pulse width; larger values create wider pulses.
            Defaults to 1.0.

    Returns:
        Gaussian pulse signal array with peak at 'center' and shape
        determined by 'width' (standard deviation).

    Examples:
        >>> t = np.linspace(0, 20, 2000)
        >>> gauss = gaussian_pulse(t, amplitude=5.0, center=10.0, width=1.5)
        >>> # Gaussian pulse centered at t=10s with σ=1.5s
    """
    return amplitude * np.exp(-((tp - center) ** 2) / (2 * width**2))


def multisine(
    tp: np.ndarray,
    frequencies: list,
    amplitudes: list,
    phases: Optional[list] = None,
) -> np.ndarray:
    """Generate multi-sine signal for multi-frequency excitation.

    Creates a signal that is the sum of multiple sinusoids with different
    frequencies, amplitudes, and phases. Particularly useful for system
    identification, frequency response testing, and MIMO system analysis
    where simultaneous excitation at multiple frequencies is desired.

    Args:
        tp: Time array in seconds.
        frequencies: List of frequencies in Hz for each sinusoidal component.
            Must have the same length as amplitudes.
        amplitudes: List of amplitudes for each sinusoidal component.
            Must have the same length as frequencies.
        phases: List of phase shifts in radians for each component.
            If None, all phases are set to 0. If provided, must have
            the same length as frequencies. Defaults to None.

    Returns:
        Multi-sine signal array that is the sum of all sinusoidal
        components with specified parameters.

    Raises:
        ValueError: If frequencies and amplitudes have different lengths,
            or if phases (when provided) has different length than frequencies.

    Examples:
        >>> t = np.linspace(0, 10, 1000)
        >>> ms = multisine(t, frequencies=[0.5, 1.0, 2.0],
        ...                amplitudes=[2.0, 1.5, 1.0],
        ...                phases=[0, np.pi/4, np.pi/2])
        >>> # Sum of three sinusoids at 0.5, 1.0, and 2.0 Hz
    """
    if len(frequencies) != len(amplitudes):
        raise ValueError("frequencies and amplitudes must have the same length")

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
    """Generate damped (decaying) sinusoidal signal.

    Creates an exponentially decaying sinusoidal signal, characteristic of
    underdamped second-order systems. Useful for modeling oscillatory
    responses with energy dissipation, such as mechanical vibrations or
    RLC circuit responses.

    The signal follows: amplitude * exp(-ζt) * sin(2πft) where ζ is the
    damping coefficient.

    Args:
        tp: Time array in seconds.
        frequency: Oscillation frequency in Hz. Defaults to 1.0.
        amplitude: Initial amplitude at time_start (before damping).
            Defaults to 1.0.
        damping: Damping coefficient (ζ, zeta) that controls the rate of
            exponential decay. Larger values cause faster decay.
            Defaults to 0.1.
        time_start: Time at which oscillations begin, in seconds.
            Signal is zero before this time. Defaults to 0.0.

    Returns:
        Damped sinusoidal signal array that oscillates at the specified
        frequency while exponentially decaying in amplitude.

    Examples:
        >>> t = np.linspace(0, 20, 2000)
        >>> damp_sin = damped_sinusoid(t, frequency=1.0, amplitude=5.0,
        ...                            damping=0.2, time_start=2.0)
        >>> # Decaying oscillation at 1 Hz, starting at t=2s
    """
    t_shifted = np.maximum(tp - time_start, 0)
    envelope = np.exp(-damping * t_shifted)
    return (
        amplitude
        * envelope
        * np.sin(2 * np.pi * frequency * t_shifted)
        * (tp >= time_start)
    )
