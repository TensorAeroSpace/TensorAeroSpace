"""Functions module for control system quality analysis.

This module contains a set of functions for calculating various quality metrics
of transient processes in automatic control systems, including:
- Overshoot
- Settling time
- Damping degree
- Static error
- Rise time
- Peak time
- Integral quality criteria (IAE, ISE, ITAE)
- Other metrics for control quality assessment
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def find_longest_repeating_series(numbers: list):
    """Find the longest series of repeating numbers in an array.

    Args:
        numbers: list
            Array of numbers in which to find the longest series of repeating numbers.

    Returns:
        tuple: Tuple of the form (start, end) representing the longest series of repeating numbers.
               If the array is empty, returns (0, 0).
    """
    if len(numbers) == 0:
        return (0, 0)

    if len(numbers) == 1:
        return (numbers[0], numbers[0])

    longest_series = (numbers[0], numbers[0])
    current_series = (
        numbers[0],
        numbers[0],
    )  # Current series starts and ends with the first number
    max_length = 1

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_series = (current_series[0], numbers[i])
        else:
            if current_series[1] - current_series[0] + 1 > max_length:
                max_length = current_series[1] - current_series[0] + 1
                longest_series = current_series
            current_series = (numbers[i], numbers[i])

    # Check the last current series after loop completion
    if current_series[1] - current_series[0] + 1 > max_length:
        longest_series = current_series

    return longest_series


def find_step_function(
    control_signal: np.ndarray, system_signal: np.ndarray, signal_val: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find control system step function based on control and system signals.

    Args:
        control_signal: numpy.ndarray
            System control signal.
        system_signal: numpy.ndarray
            System signal that is affected by control.
        signal_val: float, optional (default: 0)
            Signal value from which the step function begins.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple of two arrays: updated control signal and system signal.
    """
    if len(control_signal) != len(system_signal):
        raise ValueError(
            "Arrays control_signal and system_signal must have the same length."
        )

    indices = np.where(control_signal > signal_val)[0]
    if len(indices) == 0:
        # If there are no values greater than signal_val, return original arrays
        return control_signal, system_signal
    index_where_step_signal_start = indices[0]
    control_signal = control_signal[index_where_step_signal_start:]
    system_signal = system_signal[index_where_step_signal_start:]
    return control_signal, system_signal


def overshoot(control_signal: np.ndarray, system_signal: np.ndarray) -> float:
    """
    Calculate control system overshoot based on control and system signals.

    Args:
        control_signal: numpy.ndarray
            System control signal.
        system_signal: numpy.ndarray
            System signal that is affected by control.

    Returns:
        float: Overshoot value in percent.

    """
    # Assume steady-state value is the average value of last 10% of system response
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Maximum value of system response function
    M = np.max(system_signal)

    # Overshoot calculation
    output = (M - y_final) / y_final * 100

    return output


def settling_time(
    control_signal: np.ndarray, system_signal: np.ndarray, threshold: float = 0.05
) -> Optional[int]:
    """
    Calculate control system settling time based on control and system signals.

    Args:
        control_signal: numpy.ndarray
            System control signal.
        system_signal: numpy.ndarray
            System signal that is affected by control.
        threshold: float, optional (default: 0.05)
            Threshold value of relative deviation for determining steady-state value range.

    Returns:
        Optional[int]: System settling time in system_signal array indices. If system did not reach steady-state value in given threshold range, returns None.
    """
    # Assume steady-state value is the average value of last 10% of system response
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Define range boundaries within steady-state value
    lower_bound = y_final * (1 - threshold)
    upper_bound = y_final * (1 + threshold)

    # Find indices where signal first enters this range
    within_range_indices = np.where(
        (system_signal >= lower_bound) & (system_signal <= upper_bound)
    )[0]

    # If signal never enters range, return entire simulation time
    if len(within_range_indices) == 0:
        return len(system_signal)

    # Get longest series
    longest_series = find_longest_repeating_series(within_range_indices.tolist())

    # Return start of longest series
    return longest_series[0]


def damping_degree(system_signal: np.ndarray) -> float:
    """
    Calculate system damping degree based on system signal.

    Args:
        system_signal: numpy.ndarray
            System signal for which damping degree is calculated.

    Returns:
        float:
            Average damping degree values between all peaks of system signal.

    Raises:
        ValueError: If number of peaks is less than two, cannot calculate damping degree.
    """
    # Find peaks in system signal
    peaks, _ = find_peaks(system_signal)

    # If less than two peaks, cannot calculate damping degree
    if len(peaks) < 2:
        return 0.0  # Return 0 as default value

    # Calculate peak amplitudes
    amplitudes = system_signal[peaks]

    # Damping degree calculation
    y_values = 1 - (amplitudes[1:] / amplitudes[:-1])

    return np.mean(y_values)


def static_error(control_signal: np.ndarray, system_signal: np.ndarray) -> float:
    """
    Calculate control system static error based on control and system signals.

    Args:
        control_signal: numpy.ndarray
            System control signal.
        system_signal: numpy.ndarray
            System signal that is affected by control.

    Returns:
        float: Static error value.
    """
    # Steady-state value is the average value of last 5-10% of system response
    y_final = np.mean(system_signal[int(0.9 * len(system_signal)) :])

    # Target value is the average value of last 5-10% of given control signal
    r_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Static error is the difference between target value and steady-state value
    return r_final - y_final


def get_lower_upper_bound(
    control_signal: np.ndarray, epsilon: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lower and upper bounds for control signal.

    Args:
        control_signal: numpy.ndarray
            System control signal.
        epsilon: float, optional (default: 0.05)
            Value for determining bounds.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Tuple of two arrays: lower and upper bounds for control signal.
    """
    final_value = control_signal[-1]
    upper = np.full_like(control_signal, final_value + final_value * epsilon)
    lower = np.full_like(control_signal, final_value - final_value * epsilon)
    return lower, upper


def rise_time(
    control_signal: np.ndarray,
    system_signal: np.ndarray,
    low_threshold: float = 0.1,
    high_threshold: float = 0.9,
) -> Optional[float]:
    """
    Рассчитывает время нарастания (время перехода от 10% до 90% установившегося значения).

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        low_threshold: float, optional (default: 0.1)
            Нижний порог (10% от установившегося значения).
        high_threshold: float, optional (default: 0.9)
            Верхний порог (90% от установившегося значения).

    Returns:
        Optional[float]: Время нарастания в индексах массива или None, если не удалось определить.
    """
    # Установившееся значение
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Пороговые значения
    low_val = y_final * low_threshold
    high_val = y_final * high_threshold

    # Находим индексы пересечения порогов
    low_idx = np.where(system_signal >= low_val)[0]
    high_idx = np.where(system_signal >= high_val)[0]

    if len(low_idx) == 0 or len(high_idx) == 0:
        return None

    return high_idx[0] - low_idx[0]


def peak_time(system_signal: np.ndarray) -> Optional[int]:
    """
    Рассчитывает время достижения первого максимума (пикового времени).

    Args:
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        Optional[int]: Индекс времени достижения первого максимума или None.
    """
    peaks, _ = find_peaks(system_signal)

    if len(peaks) == 0:
        # Если нет пиков, возвращаем индекс максимального значения
        return np.argmax(system_signal)

    return peaks[0]


def maximum_deviation(control_signal: np.ndarray, system_signal: np.ndarray) -> float:
    """
    Рассчитывает максимальное отклонение от установившегося значения.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float: Максимальное отклонение от установившегося значения.
    """
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])
    return np.max(np.abs(system_signal - y_final))


def integral_absolute_error(
    control_signal: np.ndarray, system_signal: np.ndarray
) -> float:
    """
    Рассчитывает интегральную абсолютную ошибку (IAE).

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float: Значение интегральной абсолютной ошибки.
    """
    error = control_signal - system_signal
    return np.sum(np.abs(error))


def integral_squared_error(
    control_signal: np.ndarray, system_signal: np.ndarray
) -> float:
    """
    Рассчитывает интегральную квадратичную ошибку (ISE).

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float: Значение интегральной квадратичной ошибки.
    """
    error = control_signal - system_signal
    return np.sum(error**2)


def integral_time_absolute_error(
    control_signal: np.ndarray, system_signal: np.ndarray, dt: float = 1.0
) -> float:
    """
    Рассчитывает интегральную абсолютную ошибку, взвешенную по времени (ITAE).

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        dt: float, optional (default: 1.0)
            Шаг дискретизации по времени.

    Returns:
        float: Значение ITAE.
    """
    error = np.abs(control_signal - system_signal)
    time_weights = np.arange(len(error)) * dt
    return np.sum(time_weights * error)


def oscillation_count(system_signal: np.ndarray, threshold: float = 0.01) -> int:
    """
    Подсчитывает количество колебаний в переходном процессе.

    Args:
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        threshold: float, optional (default: 0.01)
            Минимальная амплитуда колебания для учета.

    Returns:
        int: Количество колебаний.
    """
    # Находим пики и впадины
    peaks, _ = find_peaks(system_signal, height=threshold)
    valleys, _ = find_peaks(-system_signal, height=threshold)

    # Общее количество экстремумов
    extrema = len(peaks) + len(valleys)

    # Количество полных колебаний (пара пик-впадина)
    return extrema // 2


def steady_state_value(control_signal: np.ndarray, percentage: float = 0.1) -> float:
    """
    Рассчитывает установившееся значение сигнала.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        percentage: float, optional (default: 0.1)
            Процент от конца сигнала для усреднения (10% по умолчанию).

    Returns:
        float: Установившееся значение.
    """
    start_idx = int((1 - percentage) * len(control_signal))
    return np.mean(control_signal[start_idx:])


def performance_index(
    control_signal: np.ndarray, system_signal: np.ndarray, dt: float = 1.0
) -> float:
    """
    Рассчитывает комплексный индекс качества переходного процесса.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        dt: float, optional (default: 1.0)
            Шаг дискретизации по времени.

    Returns:
        float: Комплексный индекс качества (чем меньше, тем лучше).
    """
    # Комбинируем различные критерии качества
    ise = integral_squared_error(control_signal, system_signal)
    itae = integral_time_absolute_error(control_signal, system_signal, dt)
    overshoot_val = overshoot(control_signal, system_signal)

    # Нормализованный индекс (веса можно настраивать)
    return 0.4 * ise + 0.4 * itae + 0.2 * abs(overshoot_val)
