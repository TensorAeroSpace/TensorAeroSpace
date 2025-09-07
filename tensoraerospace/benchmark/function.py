"""
Модуль функций для анализа качества систем управления.

Этот модуль содержит набор функций для расчета различных метрик качества
переходных процессов в системах автоматического управления, включая:
- Перерегулирование (overshoot)
- Время установления (settling time)
- Степень затухания (damping degree)
- Статическую ошибку (static error)
- Время нарастания (rise time)
- Время достижения пика (peak time)
- Интегральные критерии качества (IAE, ISE, ITAE)
- И другие метрики для оценки качества управления
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def find_longest_repeating_series(numbers: list):
    """
    Находит самую длинную серию повторяющихся чисел в массиве.

    Args:
        numbers: list
            Массив чисел, в котором нужно найти самую длинную серию повторяющихся чисел.

    Returns:
        tuple: Кортеж вида (начало, конец), представляющий самую длинную серию повторяющихся чисел.
               Если массив пустой, возвращает (0, 0).
    """
    if len(numbers) == 0:
        return (0, 0)

    if len(numbers) == 1:
        return (numbers[0], numbers[0])

    longest_series = (numbers[0], numbers[0])
    current_series = (
        numbers[0],
        numbers[0],
    )  # Текущая серия начинается и заканчивается первым числом
    max_length = 1

    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_series = (current_series[0], numbers[i])
        else:
            if current_series[1] - current_series[0] + 1 > max_length:
                max_length = current_series[1] - current_series[0] + 1
                longest_series = current_series
            current_series = (numbers[i], numbers[i])

    # Проверяем последнюю текущую серию после завершения цикла
    if current_series[1] - current_series[0] + 1 > max_length:
        longest_series = current_series

    return longest_series


def find_step_function(
    control_signal: np.ndarray, system_signal: np.ndarray, signal_val: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Находит функцию перехода системы управления на основе сигналов управления и системы.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        signal_val: float, optional (default: 0)
            Значение сигнала, с которого начинается функция перехода.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Кортеж из двух массивов: обновленный сигнал управления и сигнал системы.
    """
    if len(control_signal) != len(system_signal):
        raise ValueError(
            "Массивы control_signal и system_signal должны иметь одинаковую длину."
        )

    indices = np.where(control_signal > signal_val)[0]
    if len(indices) == 0:
        # Если нет значений больше signal_val, возвращаем исходные массивы
        return control_signal, system_signal
    index_where_step_signal_start = indices[0]
    control_signal = control_signal[index_where_step_signal_start:]
    system_signal = system_signal[index_where_step_signal_start:]
    return control_signal, system_signal


def overshoot(control_signal: np.ndarray, system_signal: np.ndarray) -> float:
    """
    Рассчитывает перерегулирование системы управления на основе сигналов управления и системы.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float: Значение перерегулирования в процентах.

    """
    # Предполагаем, что установившееся значение - это среднее значение последних 10% отклика системы
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Максимальное значение функции отклика системы
    M = np.max(system_signal)

    # Расчет перерегулирования
    output = (M - y_final) / y_final * 100

    return output


def settling_time(
    control_signal: np.ndarray, system_signal: np.ndarray, threshold: float = 0.05
) -> Optional[int]:
    """
    Рассчитывает время установления системы управления на основе сигналов управления и системы.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.
        threshold: float, optional (default: 0.05)
            Пороговое значение относительного отклонения для определения диапазона установившегося значения.

    Returns:
        Optional[int]: Время установления системы в индексах массива system_signal. Если система не достигла установившегося значения в заданном пороговом диапазоне, возвращается None.
    """
    # Предполагаем, что установившееся значение - это среднее значение последних 10% отклика системы
    y_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Определяем границы диапазона в пределах установившегося значения
    lower_bound = y_final * (1 - threshold)
    upper_bound = y_final * (1 + threshold)

    # Находим индексы, где сигнал впервые входит в этот диапазон
    within_range_indices = np.where(
        (system_signal >= lower_bound) & (system_signal <= upper_bound)
    )[0]

    # Если сигнал никогда не входит в диапазон, возвращаем все время моделирования
    if len(within_range_indices) == 0:
        return len(system_signal)

    # Получаем самую длинную серию
    longest_series = find_longest_repeating_series(within_range_indices.tolist())

    # Возвращаем начало самой длинной серии
    return longest_series[0]


def damping_degree(system_signal: np.ndarray) -> float:
    """
    Рассчитывает степень затухания системы управления на основе сигналов управления и системы.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float:
            Среднее значений степени затухания между всеми пиками сигнала системы.

    Raises:
        ValueError: Если количество пиков меньше двух, невозможно рассчитать степень затухания.
    """
    # Находим пики в сигнале системы
    peaks, _ = find_peaks(system_signal)

    # Если пиков меньше двух, то нельзя рассчитать степень затухания
    if len(peaks) < 2:
        return 0.0  # Возвращаем 0 как значение по умолчанию

    # Вычисляем амплитуды пиков
    amplitudes = system_signal[peaks]

    # Расчет степени затухания
    y_values = 1 - (amplitudes[1:] / amplitudes[:-1])

    return np.mean(y_values)


def static_error(control_signal: np.ndarray, system_signal: np.ndarray) -> float:
    """
    Рассчитывает статическую ошибку системы управления на основе сигналов управления и системы.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        system_signal: numpy.ndarray
            Сигнал системы, на которую воздействует управление.

    Returns:
        float: Значение статической ошибки.
    """
    # Установившееся значение - это среднее значение последних 5-10% отклика системы
    y_final = np.mean(system_signal[int(0.9 * len(system_signal)) :])

    # Целевое значение - это среднее значение последних 5-10% заданного сигнала управления
    r_final = np.mean(control_signal[int(0.9 * len(control_signal)) :])

    # Статическая ошибка - это разница между целевым значением и установившимся значением
    return r_final - y_final


def get_lower_upper_bound(
    control_signal: np.ndarray, epsilon: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает нижнюю и верхнюю границы для сигнала управления.

    Args:
        control_signal: numpy.ndarray
            Сигнал управления системы.
        epsilon: float, optional (default: 0.05)
            Значение для определения границ.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Кортеж из двух массивов: нижняя и верхняя границы для сигнала управления.
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
