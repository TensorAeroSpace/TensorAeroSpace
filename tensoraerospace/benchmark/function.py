from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks


def find_longest_repeating_series(numbers:list):
    """
    Находит самую длинную серию повторяющихся чисел в массиве.
    
    Args:
        numbers: list
            Массив чисел, в котором нужно найти самую длинную серию повторяющихся чисел.
    
    Returns:
        tuple: Кортеж вида (начало, конец), представляющий самую длинную серию повторяющихся чисел.
    """
    longest_series = ()
    current_series = (numbers[0], numbers[0])  # Текущая серия начинается и заканчивается первым числом
    max_length = 1
    
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i-1] + 1:
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

def find_step_function(control_signal: np.ndarray, system_signal: np.ndarray, signal_val: float = 0) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError("Массивы control_signal и system_signal должны иметь одинаковую длину.")

    index_where_step_signal_start = np.where(control_signal > signal_val)[0][0]
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
    y_final = np.mean(control_signal[int(0.9*len(control_signal)):])
    
    # Максимальное значение функции отклика системы
    M = np.max(system_signal)
    
    # Расчет перерегулирования
    output = (M - y_final) / y_final * 100
    
    return output





def settling_time(control_signal: np.ndarray, system_signal: np.ndarray, threshold: float = 0.05) -> Optional[int]:
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
    y_final = np.mean(control_signal[int(0.9*len(control_signal)):])
    

    # Определяем границы диапазона в пределах установившегося значения
    lower_bound = y_final * (1 - threshold)
    upper_bound = y_final * (1 + threshold)

    # Находим индексы, где сигнал впервые входит в этот диапазон
    within_range_indices = np.where((system_signal >= lower_bound) & (system_signal <= upper_bound))[0]

    # Если сигнал никогда не входит в диапазон, возвращаем все время моделирования
    if len(within_range_indices) == 0:
        return len(system_signal)
    return find_longest_repeating_series(within_range_indices)[0]


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
        raise ValueError("Недостаточно пиков для расчета степени затухания.")
    
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
    y_final = np.mean(system_signal[int(0.9 * len(system_signal)):])
    
    # Целевое значение - это среднее значение последних 5-10% заданного сигнала управления
    r_final = np.mean(control_signal[int(0.9 * len(control_signal)):])
    
    # Статическая ошибка - это разница между целевым значением и установившимся значением
    return r_final - y_final


def get_lower_upper_bound(control_signal: np.ndarray, epsilon: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
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
    upper = control_signal[-1] + control_signal[-1] * epsilon
    lower = control_signal[-1] - control_signal[-1] * epsilon
    return lower, upper
