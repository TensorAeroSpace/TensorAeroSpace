import numpy as np


def sinusoid(t: np.ndarray, amplitude: float, frequency: int) -> np.ndarray:
    """
    Синусоидальный сигнал

    Args:
        t: массив с временем
        amplitude: Амплитуда
        frequency: Частота

    Returns:
        Синусоидный сигнал

    **Пример использования**:

    >>> t0 = 0  # Начальное время
    >>> tn = 30  # Время моделирования
    >>> dt = 0.01  # Шаг дисретизации
    >>> number_time_steps = int(((tn - t0) / dt) + 1)  # Количество шагов моделирования
    >>> time = list(np.arange(t0, number_time_steps * dt, dt))
    >>> signal = sinusoid(time, 5, 6)
    """
    return np.sin(t * amplitude) * frequency
