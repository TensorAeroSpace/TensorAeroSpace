import numpy as np

def unit_step(tp:np.array, degree:int, time_step:int=10, dt:float=0.01, output_rad=False):
    """Генерация ступенчатого сигнала

    Args:
        degree (int): Угол отлонения
        tp (np.array): Временной промежуток
        time_step (ing): Время ступеньки
        dt (float): Частота дискретизации
        out_put_rad (bool): Выход сигнала в радианах

    Returns:
        _type_: _description_
    """
    if output_rad:
       return np.deg2rad(degree) * (tp > time_step/dt)
    else:
        return degree * (tp > time_step/dt)



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
