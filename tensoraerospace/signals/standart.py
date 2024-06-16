import numpy as np


def unit_step(tp:np.array, degree:int, time_step:int=10, dt:float=0.01, output_rad=False)-> np.array:
    """Генерация ступенчатого сигнала

    Args:
        degree (int): Угол отклонения
        tp (np.array): Временной промежуток
        time_step (ing): Время ступеньки
        dt (float): Частота дискретизации
        out_put_rad (bool): Выход сигнала в радианах

    Returns:
        np.array: Ступенчатый сигнал
    """
    if output_rad:
       return np.deg2rad(degree) * (tp > time_step/dt)
    else:
        return degree * (tp > time_step/dt)



def sinusoid(tp: np.ndarray, frequency : float, amplitude: int) -> np.ndarray:
    """Синусоидальный сигнал

    Args:
        tp (np.array): Временной промежуток
        amplitude: Амплитуда
        frequency: Частота

    Returns:
        Синусоидный сигнал
    """
    return np.sin(tp * amplitude) * frequency 

def constant_line(tp: np.ndarray, value_state:float = 2) -> np.ndarray:
    """Прямая линия y = 5

    Args:
        tp (np.ndarray): Временной промежуток

    Returns:
        np.ndarray: Массив значений y, равных 5 на каждом временном шаге
    """
    return np.full_like(tp, value_state)

def sinusoid_vertical_shift(tp: np.ndarray, frequency: float, amplitude: float, vertical_shift: float= 0.0) -> np.ndarray:
    """Синусоидальный сигнал

    Args:
        tp (np.ndarray): Временной промежуток
        frequency (float): Частота волны
        amplitude (float): Амплитуда волны
        vertical_shift (float): Вертикальный сдвиг волны

    Returns:
        np.ndarray: Синусоидный сигнал, колеблющийся между значениями (vertical_shift + amplitude) и (vertical_shift - amplitude)
    """
    return amplitude * np.sin(2 * np.pi * frequency * tp) + vertical_shift