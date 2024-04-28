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
