import numpy as np

def unit_step(degree:int, tp:np.array, time_step:int=10, dt:float=0.01, output_rad=False):
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