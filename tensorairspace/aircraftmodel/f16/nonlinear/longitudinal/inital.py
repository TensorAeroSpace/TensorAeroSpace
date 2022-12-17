import matlab
from numpy import deg2rad

alpha0 = deg2rad(0)
wz0 = deg2rad(0)
stab0 = deg2rad(0)
dstab0 = deg2rad(0)
initial_state = matlab.double([[alpha0], [wz0], [stab0], [dstab0]])

initial_state_dict = {
    'alpha': [alpha0],
    'wz': [wz0],
    'stab': [stab0],
    'dstab': [dstab0],
}


def set_initial_state(new_initial: dict):
    """
        Установка новых начальных параметров

    Args:
        new_initial: Словарь с новыми начальными состояниями

    Returns:
        Список новых начальные состояния

    Пример:

    >>> import numpy as np
    >>> set_initial_state({'alpha':np.deg2rad(10)})
    """
    if not set(list(new_initial.keys())).issubset(list(initial_state_dict.keys())):
        raise Exception(f"Состояния заданы неверно, проверьте. Доступные состояния {list(initial_state_dict.keys())}")

    for key, value in new_initial.items():
        initial_state_dict[key] = [value]
    return matlab.double(list(initial_state_dict.values()))
