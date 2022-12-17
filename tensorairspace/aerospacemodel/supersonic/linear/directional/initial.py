import matlab

theta = [0]
alpha = [0]
q = [0]
ele = [0]

initial_state = [[155], [155], [0], [0]]

initial_state_dict = {
    'theta': theta,
    'alpha': alpha,
    'q': q,
    'ele': ele,
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
