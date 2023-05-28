import matlab

phi = [0]
theta = [0]
psi = [0]
velocity = [92]
alpha = [0]
beta = [0]
p = [0]
q = [0]
r = [0]
ele = [0]
ail = [0]
rud = [0]

initial_state = [phi, theta, psi, alpha, beta, p, q, r, ele, ail, rud]

initial_state_dict = {
    'phi': phi,
    'theta': theta,
    'psi': psi,
    'alpha': alpha,
    'beta': beta,
    'p': p,
    'q': q,
    'r': r,
    'ele': ele,
    'ail': ail,
    'rud': rud,

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
