import matlab
from numpy import deg2rad

alpha0 = deg2rad(0)
beta0 = deg2rad(0)
wx0 = deg2rad(0)
wy0 = deg2rad(0)
wz0 = deg2rad(0)
gamma0 = deg2rad(0)
psi0 = deg2rad(0)
theta0 = deg2rad(0)
stab0 = deg2rad(0)
ail0 = deg2rad(0)
dir0 = deg2rad(0)
dstab0 = deg2rad(0)
dail0 = deg2rad(0)
ddir0 = deg2rad(0)

initial_state = matlab.double(
    [[alpha0], [beta0], [wx0], [wy0], [wz0], [gamma0], [psi0], [theta0], [stab0], [dstab0], [ail0], [dail0], [dir0],
     [ddir0]])

initial_state_dict = {
    'alpha': [alpha0],
    'beta': [beta0],
    'wx': [wx0],
    'wy': [wy0],
    'wz': [wz0],
    'gamma': [gamma0],
    'psi': [psi0],
    'theta': [theta0],
    'stab': [stab0],
    'ail': [ail0],
    'dir': [dir0],
    'dstab': [dstab0],
    'dail': [dail0],
    'ddir': [ddir0],
}


def set_initial_state(new_initial: dict):
    """
        Установка новых начальных параметров

    Args:
        new_initial: Словарь с новыми начальными состояниями

    Returns:
        Новые начальные состояния

    Пример:

    >>> import numpy as np
    >>> set_initial_state({'alpha':np.deg2rad(10), 'beta':np.deg2rad(1)})
    """
    if not set(list(new_initial.keys())).issubset(list(initial_state_dict.keys())):
        raise Exception(f"Состояния заданы неверно, проверьте. Доступные состояния {list(initial_state_dict.keys())}")

    for key, value in new_initial.items():
        initial_state_dict[key] = [value]
    return matlab.double(list(initial_state_dict.values()))
