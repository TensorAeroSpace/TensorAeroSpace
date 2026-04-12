"""Default initial state for the longitudinal F-16 (numpy version)."""
from __future__ import annotations

import numpy as np
from numpy import deg2rad

alpha0 = deg2rad(0.0)
wz0 = deg2rad(0.0)
stab0 = deg2rad(0.0)
dstab0 = deg2rad(0.0)

initial_state: np.ndarray = np.array(
    [[alpha0], [wz0], [stab0], [dstab0]],
    dtype=np.float64,
)

_STATE_ORDER = ("alpha", "wz", "stab", "dstab")
initial_state_dict: dict = {
    "alpha": [alpha0],
    "wz": [wz0],
    "stab": [stab0],
    "dstab": [dstab0],
}


def set_initial_state(new_initial: dict) -> np.ndarray:
    """Override one or more initial states. Returns the new column vector."""
    unknown = set(new_initial) - set(_STATE_ORDER)
    if unknown:
        raise Exception(
            "Состояния заданы неверно, проверьте."
            f" Доступные состояния {list(_STATE_ORDER)}"
        )
    for key, value in new_initial.items():
        initial_state_dict[key] = [float(value)]
    return np.array(
        [initial_state_dict[name] for name in _STATE_ORDER],
        dtype=np.float64,
    )
