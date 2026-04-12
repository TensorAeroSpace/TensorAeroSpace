"""Default initial state for the longitudinal F-16 (numpy version)."""
from __future__ import annotations

import numpy as np
from numpy import deg2rad

alpha0 = deg2rad(0.0)
wz0 = deg2rad(0.0)
stab0 = deg2rad(0.0)
dstab0 = deg2rad(0.0)

_DEFAULT_STATE: dict[str, float] = {
    "alpha": alpha0,
    "wz": wz0,
    "stab": stab0,
    "dstab": dstab0,
}

_STATE_ORDER = ("alpha", "wz", "stab", "dstab")

initial_state: np.ndarray = np.array(
    [[_DEFAULT_STATE[name]] for name in _STATE_ORDER],
    dtype=np.float64,
)

# Kept for backward compatibility with code that imports `initial_state_dict`
# directly. Do not mutate this — `set_initial_state` builds its result from
# `_DEFAULT_STATE` independently to avoid cross-call contamination.
initial_state_dict: dict[str, list[float]] = {
    name: [_DEFAULT_STATE[name]] for name in _STATE_ORDER
}


def set_initial_state(new_initial: dict) -> np.ndarray:
    """Build a new initial state column vector with given overrides applied.

    Pure: does not mutate any module-level state. Each call starts from
    `_DEFAULT_STATE`, so calling `set_initial_state({"alpha": 0.1})` followed
    by `set_initial_state({"wz": 0.2})` returns vectors with ONLY the
    respective override applied.
    """
    unknown = set(new_initial) - set(_STATE_ORDER)
    if unknown:
        raise ValueError(
            "Состояния заданы неверно, проверьте."
            f" Доступные состояния {list(_STATE_ORDER)}; неизвестные: {sorted(unknown)}"
        )
    values = {name: float(_DEFAULT_STATE[name]) for name in _STATE_ORDER}
    for key, value in new_initial.items():
        values[key] = float(value)
    return np.array(
        [[values[name]] for name in _STATE_ORDER],
        dtype=np.float64,
    )
