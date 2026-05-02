"""Default initial state for the angular F-16 (numpy version).

State ordering follows the matlab F16State_vec2struct.m convention:
[alpha, beta, wx, wy, wz, gamma, psi, theta, stab, dstab, ail, dail, dir, ddir].

Note: the legacy matlab-backed sibling at nonlinear/angular/initial.py
had a bug where set_initial_state returned a permuted vector (stab/ail/dir
were grouped before their derivatives in the dict). This module uses the
correct interleaved ordering and `set_initial_state` is pure (no module-
level mutation).
"""

from __future__ import annotations

import numpy as np
from numpy import deg2rad

_DEFAULT_STATE: dict[str, float] = {
    "alpha": deg2rad(0.0),
    "beta": deg2rad(0.0),
    "wx": deg2rad(0.0),
    "wy": deg2rad(0.0),
    "wz": deg2rad(0.0),
    "gamma": deg2rad(0.0),
    "psi": deg2rad(0.0),
    "theta": deg2rad(0.0),
    "stab": deg2rad(0.0),
    "dstab": deg2rad(0.0),
    "ail": deg2rad(0.0),
    "dail": deg2rad(0.0),
    "dir": deg2rad(0.0),
    "ddir": deg2rad(0.0),
}

_STATE_ORDER = (
    "alpha",
    "beta",
    "wx",
    "wy",
    "wz",
    "gamma",
    "psi",
    "theta",
    "stab",
    "dstab",
    "ail",
    "dail",
    "dir",
    "ddir",
)

initial_state: np.ndarray = np.array(
    [[_DEFAULT_STATE[name]] for name in _STATE_ORDER],
    dtype=np.float64,
)

initial_state_dict: dict[str, list[float]] = {
    name: [_DEFAULT_STATE[name]] for name in _STATE_ORDER
}


def set_initial_state(new_initial: dict) -> np.ndarray:
    """Pure: build a new column vector with given overrides applied."""
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
