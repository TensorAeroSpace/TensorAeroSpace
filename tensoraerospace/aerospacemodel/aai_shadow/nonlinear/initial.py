"""State vector layout and initialisation helpers (SI units)."""

from __future__ import annotations

import math

import numpy as np


STATE_LIST = [
    "u", "v", "w",
    "p", "q", "r",
    "phi", "theta", "psi",
    "x_e", "y_e", "z_e",
]
STATE_DIM = len(STATE_LIST)


def default_state() -> np.ndarray:
    return np.zeros(STATE_DIM, dtype=np.float64)


def set_initial_state(
    *,
    altitude_m: float = 1000.0,
    V_m_s: float = 36.0,
    alpha_deg: float = 4.0,
    theta_deg: float | None = None,
) -> np.ndarray:
    """Build a 12-state representing level cruise.

    Defaults: typical RQ-7 reconnaissance loiter at ~ 1 km AGL,
    cruise speed 36 m/s (~ 70 kt).
    """
    if theta_deg is None:
        theta_deg = alpha_deg
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[0] = V_m_s * math.cos(alpha)
    x[2] = V_m_s * math.sin(alpha)
    x[7] = theta
    x[11] = -float(altitude_m)
    return x
