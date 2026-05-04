"""State vector layout and initialisation helpers for the B-737 model.

Mirrors the B-747 layout — a 12-element rigid-body state in the
standard NED / ZYX 321 Euler convention:

    [u, v, w,           # body velocity, ft/s
     p, q, r,           # body angular rates, rad/s
     phi, theta, psi,   # Euler angles, rad
     x_e, y_e, z_e]     # NED position, ft  (z_e positive down)
"""

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
    """Twelve-element zero state."""
    return np.zeros(STATE_DIM, dtype=np.float64)


def set_initial_state(
    *,
    altitude_ft: float = 30_000.0,
    V_ft_s: float = 740.0,
    alpha_deg: float = 2.0,
    theta_deg: float | None = None,
) -> np.ndarray:
    """Build a 12-state representing level cruise at the requested point."""
    if theta_deg is None:
        theta_deg = alpha_deg
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[0] = V_ft_s * math.cos(alpha)
    x[2] = V_ft_s * math.sin(alpha)
    x[7] = theta
    x[11] = -float(altitude_ft)
    return x
