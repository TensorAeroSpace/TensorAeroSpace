"""ODE integrators for the X-15 nonlinear model."""

from __future__ import annotations

from typing import Callable

import numpy as np


def euler(
    f: Callable, x: np.ndarray, u: np.ndarray, t: float, dt: float, params
) -> np.ndarray:
    """First-order explicit Euler step."""
    return x + dt * f(x, u, t, params)


def rk4(
    f: Callable, x: np.ndarray, u: np.ndarray, t: float, dt: float, params
) -> np.ndarray:
    """Classical 4-stage Runge-Kutta step.

    Recommended for any flight-mechanics use; the X-15 has fast
    short-period and Dutch-roll modes (especially at hypersonic Mach)
    that overstate amplitudes under Euler integration with dt > 0.02 s.
    """
    k1 = f(x, u, t, params)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, params)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, params)
    k4 = f(x + dt * k3, u, t + dt, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
