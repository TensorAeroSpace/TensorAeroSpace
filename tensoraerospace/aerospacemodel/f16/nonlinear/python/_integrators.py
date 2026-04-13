"""Fixed-step ODE integrators for F-16 nonlinear models.

RHS signature: ``f(x, u, t, params) -> dx``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

RHS = Callable[[np.ndarray, np.ndarray, float, Any], np.ndarray]


def euler(
    f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any
) -> np.ndarray:
    return x + dt * f(x, u, t, params)


def rk4(
    f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any
) -> np.ndarray:
    k1 = f(x, u, t, params)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, params)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, params)
    k4 = f(x + dt * k3, u, t + dt, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
