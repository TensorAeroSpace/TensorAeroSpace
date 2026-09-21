"""Fixed-step ODE integrators for F-16 nonlinear models.

RHS signature: ``f(x, u, t, params) -> dx``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

RHS = Callable[[np.ndarray, np.ndarray, float, Any], np.ndarray]


def euler(
    f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any
) -> np.ndarray:
    """Return one explicit Euler step for ``f(x, u, t, params)`` with constant input."""
    return np.asarray(x + dt * f(x, u, t, params))


def rk4(
    f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float, params: Any
) -> np.ndarray:
    """Return one classical fourth-order Runge-Kutta step with input held constant."""
    k1 = f(x, u, t, params)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt, params)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt, params)
    k4 = f(x + dt * k3, u, t + dt, params)
    return np.asarray(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def integrate_events(
    advance: Callable[[np.ndarray, float, float], np.ndarray],
    x: np.ndarray,
    t: float,
    dt: float,
    events: Sequence[tuple[float, Callable[[], None]]] = (),
) -> np.ndarray:
    """Split a step at (offset_seconds, callback) discontinuities.

    Callbacks run after the segment ending at their offset. Even its final RK4
    stage therefore uses pre-event dynamics. The caller records one outer step.
    """
    ordered = sorted(events, key=lambda item: item[0])
    for offset, callback in ordered:
        if not np.isfinite(offset) or not 0 <= offset <= dt:
            raise ValueError("event offset must be finite and within the time step")
        if not callable(callback):
            raise TypeError("event callback must be callable")
    elapsed = 0.0
    for offset, callback in ordered:
        if offset > elapsed:
            x = advance(x, t + elapsed, offset - elapsed)
        callback()
        elapsed = offset
    if elapsed < dt:
        x = advance(x, t + elapsed, dt - elapsed)
    return x
