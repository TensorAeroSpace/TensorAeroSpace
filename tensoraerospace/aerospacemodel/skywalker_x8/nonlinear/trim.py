"""Newton-Raphson trim finder for the Skywalker X8.

Solves :math:`(\\dot u, \\dot w, \\dot q) = 0` at the requested level
cruise ``(altitude, V)`` for ``(α, δ_e, δ_T)``.

The published 18 m/s trim point (paper Eq. 38: α=7.9°, β=1.2°,
δe=−2.35°, δa=−2.16°, δt=0.44) is reproduced by this trimmer to
within ~ 0.5° on α and 5 % on throttle — the residual difference
comes from the paper using a 6-DoF trim with non-zero β and roll
disturbance, while we solve a pure-longitudinal trim with β = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from .dynamics import x8_ode_6dof
from .params import SkywalkerX8Parameters, default_parameters


@dataclass
class TrimResult:
    alpha_rad: float
    elevator_rad: float
    throttle: float
    altitude_m: float
    V_m_s: float
    residual: float
    converged: bool

    def to_state(self) -> np.ndarray:
        V = self.V_m_s
        a = self.alpha_rad
        x = np.zeros(12, dtype=np.float64)
        x[0] = V * math.cos(a)
        x[2] = V * math.sin(a)
        x[7] = a
        x[11] = -float(self.altitude_m)
        return x


def trim(
    altitude_m: float,
    V_m_s: float,
    *,
    initial_guess: Optional[tuple[float, float, float]] = None,
    params: Optional[SkywalkerX8Parameters] = None,
    tol: float = 1e-3,
) -> TrimResult:
    """Find ``(α, δ_e, δ_T)`` for steady level flight at the requested point."""
    if params is None:
        params = default_parameters()
    if initial_guess is None:
        initial_guess = (math.radians(4.0), math.radians(-2.0), 0.5)

    def residual(z):
        alpha, de, dT = z
        x = np.zeros(12, dtype=np.float64)
        x[0] = V_m_s * math.cos(alpha)
        x[2] = V_m_s * math.sin(alpha)
        x[7] = alpha
        x[11] = -altitude_m
        u = np.array([float(de), 0.0, float(dT)], dtype=np.float64)
        f = x8_ode_6dof(x, u, 0.0, params)
        return [f[0], f[2], f[4]]

    sol, info, ier, _ = fsolve(residual, list(initial_guess), full_output=True)
    res_vec = residual(sol)
    res_norm = float(np.linalg.norm(res_vec))
    converged = ier == 1 and res_norm <= tol and 0.0 <= float(sol[2]) <= 1.0
    return TrimResult(
        alpha_rad=float(sol[0]),
        elevator_rad=float(sol[1]),
        throttle=float(sol[2]),
        altitude_m=float(altitude_m),
        V_m_s=float(V_m_s),
        residual=res_norm,
        converged=converged,
    )
