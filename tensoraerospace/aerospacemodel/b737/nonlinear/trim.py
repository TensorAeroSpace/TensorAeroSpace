"""Newton-Raphson trim finder for the nonlinear Boeing 737.

Same approach as the B-747 module: solve ``(\\dot u, \\dot w, \\dot q)
= 0`` at the requested level-cruise ``(altitude, V)`` for the unknowns
``(α, δ_e, δ_T)``. Unlike the X-15, the 737 has air-breathing engines
that scale with Mach and altitude, so cruise trim *does* converge
across the operational envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from .dynamics import b737_ode_6dof
from .params import B737Configuration, B737Parameters, default_parameters


@dataclass
class TrimResult:
    alpha_rad: float
    elevator_rad: float
    throttle: float
    altitude_ft: float
    V_ft_s: float
    residual: float
    converged: bool
    config: B737Configuration

    def to_state(self) -> np.ndarray:
        V = self.V_ft_s
        a = self.alpha_rad
        x = np.zeros(12, dtype=np.float64)
        x[0] = V * math.cos(a)
        x[2] = V * math.sin(a)
        x[7] = a
        x[11] = -float(self.altitude_ft)
        return x


def trim(
    altitude_ft: float,
    V_ft_s: float,
    *,
    config: B737Configuration = B737Configuration.B737_100,
    initial_guess: Optional[tuple[float, float, float]] = None,
    params: Optional[B737Parameters] = None,
    tol: float = 1e-3,
) -> TrimResult:
    """Find ``(α, δ_e, δ_T)`` for steady level flight."""
    if params is None:
        params = default_parameters(config)
    if initial_guess is None:
        initial_guess = (math.radians(2.0), math.radians(-1.0), 0.5)

    def residual(z):
        alpha, de, dT = z
        x = np.zeros(12, dtype=np.float64)
        x[0] = V_ft_s * math.cos(alpha)
        x[2] = V_ft_s * math.sin(alpha)
        x[7] = alpha
        x[11] = -altitude_ft
        u = np.array([float(de), 0.0, 0.0, float(dT)], dtype=np.float64)
        f = b737_ode_6dof(x, u, 0.0, params)
        return [f[0], f[2], f[4]]

    sol, info, ier, _ = fsolve(residual, list(initial_guess), full_output=True)
    res_vec = residual(sol)
    res_norm = float(np.linalg.norm(res_vec))
    converged = ier == 1 and res_norm <= tol and 0.0 <= float(sol[2]) <= 1.0
    return TrimResult(
        alpha_rad=float(sol[0]),
        elevator_rad=float(sol[1]),
        throttle=float(sol[2]),
        altitude_ft=float(altitude_ft),
        V_ft_s=float(V_ft_s),
        residual=res_norm,
        converged=converged,
        config=config,
    )
