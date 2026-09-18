"""Six-degree-of-freedom, still-air level-flight trim for the Skywalker X8.

Solve all body accelerations and angular accelerations simultaneously. Nonzero
side-force and moment offsets require sideslip, bank and differential elevon.
The published wind-disturbed flight point is not a still-air trim reference.
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
    beta_rad: float = 0.0
    roll_rad: float = 0.0
    aileron_rad: float = 0.0

    def to_state(self) -> np.ndarray:
        """Return the balanced 12-state vector with zero vertical velocity."""
        V, a, b, phi = self.V_m_s, self.alpha_rad, self.beta_rad, self.roll_rad
        x = np.zeros(12, dtype=np.float64)
        x[:3] = V * np.array(
            [math.cos(a) * math.cos(b), math.sin(b), math.sin(a) * math.cos(b)]
        )
        x[6] = phi
        # NED z_dot = -u*sin(theta) + (v*sin(phi)+w*cos(phi))*cos(theta).
        x[7] = math.atan2(x[1] * math.sin(phi) + x[2] * math.cos(phi), x[0])
        x[11] = -float(self.altitude_m)
        return x

    def to_control(self) -> np.ndarray:
        """Return ``[elevator, aileron, throttle]`` for this equilibrium."""
        return np.array([self.elevator_rad, self.aileron_rad, self.throttle])


def trim(
    altitude_m: float,
    V_m_s: float,
    *,
    initial_guess: Optional[tuple[float, float, float]] = None,
    params: Optional[SkywalkerX8Parameters] = None,
    tol: float = 1e-3,
) -> TrimResult:
    """Find a full equilibrium with physically admissible elevons and throttle.

    ``initial_guess`` retains its ``(alpha, elevator, throttle)`` convention;
    sideslip, roll and aileron start at zero. ``converged`` includes the residual
    of all six dynamic equations and both individual elevon travel limits.
    """
    if not math.isfinite(V_m_s) or V_m_s <= 0:
        raise ValueError("V_m_s must be finite and positive")
    if not math.isfinite(altitude_m):
        raise ValueError("altitude_m must be finite")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if params is None:
        params = default_parameters()
    if initial_guess is None:
        initial_guess = (math.radians(4.0), math.radians(-2.0), 0.5)
    if len(initial_guess) != 3 or not np.all(np.isfinite(initial_guess)):
        raise ValueError("initial_guess must contain three finite values")

    def make_result(z) -> TrimResult:
        alpha, beta, phi, de, da, dT = z
        return TrimResult(
            alpha_rad=float(alpha),
            elevator_rad=float(de),
            throttle=float(dT),
            altitude_m=float(altitude_m),
            V_m_s=float(V_m_s),
            residual=math.inf,
            converged=False,
            beta_rad=float(beta),
            roll_rad=float(phi),
            aileron_rad=float(da),
        )

    def residual(z):
        result = make_result(z)
        return x8_ode_6dof(result.to_state(), result.to_control(), 0.0, params)[:6]

    alpha, de, dT = initial_guess
    sol, _, ier, _ = fsolve(residual, [alpha, 0.0, 0.0, de, 0.0, dT], full_output=True)
    result = make_result(sol)
    result.residual = float(np.linalg.norm(residual(sol)))
    elevons = [
        result.elevator_rad + result.aileron_rad,
        result.elevator_rad - result.aileron_rad,
    ]
    result.converged = bool(
        ier == 1
        and np.all(np.isfinite(sol))
        and result.residual <= tol
        and 0.0 <= result.throttle <= 1.0
        and max(abs(v) for v in elevons) <= params.elevon_max_rad
        and max(abs(result.alpha_rad), abs(result.beta_rad), abs(result.roll_rad))
        < math.pi / 2
    )
    return result
