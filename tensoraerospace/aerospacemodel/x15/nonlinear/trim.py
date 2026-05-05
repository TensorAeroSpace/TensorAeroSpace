"""Steady-state trim finder for the nonlinear X-15.

Unlike a transport aircraft, the X-15 **does not have a level-cruise
envelope** — its rocket engine produces 30-100 % of 57 000 lbf with
no airspeed/altitude scaling. At most flight conditions the only
equilibrium is along a *climbing* (powered) or *descending*
(unpowered) trajectory.

Two trim modes are exposed:

* :func:`trim` — fixes throttle, solves for ``(α, δ_e, γ)``: the
  natural steady flight-path angle at that throttle setting. This is
  the *realistic* X-15 trim — at full throttle the aircraft climbs;
  unpowered, it glides.
* :func:`level_trim` — fixes γ = 0, solves for ``(α, δ_e, δ_T)``: the
  throttle setting required for level flight. May not be feasible
  at all (M, h) combinations; flagged via ``converged=False`` with
  the residual showing how far off equilibrium the result is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import fsolve

from .dynamics import x15_ode_6dof
from .params import X15Configuration, X15Parameters, default_parameters


@dataclass
class TrimResult:
    """Output of :func:`trim`."""

    alpha_rad: float
    elevator_rad: float
    throttle: float
    altitude_ft: float
    V_ft_s: float
    propellant_lb: float
    gamma_rad: float
    residual: float
    converged: bool
    config: X15Configuration

    @property
    def theta_rad(self) -> float:
        """Pitch angle for the trim: θ = α + γ."""
        return self.alpha_rad + self.gamma_rad

    def to_state(self) -> np.ndarray:
        """Pack the trim into a 13-state vector ready for the env."""
        V = self.V_ft_s
        a = self.alpha_rad
        theta = self.theta_rad
        x = np.zeros(13, dtype=np.float64)
        x[0] = V * math.cos(a)
        x[2] = V * math.sin(a)
        x[7] = theta
        x[11] = -float(self.altitude_ft)
        x[12] = float(self.propellant_lb)
        return x


def trim(
    altitude_ft: float,
    V_ft_s: float,
    *,
    config: X15Configuration = X15Configuration.BASIC,
    propellant_lb: Optional[float] = None,
    throttle: float = 1.0,
    initial_guess: Optional[tuple[float, float, float]] = None,
    params: Optional[X15Parameters] = None,
    tol: float = 1e-3,
) -> TrimResult:
    """Steady-flight trim at fixed throttle — solves for ``(α, δ_e, γ)``.

    This is the **realistic X-15 trim**: the rocket engine cannot
    scale its thrust to match drag at every (M, h), so we fix the
    throttle (e.g. full throttle during boost, zero throttle during
    glide) and ask "what steady flight-path angle does the aircraft
    take here?"

    Args:
        altitude_ft: Geometric altitude (ft).
        V_ft_s: True airspeed (ft/s).
        config: Configuration.
        propellant_lb: Propellant mass for the trim. Defaults to the
            configuration's full load.
        throttle: Fixed XLR99 throttle setting (0 to 1). Below 0.30
            the engine treats it as off (no thrust, no propellant
            consumption).
        initial_guess: Optional ``(α₀, δ_e₀, γ₀)`` rad seed.
        params: Optional pre-built parameters.
        tol: Convergence tolerance on the body-axis acceleration
            residual (ft/s², rad/s²).
    """
    if params is None:
        params = default_parameters(config)
    if propellant_lb is None:
        propellant_lb = params.propellant_full_lb
    if initial_guess is None:
        initial_guess = (math.radians(4.0), math.radians(-2.0), math.radians(10.0))

    def residual(z):
        alpha, de, gamma = z
        theta = alpha + gamma
        x = np.zeros(13, dtype=np.float64)
        x[0] = V_ft_s * math.cos(alpha)
        x[2] = V_ft_s * math.sin(alpha)
        x[7] = theta
        x[11] = -altitude_ft
        x[12] = float(propellant_lb)
        u = np.array([float(de), 0.0, 0.0, float(throttle)], dtype=np.float64)
        f = x15_ode_6dof(x, u, 0.0, params)
        return [f[0], f[2], f[4]]

    sol, info, ier, _ = fsolve(residual, list(initial_guess), full_output=True)
    res_vec = residual(sol)
    res_norm = float(np.linalg.norm(res_vec))
    converged = ier == 1 and res_norm <= tol
    return TrimResult(
        alpha_rad=float(sol[0]),
        elevator_rad=float(sol[1]),
        throttle=float(throttle),
        altitude_ft=float(altitude_ft),
        V_ft_s=float(V_ft_s),
        propellant_lb=float(propellant_lb),
        gamma_rad=float(sol[2]),
        residual=res_norm,
        converged=converged,
        config=config,
    )


def level_trim(
    altitude_ft: float,
    V_ft_s: float,
    *,
    config: X15Configuration = X15Configuration.BASIC,
    propellant_lb: Optional[float] = None,
    initial_guess: Optional[tuple[float, float, float]] = None,
    params: Optional[X15Parameters] = None,
    tol: float = 1e-3,
) -> TrimResult:
    """Level-flight trim — solves for ``(α, δ_e, δ_T)`` at γ = 0.

    May fail for most (M, h) — see module docstring. Use :func:`trim`
    instead when you want the natural steady flight-path angle.
    """
    if params is None:
        params = default_parameters(config)
    if propellant_lb is None:
        propellant_lb = params.propellant_full_lb
    if initial_guess is None:
        initial_guess = (math.radians(4.0), math.radians(-2.0), 0.5)

    def residual(z):
        alpha, de, dT = z
        x = np.zeros(13, dtype=np.float64)
        x[0] = V_ft_s * math.cos(alpha)
        x[2] = V_ft_s * math.sin(alpha)
        x[7] = alpha
        x[11] = -altitude_ft
        x[12] = float(propellant_lb)
        u = np.array([float(de), 0.0, 0.0, float(dT)], dtype=np.float64)
        f = x15_ode_6dof(x, u, 0.0, params)
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
        propellant_lb=float(propellant_lb),
        gamma_rad=0.0,
        residual=res_norm,
        converged=converged,
        config=config,
    )
