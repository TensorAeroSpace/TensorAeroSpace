"""Bounded trim finder for the nonlinear Boeing 737.

Same approach as the B-747 module: solve ``(\\dot u, \\dot w, \\dot q)
= 0`` at the requested level-cruise ``(altitude, V)`` for the unknowns
``(α, δ_e, δ_T)``. Commands stay within actuator limits. The success flag also checks lateral
accelerations; this longitudinal solver cannot cancel asymmetric thrust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

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


def _validate_trim_inputs(altitude, speed, guess, elevator_limit, tol):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("V_ft_s must be finite and positive")
    if not np.isfinite(altitude):
        raise ValueError("altitude_ft must be finite")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(elevator_limit) or elevator_limit <= 0:
        raise ValueError("elevator_max_rad must be finite and positive")
    if len(guess) != 3 or not np.all(np.isfinite(guess)):
        raise ValueError("initial_guess must contain three finite values")


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
    _validate_trim_inputs(
        altitude_ft, V_ft_s, initial_guess, params.elevator_max_rad, tol
    )

    def residual(z):
        alpha, de, dT = z
        x = np.zeros(12, dtype=np.float64)
        x[0] = V_ft_s * math.cos(alpha)
        x[2] = V_ft_s * math.sin(alpha)
        x[7] = alpha
        x[11] = -altitude_ft
        u = np.array([float(de), 0.0, 0.0, float(dT)], dtype=np.float64)
        f = b737_ode_6dof(x, u, 0.0, params)
        # Scale translational and angular equations to comparable magnitudes.
        return np.array([f[0], f[2], f[4] * params.cbar_ft]) / params.g_ft_s2

    lower = np.array([-np.pi / 2 + 1e-6, -params.elevator_max_rad, 0.0])
    upper = np.array([np.pi / 2 - 1e-6, params.elevator_max_rad, 1.0])
    fit = least_squares(
        residual,
        np.clip(initial_guess, lower, upper),
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    sol = fit.x
    alpha, de, dT = sol
    state = np.array(
        [
            V_ft_s * np.cos(alpha),
            0,
            V_ft_s * np.sin(alpha),
            0,
            0,
            0,
            0,
            alpha,
            0,
            0,
            0,
            -altitude_ft,
        ]
    )
    acceleration = b737_ode_6dof(state, np.array([de, 0, 0, dT]), 0.0, params)[:6]
    res_norm = float(np.linalg.norm(acceleration))
    converged = bool(fit.success and np.isfinite(res_norm) and res_norm <= tol)
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
