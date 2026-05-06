"""Newton-Raphson trim-finder for the B-747 nonlinear 6-DoF model.

Given (altitude, true airspeed, configuration), solve for the steady
level-flight trim:

    α — angle of attack, rad
    δ_e — elevator deflection, rad
    δ_T — throttle setting in [0, 1]

such that, in body-axis NED frame with γ = 0 (level flight) and
β = p = q = r = φ = ψ = 0:

    u̇ = 0     (no longitudinal acceleration)
    ẇ = 0     (no vertical acceleration ⇒ L = W·cosθ)
    q̇ = 0     (zero pitching moment)

The solution is the same set of three equations Stevens-Lewis Ch. 3 §3.7
solves analytically; here we just call :func:`scipy.optimize.fsolve` on
the model's ODE — it converges in a few Newton steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from .dynamics import b747_ode_6dof
from .params import B747Configuration, B747Parameters, default_parameters


@dataclass(frozen=True)
class TrimResult:
    """Trimmed flight condition returned by :func:`trim`.

    All angles in radians, throttle in [0, 1], airspeed in ft/s, altitude
    in ft. ``residual`` is the L2 norm of the (u̇, ẇ, q̇) at convergence
    — should be ≪ 1 in well-behaved cases.
    """

    altitude_ft: float
    V_ft_s: float
    alpha_rad: float
    elevator_rad: float
    throttle: float
    theta_rad: float
    residual: float
    converged: bool

    def to_state(self, *, psi: float = 0.0) -> np.ndarray:
        """Pack the trim into a 12-element state vector ready for the model."""
        u_b = self.V_ft_s * np.cos(self.alpha_rad)
        w_b = self.V_ft_s * np.sin(self.alpha_rad)
        return np.array(
            [
                u_b,
                0.0,
                w_b,  # u, v, w
                0.0,
                0.0,
                0.0,  # p, q, r
                0.0,
                self.theta_rad,
                psi,  # phi, theta, psi
                0.0,
                0.0,
                -self.altitude_ft,  # x_e, y_e, z_e
            ],
            dtype=np.float64,
        )


def trim(
    altitude_ft: float,
    V_ft_s: float,
    *,
    config: B747Configuration = B747Configuration.NOMINAL,
    initial_guess: tuple[float, float, float] = (0.05, 0.0, 0.4),
    params: B747Parameters | None = None,
    tol: float = 1e-6,
) -> TrimResult:
    """Solve for steady level-flight trim at (altitude, airspeed).

    Args:
        altitude_ft: Altitude (ft).
        V_ft_s: Target true airspeed (ft/s).
        config: Aerodynamic configuration to trim in.
        initial_guess: ``(α, δ_e, δ_T)`` seed for the solver.
        params: Override parameters (defaults to ``default_parameters(config)``).
        tol: Convergence tolerance for the residual norm.

    Returns:
        :class:`TrimResult` with the trimmed angles and throttle.
    """
    p = params if params is not None else default_parameters(config)

    def residual(z: np.ndarray) -> np.ndarray:
        alpha, de, dT = float(z[0]), float(z[1]), float(np.clip(z[2], 0.0, 1.0))
        # In level flight γ = 0 ⇒ θ = α
        u_b = V_ft_s * np.cos(alpha)
        w_b = V_ft_s * np.sin(alpha)
        x = np.array(
            [
                u_b,
                0.0,
                w_b,
                0.0,
                0.0,
                0.0,
                0.0,
                alpha,
                0.0,
                0.0,
                0.0,
                -altitude_ft,
            ],
            dtype=np.float64,
        )
        u = np.array([de, 0.0, 0.0, dT], dtype=np.float64)
        dx = b747_ode_6dof(x, u, 0.0, p)
        # Three equations: u̇, ẇ, q̇  (indices 0, 2, 4)
        return np.array([dx[0], dx[2], dx[4]], dtype=np.float64)

    z0 = np.array(initial_guess, dtype=np.float64)
    z, info, ier, msg = fsolve(residual, z0, full_output=True, xtol=tol)
    res_norm = float(np.linalg.norm(info["fvec"]))
    converged = (ier == 1) and (res_norm < 1e-3)
    alpha, de, dT = float(z[0]), float(z[1]), float(np.clip(z[2], 0.0, 1.0))
    return TrimResult(
        altitude_ft=altitude_ft,
        V_ft_s=V_ft_s,
        alpha_rad=alpha,
        elevator_rad=de,
        throttle=dT,
        theta_rad=alpha,  # γ = 0 ⇒ θ = α
        residual=res_norm,
        converged=converged,
    )
