"""Trim search for the F-16 angular nonlinear model.

Solves for level-flight equilibrium given a target true airspeed and
altitude, returning the 16-element initial state and the constant thrust
that holds it. Requires the model to be in track_altitude mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dynamics import f16_ode_6dof
from .params import F16AngularParameters


@dataclass
class TrimSolution:
    alpha_rad: float       # angle of attack at trim (= theta at level flight)
    stab_rad: float        # commanded / settled stabilator deflection
    T_thrust: float        # thrust value that holds level flight (N)
    x0: np.ndarray         # 16-element initial state ready for AngularF16
    residuals: tuple       # (dalpha, dwz, dV) at the trim point — should be ≈ 0
    converged: bool        # whether fsolve converged within tolerance


def find_trim(
    V_target: float = 120.0,
    h_target: float = 3000.0,
    params: Optional[F16AngularParameters] = None,
    *,
    alpha0: float = 0.087,        # 5°  — typical first guess
    stab0: float = -0.078,        # -4.5° — F-16 trim stab
    thrust0: float = 10000.0,
    tol: float = 1e-6,
) -> TrimSolution:
    """Find the level-flight trim for the F-16 angular model.

    Parameters
    ----------
    V_target : float
        Desired true airspeed (m/s).
    h_target : float
        Desired altitude (m).
    params : F16AngularParameters, optional
        Aircraft parameters. Uses default if None.
    alpha0, stab0, thrust0 : float
        Initial guesses for the trim search.
    tol : float
        Residual tolerance for declaring convergence.

    Returns
    -------
    TrimSolution
        Trim alpha (= theta), stab, T_thrust, plus the 16-element x0
        the AngularF16 model can be initialised with.
    """
    from scipy.optimize import fsolve

    if params is None:
        params = F16AngularParameters()

    def _residuals(unknowns: np.ndarray) -> np.ndarray:
        alpha, stab, thrust = float(unknowns[0]), float(unknowns[1]), float(unknowns[2])
        # Build a 16-state vector for f16_ode_6dof.
        x = np.zeros(16, dtype=np.float64)
        x[0]  = alpha
        x[1]  = 0.0          # beta
        x[2]  = 0.0          # wx
        x[3]  = 0.0          # wy
        x[4]  = 0.0          # wz
        x[5]  = 0.0          # gamma (roll)
        x[6]  = 0.0          # psi   (yaw)
        x[7]  = alpha        # theta = alpha at level flight
        x[8]  = stab         # current stab actuator state
        x[9]  = 0.0          # dstab
        x[10] = 0.0          # ail
        x[11] = 0.0          # dail
        x[12] = 0.0          # dir
        x[13] = 0.0          # ddir
        x[14] = h_target     # altitude
        x[15] = V_target     # true airspeed

        # Use stab as both state and command so the actuator equilibrium
        # holds (ddstab → 0 if stab_act_c == stab and dstab == 0).
        u = np.array([stab, 0.0, 0.0], dtype=np.float64)

        # Set thrust as constant in params for this evaluation.
        params.T_thrust = thrust
        params.T_active = thrust   # ensure the ODE picks the right value

        dx = f16_ode_6dof(x, u, 0.0, params)

        return np.array([dx[0], dx[4], dx[15]], dtype=np.float64)

    sol, _info, ier, _msg = fsolve(
        _residuals,
        x0=[alpha0, stab0, thrust0],
        full_output=True,
        xtol=tol,
    )
    alpha, stab, thrust = float(sol[0]), float(sol[1]), float(sol[2])
    res = _residuals(sol)
    converged = (ier == 1) and (np.max(np.abs(res)) < 1e-3)

    # Build the 16-element x0 the user can hand to AngularF16(track_altitude=True)
    x0 = np.zeros(16, dtype=np.float64)
    x0[0]  = alpha
    x0[7]  = alpha          # theta = alpha
    x0[8]  = stab
    x0[14] = h_target
    x0[15] = V_target

    # Restore params.T_thrust to the trim value
    params.T_thrust = thrust

    return TrimSolution(
        alpha_rad=alpha,
        stab_rad=stab,
        T_thrust=thrust,
        x0=x0,
        residuals=tuple(float(r) for r in res),
        converged=converged,
    )
