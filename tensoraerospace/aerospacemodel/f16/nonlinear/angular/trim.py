"""Trim search for the F-16 angular nonlinear model.

Solves for level-flight equilibrium given a target true airspeed and
altitude, returning the 16-element initial state and the constant thrust
that holds it. Requires the model to be in track_altitude mode.

The unknowns are angle of attack, stabilator deflection, and thrust.
All three residuals (angle-of-attack rate, pitch acceleration, and speed
acceleration) must vanish. Thrust must lie within engine capability and
stabilator deflection within its physical limits. An infeasible solution
is returned with ``converged=False``; thrust is never floored to hide a
nonzero speed acceleration.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dynamics import f16_ode_6dof
from .params import F16AngularParameters


@dataclass
class TrimSolution:
    alpha_rad: float  # angle of attack at trim (= theta at level flight)
    stab_rad: float  # commanded / settled stabilator deflection
    T_thrust: float  # thrust value that holds level flight (N)
    x0: np.ndarray  # 16-element initial state ready for AngularF16
    residuals: tuple  # (dalpha, dwz, dV) at the trim point — ≈ 0
    converged: bool  # whether fsolve converged within tolerance


def find_trim(
    V_target: float = 120.0,
    h_target: float = 3000.0,
    params: Optional[F16AngularParameters] = None,
    *,
    alpha0: float = 0.087,  # 5°  — typical first guess
    stab0: float = -0.078,  # -4.5° — F-16 trim stab
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
    alpha0, stab0 : float
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

    if not np.isfinite(V_target) or V_target <= 0:
        raise ValueError("V_target must be positive and finite")
    if not np.isfinite(h_target) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("h_target must be finite and tol positive and finite")
    # Solver iterations must not modify the caller's live plant parameters.
    params = copy(params) if params is not None else F16AngularParameters()

    def _state(alpha: float, stab: float) -> np.ndarray:
        x = np.zeros(16, dtype=np.float64)
        x[0] = x[7] = alpha
        x[8] = stab
        x[14:] = [h_target, V_target]
        return x

    def _residuals(unknowns: np.ndarray) -> np.ndarray:
        alpha, stab, thrust = map(float, unknowns)
        params.T_thrust = params.T_active = thrust
        dx = f16_ode_6dof(_state(alpha, stab), np.array([stab, 0.0, 0.0]), 0.0, params)
        return dx[[0, 4, 15]]

    solution, _info, ier, _msg = fsolve(
        _residuals,
        x0=[alpha0, stab0, params.T_thrust],
        full_output=True,
        xtol=tol,
    )
    alpha, stab, thrust = map(float, solution)
    residuals = _residuals(solution)
    converged = bool(
        ier == 1
        and np.all(np.isfinite(solution))
        and np.max(np.abs(residuals)) <= tol
        and 0.0 <= thrust <= params.T_max_thrust
        and abs(stab) <= params.maxabsstab
    )
    return TrimSolution(
        alpha_rad=alpha,
        stab_rad=stab,
        T_thrust=thrust,
        x0=_state(alpha, stab),
        residuals=tuple(map(float, residuals)),
        converged=converged,
    )
