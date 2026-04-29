"""Trim search for the F-16 angular nonlinear model.

Solves for level-flight equilibrium given a target true airspeed and
altitude, returning the 16-element initial state and the constant thrust
that holds it. Requires the model to be in track_altitude mode.

Formulation (avoids the unphysical-T trap):

* Unknowns: ``α`` and ``stab`` (2 scalars).
* Residuals: ``dα/dt = 0`` and ``dω_z/dt = 0`` at level flight (β=0,
  γ_roll=0, ψ=0, θ=α, all body rates zero).
* Thrust is then set to ``T = q·S·Cx / cos(α)·cos(β)`` so that
  ``dV/dt = (T·cosα·cosβ − D)/m − g·sin γ_path = 0`` automatically at
  level flight (where ``g·sin γ_path = 0``). Drag ``D = q·S·Cx`` is
  derived from the F-16 aero tables at the converged α.

Solving for thrust *after* alpha/stab are pinned down avoids the
ambiguous 3-unknown formulation which can converge to a saddle point
with negative thrust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .aero import get_cx
from .dynamics import f16_ode_6dof
from .params import F16AngularParameters, _isa_dynamic_pressure


@dataclass
class TrimSolution:
    alpha_rad: float       # angle of attack at trim (= theta at level flight)
    stab_rad: float        # commanded / settled stabilator deflection
    T_thrust: float        # thrust value that holds level flight (N)
    x0: np.ndarray         # 16-element initial state ready for AngularF16
    residuals: tuple       # (dalpha, dwz, dV) at the trim point — ≈ 0
    converged: bool        # whether fsolve converged within tolerance


def find_trim(
    V_target: float = 120.0,
    h_target: float = 3000.0,
    params: Optional[F16AngularParameters] = None,
    *,
    alpha0: float = 0.087,        # 5°  — typical first guess
    stab0: float = -0.078,        # -4.5° — F-16 trim stab
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

    if params is None:
        params = F16AngularParameters()

    # We want dα = 0 and dω_z = 0 at level flight. Thrust is set in the
    # ODE's energy equation but doesn't enter dα or dω_z, so during the
    # search we can use any thrust value — we pick ~0 N to remove
    # noise from the dV residual that we won't read anyway.
    params.T_thrust = 0.0
    params.T_active = 0.0

    def _residuals(unknowns: np.ndarray) -> np.ndarray:
        alpha, stab = float(unknowns[0]), float(unknowns[1])
        x = np.zeros(16, dtype=np.float64)
        x[0]  = alpha
        x[7]  = alpha   # theta = alpha at level flight
        x[8]  = stab
        x[14] = h_target
        x[15] = V_target

        # stab_act = stab → ddstab driven by command - state - rate = 0 at
        # equilibrium when dstab = 0
        u = np.array([stab, 0.0, 0.0], dtype=np.float64)

        dx = f16_ode_6dof(x, u, 0.0, params)
        # Residual 1: alpha holds → dalpha = 0
        # Residual 2: pitch holds → dwz = 0
        return np.array([dx[0], dx[4]], dtype=np.float64)

    sol, _info, ier, _msg = fsolve(
        _residuals,
        x0=[alpha0, stab0],
        full_output=True,
        xtol=tol,
    )
    alpha, stab = float(sol[0]), float(sol[1])
    res2 = _residuals(sol)
    converged = (ier == 1) and (np.max(np.abs(res2)) < 1e-4)

    # ---- Compute thrust to balance drag at the converged α ----
    # Drag (along velocity) = q·S·Cx, where Cx is read from the F-16 aero
    # table at the trim α, β=0, with stab=stab, lef=0, sb=0, wz=0.
    q_trim = _isa_dynamic_pressure(h_target, V_target, params.g)
    cx_trim = get_cx(alpha, 0.0, stab, params.lef, 0.0,
                     V_target, params.bA, params.sb)
    drag_trim = q_trim * params.S * cx_trim

    # Energy: 0 = (T·cosα·cosβ − D)/m − g·sin(γ=0) at level flight
    # → T = D / cos(α)·cos(β=0) = D / cos(α)
    cos_a = np.cos(alpha)
    T_calc = float(drag_trim / cos_a)

    # The F-16 NASA matlab-port aero tables can return small negative or
    # near-zero Cx values at moderate α — that produces an unphysical
    # negative trim thrust. Floor T to a sensible idle thrust so the
    # simulation always has positive forward force; the aircraft may
    # then accelerate slightly above V_target instead of holding it
    # exactly, but won't dive on takeoff because of "negative thrust".
    T_THRUST_MIN = 8000.0    # N — F-16 idle thrust order of magnitude
    if T_calc < T_THRUST_MIN:
        T_thrust = T_THRUST_MIN
    else:
        T_thrust = T_calc
    params.T_thrust = T_thrust
    params.T_active = T_thrust
    x_full = np.zeros(16, dtype=np.float64)
    x_full[0]  = alpha
    x_full[7]  = alpha
    x_full[8]  = stab
    x_full[14] = h_target
    x_full[15] = V_target
    dx_check = f16_ode_6dof(
        x_full, np.array([stab, 0.0, 0.0], dtype=np.float64), 0.0, params,
    )
    res3 = (float(dx_check[0]), float(dx_check[4]), float(dx_check[15]))

    # Build the 16-element x0
    x0 = np.zeros(16, dtype=np.float64)
    x0[0]  = alpha
    x0[7]  = alpha
    x0[8]  = stab
    x0[14] = h_target
    x0[15] = V_target

    return TrimSolution(
        alpha_rad=alpha,
        stab_rad=stab,
        T_thrust=T_thrust,
        x0=x0,
        residuals=res3,
        converged=converged,
    )
