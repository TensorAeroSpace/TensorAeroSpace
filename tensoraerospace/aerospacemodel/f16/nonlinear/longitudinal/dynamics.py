"""ODE right-hand side for the F-16 longitudinal model.

Direct line-by-line port of longitudinal/matlab_code/F16ODE.m.
State: [alpha, wz, stab, dstab]. Control: [stab_act].
"""

from __future__ import annotations

import numpy as np

from .aero import get_cy, get_mz
from .params import F16LongParameters


def f16_ode_long(
    x: np.ndarray, u: np.ndarray, t: float, params: F16LongParameters
) -> np.ndarray:
    alpha, wz, stab, dstab = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    stab_act = float(u[0])
    p = params

    cy = get_cy(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)
    mz = get_mz(alpha, 0.0, stab, p.lef, wz, p.V, p.bA, p.sb)

    # ----------------------------------------------------------------
    # Apply damage corrections (no-op if damage_state is None).
    # In this model, mz is the PITCHING moment coefficient (not yaw),
    # so we add delta_my from aero_corrections (which uses standard
    # convention M_y = pitch).
    # ----------------------------------------------------------------
    damage_state = p.damage_state
    damage_geo = p.damage_geometry
    if damage_state is not None and damage_geo is not None:
        from ..damage import aero_corrections as _ac
        cy = cy + _ac.delta_cy(alpha, 0.0, damage_geo, damage_state)
        mz = mz + _ac.delta_my(alpha, 0.0, damage_geo, damage_state)

    Y = p.q * p.S * cy
    Mz = p.q * p.S * p.bA * mz

    Ry = Y
    MRz = Mz + p.rcgx * Ry

    dwz = MRz / p.Jz
    dalpha = wz - (Ry - p.m * p.g) / (p.m * p.V)

    dstab_clip = float(np.clip(dstab, -p.maxabsdstab, p.maxabsdstab))
    stab_act_clip = float(np.clip(stab_act, -p.maxabsstab, p.maxabsstab))
    ddstab = (-2.0 * p.Tstab * p.Xistab * dstab - stab + stab_act_clip) / (p.Tstab**2)

    return np.array([dalpha, dwz, dstab_clip, ddstab], dtype=np.float64)
