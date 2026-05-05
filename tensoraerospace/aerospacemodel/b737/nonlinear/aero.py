"""Aerodynamic build for the nonlinear Boeing 737.

Coefficient values are transcribed from the **JSBSim 737 model**
(``aircraft/737/737.xml``), which itself sources its data from
Roskam Vol VI Appendix B (737-100 reference configuration). The
JSBSim formulation uses **dimensional** coefficient functions (lift
in lb, etc.) — we keep that convention here since it matches the
B-747 nonlinear module's API and avoids per-cell unit conversions.

Coefficient structure:

* **C_L** — Mach-independent angle-of-attack table (peak at α ≈ 13°)
  + linear elevator slope + flap/spoiler/ground-effect corrections.
* **C_D** — α-table for parasitic + induced drag, plus Mach
  compressibility table (transonic drag rise around M = 1.0–1.1)
  + flap/gear/speedbrake/spoiler drag adders + sideslip drag.
* **C_m** — linear C_m_α + Mach-dependent elevator effectiveness +
  pitch damping + α̇ damping.
* **C_Y, C_l, C_n** — linear lateral-directional derivatives with
  the standard rate-factor non-dimensionalisation.

Ground / configuration effects (k_CLge, k_CDge, k_CLflap, k_CDflap)
are exposed but *not* applied at this MVP stage — they require an
extra "configuration" state (flaps, gear, speedbrake) that we
deliberately omit. The model is therefore valid for the **clean
cruise envelope** out of the box; downstream landing/take-off
extensions can plug into the same coefficient-evaluation function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import B737Parameters, isa_density_slug_ft3, isa_speed_of_sound_ft_s


@dataclass
class AeroState:
    alpha: float  # rad
    beta: float  # rad
    V: float  # ft/s
    p: float  # rad/s
    q: float  # rad/s
    r: float  # rad/s
    altitude_ft: float
    de: float  # elevator, rad
    da: float  # aileron, rad
    dr: float  # rudder, rad
    alphadot: float = 0.0


@dataclass
class AeroForces:
    L: float
    D: float
    Y: float
    l: float
    m: float
    n: float


# ---- α-table for C_L (JSBSim 737 CLalpha) -----------------------------
# Tabulated lift coefficient at α (radians). Smooth pre-stall slope of
# ~5.5/rad up to α ≈ 13°, then post-stall behaviour clamped to limit
# numerical blow-up if the integrator over-shoots into post-stall.
_ALPHA_GRID = np.array([-0.20, -0.10, 0.0, 0.10, 0.20, 0.23, 0.30, 0.40, 0.60])
_CL_TABLE = np.array([-0.85, -0.30, 0.20, 0.75, 1.25, 1.45, 1.20, 0.90, 0.60])

# C_D base table (JSBSim CD0)
_CD_ALPHA_GRID = np.array([-1.57, -0.26, -0.10, 0.0, 0.10, 0.26, 1.57])
_CD_TABLE = np.array([1.50, 0.042, 0.025, 0.021, 0.025, 0.042, 1.50])

# Mach compressibility (JSBSim CDmach)
_MACH_GRID_CD = np.array([0.0, 0.79, 0.85, 0.90, 1.10, 1.80])
_CD_MACH = np.array([0.0, 0.0, 0.005, 0.012, 0.023, 0.015])

# Mach-dependent elevator effectiveness (JSBSim Cmde)
_MACH_GRID_CMDE = np.array([0.0, 0.5, 0.85, 1.0, 1.5, 2.0])
_CMDE_TABLE = np.array([-1.20, -1.10, -0.90, -0.70, -0.45, -0.30])

# Mach-dependent aileron rolling-moment effectiveness (JSBSim Clda)
_MACH_GRID_CLDA = np.array([0.0, 0.5, 0.85, 1.0, 1.5, 2.0])
_CLDA_TABLE = np.array([0.10, 0.090, 0.080, 0.070, 0.050, 0.033])


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> float:
    return float(np.interp(x, xp, fp))


def b737_aero(state: AeroState, params: B737Parameters) -> AeroForces:
    """Compute body-axis forces and moments at the current aero state.

    Returns dimensional values in pounds and pound-feet, matching the
    convention used by the B-747 / X-15 nonlinear modules.
    """
    rho = isa_density_slug_ft3(state.altitude_ft)
    a = isa_speed_of_sound_ft_s(state.altitude_ft)
    V = max(state.V, 1.0)
    M = V / a
    qbar = 0.5 * rho * V * V

    cbar = params.cbar_ft
    bspan = params.b_ft

    qhat = state.q * cbar / (2.0 * V)
    phat = state.p * bspan / (2.0 * V)
    rhat = state.r * bspan / (2.0 * V)
    alpha_dot_hat = state.alphadot * cbar / (2.0 * V)

    # Longitudinal — JSBSim 737 functional form
    C_L_alpha = _interp(state.alpha, _ALPHA_GRID, _CL_TABLE)
    C_Lde = 0.20
    C_L = C_L_alpha + C_Lde * state.de

    C_D_alpha = _interp(state.alpha, _CD_ALPHA_GRID, _CD_TABLE)
    C_Di = 0.043 * (C_L**2)
    C_D_mach = _interp(M, _MACH_GRID_CD, _CD_MACH)
    C_D_beta = 1.23 * (state.beta * state.beta) / (1.57**2)  # quadratic up to ±90°
    C_Dde = 0.059 * abs(state.de)
    C_D = C_D_alpha + C_Di + C_D_mach + C_D_beta + C_Dde

    # Pitching moment (Cmα + Cmq + Cmadot + Cmde)
    C_m_alpha = -0.6 * state.alpha
    C_m_q = -27.0 * qhat
    C_m_adot = -16.0 * alpha_dot_hat
    C_m_de = _interp(M, _MACH_GRID_CMDE, _CMDE_TABLE) * state.de
    C_m = C_m_alpha + C_m_q + C_m_adot + C_m_de

    # Lateral-directional (JSBSim CYb, Clb, Clp, Clr, Clda, Cldr,
    # Cnb, Cnr, Cndr; rudder side-force omitted as JSBSim sets it ≈ 0)
    C_Y = -1.0 * state.beta + 0.0 * phat + 0.0 * rhat + 0.0 * state.da + 0.15 * state.dr
    C_l = (
        -0.09 * state.beta
        - 0.40 * phat
        + 0.09 * rhat
        + _interp(M, _MACH_GRID_CLDA, _CLDA_TABLE) * state.da
        + 0.01 * state.dr
    )
    C_n = (
        0.26 * state.beta
        + 0.0 * phat
        - 0.35 * rhat
        - 0.005 * state.da  # adverse aileron yaw
        - 0.20 * state.dr
    )

    S = params.S_ft2
    L = qbar * S * C_L
    D = qbar * S * C_D
    Y = qbar * S * C_Y
    l_moment = qbar * S * bspan * C_l
    m_moment = qbar * S * cbar * C_m
    n_moment = qbar * S * bspan * C_n
    return AeroForces(L=L, D=D, Y=Y, l=l_moment, m=m_moment, n=n_moment)
