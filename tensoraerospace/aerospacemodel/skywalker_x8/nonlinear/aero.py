"""Aerodynamic build for the Skywalker X8 — Løw-Hansen CEAS 2025.

All coefficient values are transcribed verbatim from Table 8 of:

    Løw-Hansen, Hann, Gryte, Johansen, Deiler.
    "Modeling and identification of a small fixed-wing UAV using
    estimated aerodynamic angles".
    CEAS Aeronautical Journal (2025).
    DOI: 10.1007/s13272-025-00816-3

The model is **stability-frame parameterised** — i.e. lift and drag
are reported in the wind axis system (rotated from body by α). This
matches the CFD literature and most icing-research datasets, so the
identified coefficients can be directly compared with wind-tunnel
data on the same airframe.

Functional forms (paper Eqs. 17, 18):

    CL = CL0 + CLα·α + CLq·q* + CLδe·δe
    CD = CD0 + CDq·q* + CDCT·CT + CDk1·CL + CDk2·CL²
    CY = CY0 + CYβ·β + CYp·p* + CYr·r* + CYδa·δa
    Cl = Cl0 + Clβ·β + Clp·p* + Clr·r* + Clδa·δa
    Cm = Cm0 + Cmα·α + Cmq·q* + Cmδe·δe
    Cn = Cn0 + Cnβ·β + Cnp·p* + Cnr·r* + Cnδa·δa

Non-dimensional rates: p* = p·b/(2V), q* = q·c̄/(2V), r* = r·b/(2V).

The drag's CDCT coupling is the only non-trivial piece — it lets the
propeller's thrust influence the airframe drag (the X8's pusher prop
is mounted close to the elevon trailing edge). The thrust coefficient
``CT`` is supplied by the engine module at every ODE evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import (
    SkywalkerX8Parameters,
    isa_density_kg_m3,
    isa_speed_of_sound_m_s,
)


# Identified coefficient values — paper Table 8.

# Drag
_CD0 = 0.058
_CDq = 0.480
_CDCT = -0.217   # propeller-thrust coupling
_CDk1 = -0.034   # linear in CL
_CDk2 = 0.225    # quadratic in CL

# Lift
_CL0 = -0.077    # negative — see paper Sec. 3.5 discussion
_CLa = 2.573     # per rad
_CLq = 17.119
_CLde = 1.369

# Pitch
_Cm0 = 0.027
_Cma = -0.274
_Cmq = -1.608
_Cmde = -0.276

# Side force
_CY0 = 0.011
_CYb = -0.285
_CYp = -0.270
_CYr = 0.108
_CYda = 0.097

# Roll
_Cl0 = 0.007
_Clb = -0.108
_Clp = -0.313
_Clr = 0.037
_Clda = 0.102

# Yaw (high uncertainty in CYr, Clr, Cnp due to lack of rudder excitation)
_Cn0 = -6.3e-4
_Cnb = 0.022
_Cnp = -0.009
_Cnr = -0.050
_Cnda = -0.007


@dataclass
class AeroState:
    """Inputs to one aero evaluation (SI units)."""

    alpha: float       # rad
    beta: float        # rad
    V: float           # m/s
    p: float           # rad/s
    q: float           # rad/s
    r: float           # rad/s
    altitude_m: float
    de: float          # collective elevon (elevator), rad
    da: float          # differential elevon (aileron), rad
    CT: float = 0.0    # propeller thrust coefficient (passed by engine module)
    alphadot: float = 0.0


@dataclass
class AeroForces:
    """Stability-axis L, D, Y and body-axis l, m, n in N / N·m."""

    L: float
    D: float
    Y: float
    l: float
    m: float
    n: float


def x8_aero(state: AeroState, params: SkywalkerX8Parameters) -> AeroForces:
    """Compute body-axis forces and moments for the Skywalker X8.

    Returns dimensional values in newtons and newton-metres. Stability-
    to-body rotation by α is performed downstream in :mod:`.dynamics`,
    matching the convention of the other tensoraerospace nonlinear
    aerospaceplane modules.
    """
    rho = isa_density_kg_m3(state.altitude_m)
    V = max(state.V, 1.0)
    qbar = 0.5 * rho * V * V

    cbar = params.cbar_m
    bspan = params.b_m
    S = params.S_m2

    qhat = state.q * cbar / (2.0 * V)
    phat = state.p * bspan / (2.0 * V)
    rhat = state.r * bspan / (2.0 * V)

    # Lift (stability axis)
    C_L = _CL0 + _CLa * state.alpha + _CLq * qhat + _CLde * state.de

    # Drag — depends on CL² and prop thrust coefficient
    C_D = (
        _CD0
        + _CDq * qhat
        + _CDCT * float(state.CT)
        + _CDk1 * C_L
        + _CDk2 * C_L * C_L
    )

    # Pitching moment
    C_m = _Cm0 + _Cma * state.alpha + _Cmq * qhat + _Cmde * state.de

    # Side force
    C_Y = (
        _CY0
        + _CYb * state.beta
        + _CYp * phat
        + _CYr * rhat
        + _CYda * state.da
    )

    # Rolling moment
    C_l = (
        _Cl0
        + _Clb * state.beta
        + _Clp * phat
        + _Clr * rhat
        + _Clda * state.da
    )

    # Yawing moment (no rudder term — flying wing)
    C_n = (
        _Cn0
        + _Cnb * state.beta
        + _Cnp * phat
        + _Cnr * rhat
        + _Cnda * state.da
    )

    L = qbar * S * C_L
    D = qbar * S * C_D
    Y = qbar * S * C_Y
    l_moment = qbar * S * bspan * C_l
    m_moment = qbar * S * cbar * C_m
    n_moment = qbar * S * bspan * C_n
    return AeroForces(L=L, D=D, Y=Y, l=l_moment, m=m_moment, n=n_moment)
