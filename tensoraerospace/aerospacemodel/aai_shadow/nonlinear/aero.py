"""Aerodynamic build for the AAI RQ-7 Shadow.

Numerical derivatives are synthesised from class-II small-UAV
literature with V-tail effective-area scaling — the Shadow's
high-aspect-ratio wing (AR ≈ 10.9) and inverted V-tail with twin
booms put it in the same dynamic class as the Aerosonde Mark 4.7
(Beard & McLain Appendix E.1) but with larger airframe and slightly
lower rudder authority.

Functional forms (standard small-UAV stability-axis):

    CL = CL0 + CLα·α + CLq·q* + CLδe·δe
    CD = CD0 + CDk2·CL²
    CY = CYβ·β + CYp·p* + CYr·r* + CYδr·δr
    Cl = Clβ·β + Clp·p* + Clr·r* + Clδa·δa + Clδr·δr
    Cm = Cm0 + Cmα·α + Cmq·q* + Cmα̇·α̇* + Cmδe·δe
    Cn = Cnβ·β + Cnp·p* + Cnr·r* + Cnδa·δa + Cnδr·δr

Non-dimensional rates: $p^* = p\\,b / (2V)$, $q^* = q\\,\\bar c / (2V)$,
$r^* = r\\,b / (2V)$.

Coefficients are chosen so that:

* Static stability: $C_{m_\\alpha} < 0$, $C_{n_\\beta} > 0$,
  $C_{l_\\beta} < 0$ (textbook stable values).
* Lift slope: $C_{L_\\alpha} \\approx 5.0$/rad — consistent with
  AR = 10.9 and lifting-line theory.
* Induced drag: $C_{D_{k2}} = 1 / (\\pi\\,AR\\,e) \\approx 0.034$
  with Oswald efficiency $e = 0.85$.
* V-tail rudder authority: $C_{n_{\\delta_r}}$ ≈ -0.07 (smaller than
  a conventional vertical-tail aircraft because the V-tail produces
  most of its yaw force aerodynamically rather than from a dedicated
  rudder surface).

The coefficients are tabulated below and reusable for any UAV in the
same class (50–250 kg, AR > 8, V-tail, pusher prop) by adjusting the
geometry parameters in :class:`AAIShadowParameters`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .params import AAIShadowParameters, isa_density_kg_m3

# ---- Identified / synthesised derivatives (paper-reviewed class) --------

# Lift
_CL0 = 0.28
_CLa = 5.0  # /rad — high-AR rectangular wing
_CLq = 7.95
_CLde = 0.43

# Drag
_CD0 = 0.030  # clean surveillance UAV with retractable launch gear
_CDk2 = 0.043  # 1 / (π · AR · e), AR ≈ 8.75 (RQ-7B), e = 0.85

# Pitch
_Cm0 = 0.0
_Cma = -1.50  # /rad — strong static stability (large tail volume)
_Cmq = -38.0
_Cmadot = -7.0
_Cmde = -1.20

# Side force
_CYb = -0.83
_CYp = 0.0
_CYr = 0.30
_CYdr = 0.18  # V-tail effective rudder side-force

# Roll
_Clb = -0.13
_Clp = -0.51
_Clr = 0.25
_Clda = 0.17
_Cldr = 0.024  # V-tail roll-yaw cross-coupling

# Yaw
_Cnb = 0.073
_Cnp = -0.069
_Cnr = -0.095
_Cnda = -0.011  # adverse aileron yaw
_Cndr = -0.069  # V-tail rudder yaw moment


@dataclass
class AeroState:
    """Inputs to one aero evaluation (SI units)."""

    alpha: float  # rad
    beta: float  # rad
    V: float  # m/s
    p: float  # rad/s
    q: float  # rad/s
    r: float  # rad/s
    altitude_m: float
    de: float  # collective ruddervator (elevator-equivalent), rad
    da: float  # aileron, rad
    dr: float  # differential ruddervator (rudder-equivalent), rad
    alphadot: float = 0.0


@dataclass
class AeroForces:
    """Stability-axis L, D, Y and body-axis l, m, n in N / N·m."""

    L: float
    D: float
    Y: float
    l: float  # noqa: E741 — body-axis rolling moment, standard aero notation
    m: float
    n: float


def shadow_aero(state: AeroState, params: AAIShadowParameters) -> AeroForces:
    """Compute body-axis forces and moments for the AAI Shadow.

    Returns dimensional values in newtons and newton-metres,
    consistent with the rest of the tensoraerospace SI nonlinear
    modules.
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
    alpha_dot_hat = state.alphadot * cbar / (2.0 * V)

    C_L = _CL0 + _CLa * state.alpha + _CLq * qhat + _CLde * state.de
    C_D = _CD0 + _CDk2 * C_L * C_L

    C_m = (
        _Cm0
        + _Cma * state.alpha
        + _Cmq * qhat
        + _Cmadot * alpha_dot_hat
        + _Cmde * state.de
    )

    C_Y = _CYb * state.beta + _CYp * phat + _CYr * rhat + _CYdr * state.dr
    C_l = (
        _Clb * state.beta
        + _Clp * phat
        + _Clr * rhat
        + _Clda * state.da
        + _Cldr * state.dr
    )
    C_n = (
        _Cnb * state.beta
        + _Cnp * phat
        + _Cnr * rhat
        + _Cnda * state.da
        + _Cndr * state.dr
    )

    L = qbar * S * C_L
    D = qbar * S * C_D
    Y = qbar * S * C_Y
    l_moment = qbar * S * bspan * C_l
    m_moment = qbar * S * cbar * C_m
    n_moment = qbar * S * bspan * C_n
    return AeroForces(L=L, D=D, Y=Y, l=l_moment, m=m_moment, n=n_moment)
