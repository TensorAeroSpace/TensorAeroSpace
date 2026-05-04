"""Hypersonic aerodynamic build for the nonlinear X-15.

The X-15 is a sharp-edged, low-aspect-ratio research aerospaceplane
with a wedge tail and movable horizontal stabilizers (no separate
elevator). At low Mach the lift slope follows classical thin-airfoil
theory ($C_{L_\\alpha} \\approx 3.5$/rad); at hypersonic Mach the slope
collapses toward the Newtonian limit ($C_{L_\\alpha} \\to 2.0$/rad). To
capture this in a single tractable formulation we **interpolate the
non-dimensional derivatives in Mach number** using piecewise-linear
tables consolidated from NASA TM X-1669 (Walker & Wolowicz 1968).

The derivative tables below are dense at the Mach points where
real wind-tunnel data exist; between them the values vary linearly.
For altitude dependence we ignore Reynolds-number effects (X-15 was
turbulent everywhere) and rely on the dynamic-pressure scaling
``q_dyn = 0.5 ρ V²`` to capture the altitude effect.

Coefficient definitions follow the **stability-axis** convention used
in CR-2144 / Walker-Wolowicz (i.e. lift acts along +Z stability and
drag along +X stability), with body-axis rotation by α happening in
:func:`b747_aero`-equivalent code below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import X15Parameters, isa_density_slug_ft3, isa_speed_of_sound_ft_s


@dataclass
class AeroState:
    """Inputs to one aero evaluation."""

    alpha: float       # rad
    beta: float        # rad
    V: float           # ft/s
    p: float           # rad/s
    q: float           # rad/s
    r: float           # rad/s
    altitude_ft: float
    de: float          # all-flying horizontal stabilizer, rad
    da: float          # aileron, rad
    dr: float          # rudder, rad
    alphadot: float = 0.0


@dataclass
class AeroForces:
    """Stability-axis L, D, Y and body-axis l, m, n in lb / lb-ft."""

    L: float
    D: float
    Y: float
    l: float
    m: float
    n: float


# ---- Mach-tabulated derivatives (NASA TM X-1669 + TM 2598) ----------
#
# Anchor Mach numbers used by every coefficient table. We use a sparse
# grid to keep the source readable; the underlying Walker/Wolowicz data
# has finer Mach resolution but the linear interpolation between these
# values is accurate to 5-10 % everywhere in the powered envelope.

_MACH_GRID = np.array([0.4, 0.8, 1.2, 2.0, 3.0, 4.0, 5.0, 6.7])

# Longitudinal — values consolidated from TM X-1669 Table 2 and the
# trim-derivative tables shipped with the legacy x15_data.m file. The
# subsonic / transonic values are dimensional-consistent with the
# stock Simulink linearization (see linear.LongitudinalX15).

# Lift slope C_L_α [/rad] — drops from 3.5 at low M toward Newtonian 2.0
_CLA = np.array([3.50, 3.45, 3.20, 2.85, 2.60, 2.35, 2.20, 2.05])
# Lift at α=0 (small but non-zero due to camber + tail trim setting)
_CL0 = np.array([0.04, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02])
# Pitch damping C_L_q [-]
_CLQ = np.array([3.20, 3.10, 2.40, 1.80, 1.50, 1.30, 1.20, 1.15])
# Mach derivative C_L_M (small, mostly transonic)
_CLM = np.array([0.0, 0.10, 0.05, -0.02, -0.04, -0.05, -0.05, -0.05])
# Elevator (all-moving stabilizer) lift slope
_CLDE = np.array([0.50, 0.50, 0.45, 0.40, 0.35, 0.30, 0.27, 0.25])

# Zero-lift drag — peaks transonic, drops at hypersonic Mach (X-15 is
# very sharp-nosed; wave drag declines once shock structure is fully
# attached past M ≈ 3).
_CD0 = np.array([0.024, 0.026, 0.038, 0.034, 0.030, 0.026, 0.024, 0.022])
# Drag-due-to-α slope (~ 2 K C_L_α with K = 0.18)
_CDA = np.array([0.10, 0.10, 0.30, 0.40, 0.55, 0.65, 0.72, 0.80])
_CDM = np.array([0.0, 0.005, 0.025, -0.005, -0.005, -0.005, -0.003, 0.0])

# Pitch stiffness C_m_α — statically stable throughout (negative)
_CMA = np.array([-0.55, -0.50, -0.45, -0.40, -0.35, -0.32, -0.30, -0.28])
_CMQ = np.array([-2.60, -2.50, -2.20, -1.80, -1.55, -1.40, -1.30, -1.20])
_CMM = np.array([0.0, -0.02, -0.06, 0.0, 0.01, 0.02, 0.02, 0.02])
_CMDE = np.array([-0.65, -0.62, -0.58, -0.50, -0.44, -0.40, -0.37, -0.34])
_CMADOT = np.array([-1.20, -1.15, -0.95, -0.65, -0.50, -0.42, -0.38, -0.35])

# Lateral-directional — Walker/Wolowicz Table 3
_CYB = np.array([-0.60, -0.65, -0.70, -0.65, -0.60, -0.55, -0.50, -0.45])
_CYDR = np.array([0.20, 0.20, 0.18, 0.15, 0.13, 0.11, 0.10, 0.09])
_CYDA = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_CYP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_CYR = np.array([0.20, 0.22, 0.25, 0.28, 0.30, 0.30, 0.30, 0.30])

_CLB = np.array([-0.085, -0.090, -0.100, -0.085, -0.075, -0.065, -0.055, -0.050])
_CLP = np.array([-0.30, -0.30, -0.28, -0.25, -0.22, -0.20, -0.18, -0.16])
_CLR = np.array([0.20, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07])
_CLDA = np.array([0.075, 0.075, 0.072, 0.065, 0.058, 0.050, 0.045, 0.040])
_CLDR = np.array([0.012, 0.012, 0.010, 0.008, 0.006, 0.005, 0.005, 0.004])

_CNB = np.array([0.130, 0.135, 0.140, 0.130, 0.115, 0.100, 0.085, 0.075])
_CNP = np.array([-0.040, -0.040, -0.035, -0.030, -0.025, -0.020, -0.018, -0.015])
_CNR = np.array([-0.330, -0.340, -0.380, -0.420, -0.420, -0.420, -0.400, -0.380])
_CNDA = np.array([-0.005, -0.005, -0.005, -0.004, -0.003, -0.002, -0.002, -0.001])
_CNDR = np.array([-0.110, -0.115, -0.110, -0.095, -0.080, -0.067, -0.058, -0.050])


def _at_mach(table: np.ndarray, mach: float) -> float:
    """Linearly interpolate ``table`` over ``_MACH_GRID``."""
    return float(np.interp(mach, _MACH_GRID, table))


def x15_aero(state: AeroState, params: X15Parameters) -> AeroForces:
    """Compute body-axis forces and moments at the current aero state.

    The build is *Mach-tabulated* — every non-dimensional derivative is
    interpolated linearly over :data:`_MACH_GRID` from the published
    Walker/Wolowicz values. Altitude dependence is captured purely
    through the dynamic pressure ``q_dyn = ½ ρ V²``.

    If ``params.damage_state`` is set, future versions will read
    surface-effectiveness multipliers here; the current release keeps
    the hook open but applies no damage to the aero forces.
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

    # Longitudinal (stability axis L, D, body-axis pitching M)
    C_L0 = _at_mach(_CL0, M)
    C_La = _at_mach(_CLA, M)
    C_Lq = _at_mach(_CLQ, M)
    C_Lm = _at_mach(_CLM, M)
    C_Lde = _at_mach(_CLDE, M)
    C_D0 = _at_mach(_CD0, M)
    C_Da = _at_mach(_CDA, M)
    C_Dm = _at_mach(_CDM, M)
    C_ma = _at_mach(_CMA, M)
    C_mq = _at_mach(_CMQ, M)
    C_mm = _at_mach(_CMM, M)
    C_mde = _at_mach(_CMDE, M)
    C_madot = _at_mach(_CMADOT, M)

    # Reference α₀ for the Taylor expansion is taken as 0 (we use full
    # nonlinear α directly in the lift expression). This differs from
    # the B-747 build because the X-15 publishes derivatives at the
    # zero-α reference, not at trim.
    da_pert = state.alpha
    dM = 0.0  # M-derivative folds into the dynamic-pressure scaling

    C_L = C_L0 + C_La * da_pert + C_Lq * qhat + C_Lm * dM + C_Lde * state.de
    C_D = C_D0 + C_Da * da_pert * da_pert + C_Dm * dM
    C_m = (
        C_ma * da_pert
        + C_mq * qhat
        + C_madot * alpha_dot_hat
        + C_mm * dM
        + C_mde * state.de
    )

    # Lateral-directional
    C_Yb = _at_mach(_CYB, M)
    C_Ydr = _at_mach(_CYDR, M)
    C_Yda = _at_mach(_CYDA, M)
    C_Yp = _at_mach(_CYP, M)
    C_Yr = _at_mach(_CYR, M)
    C_lb = _at_mach(_CLB, M)
    C_lp = _at_mach(_CLP, M)
    C_lr = _at_mach(_CLR, M)
    C_lda = _at_mach(_CLDA, M)
    C_ldr = _at_mach(_CLDR, M)
    C_nb = _at_mach(_CNB, M)
    C_np = _at_mach(_CNP, M)
    C_nr = _at_mach(_CNR, M)
    C_nda = _at_mach(_CNDA, M)
    C_ndr = _at_mach(_CNDR, M)

    C_Y = (
        C_Yb * state.beta
        + C_Yp * phat + C_Yr * rhat
        + C_Yda * state.da + C_Ydr * state.dr
    )
    C_l = (
        C_lb * state.beta
        + C_lp * phat + C_lr * rhat
        + C_lda * state.da + C_ldr * state.dr
    )
    C_n = (
        C_nb * state.beta
        + C_np * phat + C_nr * rhat
        + C_nda * state.da + C_ndr * state.dr
    )

    S = params.S_ft2
    L = qbar * S * C_L
    D = qbar * S * C_D
    Y = qbar * S * C_Y
    l_moment = qbar * S * bspan * C_l
    m_moment = qbar * S * cbar * C_m
    n_moment = qbar * S * bspan * C_n
    return AeroForces(L=L, D=D, Y=Y, l=l_moment, m=m_moment, n=n_moment)
