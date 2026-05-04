"""Aerodynamic force and moment computation for the B-747 nonlinear model.

The build is *Taylor-expansion around trim*: each non-dimensional
coefficient is the trimmed value at the operating Mach/altitude plus
linear contributions from (α − α₀, β, control deflections, body rates,
α̇, M − M₀). This is exactly the formulation that Stevens-Lewis Ch. 3
and CR-2144 itself use for handling-qualities analysis.

For the cruise envelope (h ∈ [SL, 40 kft], M ∈ [0.45, 0.90]) the
trimmed derivatives are bilinearly interpolated over the published
grid; for the landing and power-approach configurations the explicit
Tables IX-1 / IX-2 are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .derivatives import (
    LateralDirectionalDerivatives,
    LongitudinalDerivatives,
    cruise_lateral_at,
    cruise_longitudinal_at,
    get_lateral,
    get_longitudinal,
)
from .flight_conditions import B747FlightCondition, get_flight_condition
from .params import (
    B747Configuration,
    B747Parameters,
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
)


@dataclass
class AeroState:
    """Inputs to the aerodynamic build (one ODE evaluation)."""

    alpha: float       # angle of attack, rad
    beta: float        # sideslip, rad
    V: float           # true airspeed, ft/s
    p: float           # body roll rate, rad/s
    q: float           # body pitch rate, rad/s
    r: float           # body yaw rate, rad/s
    altitude_ft: float
    de: float          # elevator deflection, rad
    da: float          # aileron deflection, rad
    dr: float          # rudder deflection, rad
    alphadot: float = 0.0


@dataclass
class AeroForces:
    """Stability-axis forces & moments from one aero evaluation."""

    L: float           # lift, lb (acts +Z stability)
    D: float           # drag, lb (acts +X stability, *backward*)
    Y: float           # side force, lb (body axis)
    l: float           # rolling moment, lb·ft (body axis)
    m: float           # pitching moment, lb·ft (body axis)
    n: float           # yawing moment, lb·ft (body axis)


def _select_trim_point(
    altitude_ft: float, mach: float, config: B747Configuration
) -> tuple[B747FlightCondition, LongitudinalDerivatives, LateralDirectionalDerivatives]:
    """Pick a trim FC and a derivative bank for the current operating point.

    For LANDING / POWER_APPROACH we pin to FC1 / FC2 (their respective
    explicit tables). For NOMINAL (cruise) we bilinear-interpolate the
    8-point cruise grid at the current (h, M); the trim FC used for
    α₀ and dynamic-pressure book-keeping is the nearest one in the
    same altitude band.
    """
    if config is B747Configuration.LANDING:
        fc = get_flight_condition(1)
        return fc, get_longitudinal(1), get_lateral(1)
    if config is B747Configuration.POWER_APPROACH:
        fc = get_flight_condition(2)
        return fc, get_longitudinal(2), get_lateral(2)
    # cruise: pick the closest published FC in (h, M)
    best, best_d = 3, float("inf")
    for fc_id in range(3, 11):
        cand = get_flight_condition(fc_id)
        d = abs(cand.altitude_ft - altitude_ft) / 40000.0 + abs(cand.mach - mach)
        if d < best_d:
            best, best_d = fc_id, d
    fc = get_flight_condition(best)
    return (
        fc,
        cruise_longitudinal_at(altitude_ft, mach),
        cruise_lateral_at(altitude_ft, mach),
    )


def b747_aero(state: AeroState, params: B747Parameters) -> AeroForces:
    """Compute body-axis forces and moments from the current aero state.

    Returns dimensional values in pounds and pound-feet. The ODE
    converts these to body-axis components via the standard
    body↔stability rotation by α (β is small, so the small-angle
    approximation cos β ≈ 1 is used in this initial release).

    If ``params.damage_state.flap_jam_config`` is set, the aerodynamic
    derivatives are sourced from the *jammed* configuration instead of
    ``params.config`` — modelling high-lift devices stuck at one
    position.
    """
    rho = isa_density_slug_ft3(state.altitude_ft)
    a = isa_speed_of_sound_ft_s(state.altitude_ft)
    V = max(state.V, 1.0)  # avoid divide-by-zero at hover-like speeds
    M = V / a
    qbar = 0.5 * rho * V * V

    # Flap-jam override: damage state can pin the aero config.
    effective_config = params.config
    damage_state = getattr(params, "damage_state", None)
    if damage_state is not None:
        jammed = getattr(damage_state, "flap_jam_config", None)
        if jammed is not None:
            effective_config = jammed
    fc, lon, lat = _select_trim_point(state.altitude_ft, M, effective_config)

    alpha0 = np.deg2rad(fc.alpha0_deg)
    da_pert = state.alpha - alpha0
    M0 = fc.mach
    dM = M - M0

    cbar = params.cbar_ft
    bspan = params.b_ft

    # Non-dimensional rate factors
    qhat = state.q * cbar / (2.0 * V)
    phat = state.p * bspan / (2.0 * V)
    rhat = state.r * bspan / (2.0 * V)
    alpha_dot_hat = state.alphadot * cbar / (2.0 * V)

    # Longitudinal coefficients (stability axis)
    C_L = (
        lon.C_L0
        + lon.C_La * da_pert
        + lon.C_Lq * qhat
        + lon.C_Ladot * alpha_dot_hat
        + lon.C_LM * dM
        + lon.C_Lde * state.de
    )
    C_D = (
        lon.C_D0
        + lon.C_Da * da_pert
        + lon.C_DM * dM
    )
    C_m = (
        lon.C_ma * da_pert
        + lon.C_mq * qhat
        + lon.C_madot * alpha_dot_hat
        + lon.C_mM * dM
        + lon.C_mde * state.de
    )

    # Lateral-directional coefficients (stability axis for β-driven, body axis for rate-driven)
    C_Y = (
        lat.C_Yb * state.beta
        + lat.C_Yp * phat
        + lat.C_Yr * rhat
        + lat.C_Yda * state.da
        + lat.C_Ydr * state.dr
    )
    C_l = (
        lat.C_lb * state.beta
        + lat.C_lp * phat
        + lat.C_lr * rhat
        + lat.C_lda * state.da
        + lat.C_ldr * state.dr
    )
    C_n = (
        lat.C_nb * state.beta
        + lat.C_np * phat
        + lat.C_nr * rhat
        + lat.C_nda * state.da
        + lat.C_ndr * state.dr
    )

    S = params.S_ft2
    L = qbar * S * C_L
    D = qbar * S * C_D
    Y = qbar * S * C_Y
    l_moment = qbar * S * bspan * C_l
    m_moment = qbar * S * cbar * C_m
    n_moment = qbar * S * bspan * C_n
    return AeroForces(L=L, D=D, Y=Y, l=l_moment, m=m_moment, n=n_moment)
