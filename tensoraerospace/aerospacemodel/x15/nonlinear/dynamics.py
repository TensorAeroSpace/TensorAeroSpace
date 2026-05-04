"""Newton-Euler 6-DoF body-axis ODE for the X-15 hypersonic model.

State (13-dim, NED, ZYX 321 Euler):

    x[0]  =  u    body x-velocity, ft/s
    x[1]  =  v    body y-velocity, ft/s
    x[2]  =  w    body z-velocity, ft/s
    x[3]  =  p    body roll rate, rad/s
    x[4]  =  q    body pitch rate, rad/s
    x[5]  =  r    body yaw rate, rad/s
    x[6]  =  phi    Euler bank, rad
    x[7]  =  theta  Euler pitch, rad
    x[8]  =  psi    Euler heading, rad
    x[9]  =  x_e    NED north position, ft
    x[10] =  y_e    NED east position, ft
    x[11] =  z_e    NED z (positive down), ft  ⇒ altitude = -z_e
    x[12] =  m_prop  remaining propellant, lb (variable mass channel)

Control (4-dim, all in radians except throttle):

    u[0] = δ_e    horizontal stabilizer (all-flying), rad
    u[1] = δ_a    aileron, rad
    u[2] = δ_r    rudder, rad
    u[3] = δ_T    XLR99 throttle [0, 1] (0 = off below 30 % cutoff)

Variable-mass effect: the rocket equation form ``m·v̇ = ΣF + (v_e -
v) ṁ_e`` reduces to the ordinary ``m·v̇ = ΣF + T_x x̂`` because the
exhaust thrust ``T = ṁ_e v_e`` is already accounted for in the
thrust force, and the *velocity-of-mass-loss* term vanishes for an
on-axis exhaust (a standard simplification used by every textbook
rocket-aircraft model). Mass and inertias are then updated each
step from the propellant decrement.
"""

from __future__ import annotations

import numpy as np

from .aero import AeroState, x15_aero
from .params import X15Parameters, isa_speed_of_sound_ft_s


def x15_ode_6dof(
    x: np.ndarray, u: np.ndarray, t: float, params: X15Parameters
) -> np.ndarray:
    """13-D Newton-Euler RHS for the nonlinear X-15."""
    u_b, v_b, w_b = float(x[0]), float(x[1]), float(x[2])
    p, q, r = float(x[3]), float(x[4]), float(x[5])
    phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
    z_e = float(x[11])
    m_prop = max(0.0, float(x[12]))
    altitude_ft = -z_e
    de, da, dr, dT = float(u[0]), float(u[1]), float(u[2]), float(u[3])

    V = float(np.sqrt(u_b * u_b + v_b * v_b + w_b * w_b))
    V_safe = max(V, 1.0)
    alpha = float(np.arctan2(w_b, u_b))
    beta = float(np.arcsin(np.clip(v_b / V_safe, -1.0, 1.0)))

    # Aerodynamic forces & moments — body axis components
    aero = x15_aero(
        AeroState(
            alpha=alpha, beta=beta, V=V_safe,
            p=p, q=q, r=r,
            altitude_ft=altitude_ft,
            de=de, da=da, dr=dr,
        ),
        params,
    )

    sa, ca = np.sin(alpha), np.cos(alpha)
    X_aero = -aero.D * ca + aero.L * sa
    Z_aero = -aero.D * sa - aero.L * ca
    Y_aero = aero.Y

    # XLR99 thrust — constant w.r.t. Mach/altitude, modulated by throttle
    # and propellant availability.
    from .engine import xlr99_thrust

    T_eng, mdot_lb_s = xlr99_thrust(dT, m_prop, params)

    # Variable-mass body — current mass and inertias from propellant
    mass = params.current_mass_slug(m_prop)
    g = params.g_ft_s2

    # Body-axis gravity
    g_x = -g * np.sin(theta)
    g_y = g * np.cos(theta) * np.sin(phi)
    g_z = g * np.cos(theta) * np.cos(phi)

    # Translational dynamics
    du = (X_aero + T_eng) / mass + g_x - (q * w_b - r * v_b)
    dv = Y_aero / mass + g_y - (r * u_b - p * w_b)
    dw = Z_aero / mass + g_z - (p * v_b - q * u_b)

    # Rotational dynamics with I_xz cross-coupling
    Ix, Iy, Iz, Ixz = params.inertia_at(m_prop)
    L_moment = aero.l
    M_moment = aero.m
    N_moment = aero.n

    Gamma = Ix * Iz - Ixz * Ixz
    L_bar = L_moment + Ixz * (p * q) - (Iz - Iy) * q * r
    N_bar = N_moment - Ixz * (q * r) - (Iy - Ix) * p * q
    dp = (Iz * L_bar + Ixz * N_bar) / Gamma
    dr_dot = (Ixz * L_bar + Ix * N_bar) / Gamma
    dq = (M_moment - (Ix - Iz) * p * r - Ixz * (p * p - r * r)) / Iy

    # Euler kinematics (ZYX 321)
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    tth = sth / max(cth, 1e-9)
    dphi = p + (q * sphi + r * cphi) * tth
    dtheta = q * cphi - r * sphi
    dpsi = (q * sphi + r * cphi) / max(cth, 1e-9)

    # Earth-fixed (NED) position rate from body-axis velocity
    DCM = np.array([
        [cth * np.cos(psi),
         sphi * sth * np.cos(psi) - cphi * np.sin(psi),
         cphi * sth * np.cos(psi) + sphi * np.sin(psi)],
        [cth * np.sin(psi),
         sphi * sth * np.sin(psi) + cphi * np.cos(psi),
         cphi * sth * np.sin(psi) - sphi * np.cos(psi)],
        [-sth, sphi * cth, cphi * cth],
    ])
    pos_dot = DCM @ np.array([u_b, v_b, w_b])

    # Mass loss — propellant decrement is *negative* time derivative
    # because mdot_lb_s is reported as a positive magnitude.
    dm_prop = -float(mdot_lb_s)

    return np.array([
        du, dv, dw,
        dp, dq, dr_dot,
        dphi, dtheta, dpsi,
        pos_dot[0], pos_dot[1], pos_dot[2],
        dm_prop,
    ], dtype=np.float64)
