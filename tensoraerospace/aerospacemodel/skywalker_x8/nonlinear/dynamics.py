"""Newton-Euler 6-DoF body-axis ODE for the Skywalker X8 (SI units).

12-D state — same layout as the B-747 / B-737 modules:

    x[0]  = u    body x-velocity, m/s
    x[1]  = v    body y-velocity, m/s
    x[2]  = w    body z-velocity, m/s
    x[3]  = p    roll rate, rad/s
    x[4]  = q    pitch rate, rad/s
    x[5]  = r    yaw rate, rad/s
    x[6]  = phi    bank, rad
    x[7]  = theta  pitch, rad
    x[8]  = psi    heading, rad
    x[9]  = x_e    NED north position, m
    x[10] = y_e    NED east position, m
    x[11] = z_e    NED z (positive down), m  ⇒  altitude = -z_e

3-D control — note the X8 has **no rudder** (flying wing):

    u[0] = δ_e collective elevon (elevator), rad
    u[1] = δ_a differential elevon (aileron), rad
    u[2] = δ_T throttle [0, 1]
"""

from __future__ import annotations

import numpy as np

from .aero import AeroState, x8_aero
from .params import SkywalkerX8Parameters


def x8_ode_6dof(
    x: np.ndarray, u: np.ndarray, t: float, params: SkywalkerX8Parameters
) -> np.ndarray:
    u_b, v_b, w_b = float(x[0]), float(x[1]), float(x[2])
    p, q, r = float(x[3]), float(x[4]), float(x[5])
    phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
    z_e = float(x[11])
    altitude_m = -z_e
    de, da, dT = float(u[0]), float(u[1]), float(u[2])

    V = float(np.sqrt(u_b * u_b + v_b * v_b + w_b * w_b))
    V_safe = max(V, 1.0)
    alpha = float(np.arctan2(w_b, u_b))
    beta = float(np.arcsin(np.clip(v_b / V_safe, -1.0, 1.0)))

    # Engine first — its CT feeds into the aero drag (CDCT coupling)
    from .engine import x8_thrust

    T_eng, CT_val = x8_thrust(dT, V_safe, altitude_m, params)

    aero = x8_aero(
        AeroState(
            alpha=alpha, beta=beta, V=V_safe,
            p=p, q=q, r=r,
            altitude_m=altitude_m,
            de=de, da=da,
            CT=CT_val,
        ),
        params,
    )

    sa, ca = np.sin(alpha), np.cos(alpha)
    X_aero = -aero.D * ca + aero.L * sa
    Z_aero = -aero.D * sa - aero.L * ca
    Y_aero = aero.Y

    mass = params.mass_kg
    g = params.g_m_s2
    g_x = -g * np.sin(theta)
    g_y = g * np.cos(theta) * np.sin(phi)
    g_z = g * np.cos(theta) * np.cos(phi)

    du = (X_aero + T_eng) / mass + g_x - (q * w_b - r * v_b)
    dv = Y_aero / mass + g_y - (r * u_b - p * w_b)
    dw = Z_aero / mass + g_z - (p * v_b - q * u_b)

    Ix, Iy, Iz, Ixz = params.Ix, params.Iy, params.Iz, params.Ixz
    L_moment = aero.l
    M_moment = aero.m
    N_moment = aero.n

    Gamma = Ix * Iz - Ixz * Ixz
    L_bar = L_moment + Ixz * (p * q) - (Iz - Iy) * q * r
    N_bar = N_moment - Ixz * (q * r) - (Iy - Ix) * p * q
    dp = (Iz * L_bar + Ixz * N_bar) / Gamma
    dr_dot = (Ixz * L_bar + Ix * N_bar) / Gamma
    dq = (M_moment - (Ix - Iz) * p * r - Ixz * (p * p - r * r)) / Iy

    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    tth = sth / max(cth, 1e-9)
    dphi = p + (q * sphi + r * cphi) * tth
    dtheta = q * cphi - r * sphi
    dpsi = (q * sphi + r * cphi) / max(cth, 1e-9)

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

    return np.array([
        du, dv, dw,
        dp, dq, dr_dot,
        dphi, dtheta, dpsi,
        pos_dot[0], pos_dot[1], pos_dot[2],
    ], dtype=np.float64)
