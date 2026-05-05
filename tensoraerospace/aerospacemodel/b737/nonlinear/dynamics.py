"""Newton-Euler 6-DoF body-axis ODE for the nonlinear Boeing 737.

Same 12-D state layout as the B-747 model:

    x[0]  = u   body x-velocity, ft/s
    x[1]  = v   body y-velocity, ft/s
    x[2]  = w   body z-velocity, ft/s
    x[3]  = p   body roll rate, rad/s
    x[4]  = q   body pitch rate, rad/s
    x[5]  = r   body yaw rate, rad/s
    x[6]  = phi    Euler bank, rad
    x[7]  = theta  Euler pitch, rad
    x[8]  = psi    Euler heading, rad
    x[9]  = x_e    NED north position, ft
    x[10] = y_e    NED east position, ft
    x[11] = z_e    NED z (positive down), ft

Control:

    u[0] = δ_e elevator, rad
    u[1] = δ_a aileron, rad
    u[2] = δ_r rudder, rad
    u[3] = δ_T throttle [0, 1]
"""

from __future__ import annotations

import numpy as np

from .aero import AeroState, b737_aero
from .params import B737Parameters, isa_speed_of_sound_ft_s


def b737_ode_6dof(
    x: np.ndarray, u: np.ndarray, t: float, params: B737Parameters
) -> np.ndarray:
    u_b, v_b, w_b = float(x[0]), float(x[1]), float(x[2])
    p, q, r = float(x[3]), float(x[4]), float(x[5])
    phi, theta, psi = float(x[6]), float(x[7]), float(x[8])
    z_e = float(x[11])
    altitude_ft = -z_e
    de, da, dr, dT = float(u[0]), float(u[1]), float(u[2]), float(u[3])

    V = float(np.sqrt(u_b * u_b + v_b * v_b + w_b * w_b))
    V_safe = max(V, 1.0)
    alpha = float(np.arctan2(w_b, u_b))
    beta = float(np.arcsin(np.clip(v_b / V_safe, -1.0, 1.0)))

    aero = b737_aero(
        AeroState(
            alpha=alpha,
            beta=beta,
            V=V_safe,
            p=p,
            q=q,
            r=r,
            altitude_ft=altitude_ft,
            de=de,
            da=da,
            dr=dr,
        ),
        params,
    )

    sa, ca = np.sin(alpha), np.cos(alpha)
    X_aero = -aero.D * ca + aero.L * sa
    Z_aero = -aero.D * sa - aero.L * ca
    Y_aero = aero.Y

    # Engine — JT8D or CFM56 cluster + asymmetric-thrust yaw moment
    from .engine import b737_thrust_with_asymmetry

    M_local = V_safe / isa_speed_of_sound_ft_s(altitude_ft)
    T_eng, N_eng = b737_thrust_with_asymmetry(dT, M_local, altitude_ft, params)

    mass = params.mass_slug
    g = params.g_ft_s2
    g_x = -g * np.sin(theta)
    g_y = g * np.cos(theta) * np.sin(phi)
    g_z = g * np.cos(theta) * np.cos(phi)

    du = (X_aero + T_eng) / mass + g_x - (q * w_b - r * v_b)
    dv = Y_aero / mass + g_y - (r * u_b - p * w_b)
    dw = Z_aero / mass + g_z - (p * v_b - q * u_b)

    Ix, Iy, Iz, Ixz = params.Ix, params.Iy, params.Iz, params.Ixz
    L_moment = aero.l
    M_moment = aero.m
    N_moment_total = aero.n + N_eng

    Gamma = Ix * Iz - Ixz * Ixz
    L_bar = L_moment + Ixz * (p * q) - (Iz - Iy) * q * r
    N_bar = N_moment_total - Ixz * (q * r) - (Iy - Ix) * p * q
    dp = (Iz * L_bar + Ixz * N_bar) / Gamma
    dr_dot = (Ixz * L_bar + Ix * N_bar) / Gamma
    dq = (M_moment - (Ix - Iz) * p * r - Ixz * (p * p - r * r)) / Iy

    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    tth = sth / max(cth, 1e-9)
    dphi = p + (q * sphi + r * cphi) * tth
    dtheta = q * cphi - r * sphi
    dpsi = (q * sphi + r * cphi) / max(cth, 1e-9)

    DCM = np.array(
        [
            [
                cth * np.cos(psi),
                sphi * sth * np.cos(psi) - cphi * np.sin(psi),
                cphi * sth * np.cos(psi) + sphi * np.sin(psi),
            ],
            [
                cth * np.sin(psi),
                sphi * sth * np.sin(psi) + cphi * np.cos(psi),
                cphi * sth * np.sin(psi) - sphi * np.cos(psi),
            ],
            [-sth, sphi * cth, cphi * cth],
        ]
    )
    pos_dot = DCM @ np.array([u_b, v_b, w_b])

    return np.array(
        [
            du,
            dv,
            dw,
            dp,
            dq,
            dr_dot,
            dphi,
            dtheta,
            dpsi,
            pos_dot[0],
            pos_dot[1],
            pos_dot[2],
        ],
        dtype=np.float64,
    )
