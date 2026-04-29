"""F-16 6-DoF angular ODE right-hand side.

Direct line-by-line port of angular/matlab_code/F16ODE.m.

State vector (14 elements):
    [alpha, beta, wx, wy, wz, gamma, psi, theta,
     stab, dstab, ail, dail, dir, ddir]

Control vector (3 elements):
    [stab_act, ail_act, dir_act]

Note: split-stab differential is communicated through
``params.delta_stab_cmd`` (set by AngularF16.run_step in split_stab
mode), not via the control vector. This keeps the ODE arity stable.
"""

from __future__ import annotations

import numpy as np

from .aero import get_cx, get_cy, get_cz, get_mx, get_my, get_mz
from .params import F16AngularParameters, _isa_dynamic_pressure

# State indices
I_ALPHA, I_BETA, I_WX, I_WY, I_WZ = 0, 1, 2, 3, 4
I_GAMMA, I_PSI, I_THETA = 5, 6, 7
I_STAB, I_DSTAB = 8, 9
I_AIL, I_DAIL = 10, 11
I_DIR, I_DDIR = 12, 13


def f16_ode_6dof(
    x: np.ndarray, u: np.ndarray, t: float, params: F16AngularParameters
) -> np.ndarray:
    """Compute the 6-DoF angular ODE right-hand side.

    Parameters
    ----------
    x : array_like, shape (14,)
        State vector.
    u : array_like, shape (3,)
        Control vector [stab_act, ail_act, dir_act] in radians.
    t : float
        Current time (unused; kept for integrator compatibility).
    params : F16AngularParameters
        Aircraft and flight parameters.

    Returns
    -------
    dx : ndarray, shape (14,)
        Time derivative of the state vector.
    """
    p = params

    # Unpack state
    alpha = float(x[I_ALPHA])
    beta = float(x[I_BETA])
    wx = float(x[I_WX])
    wy = float(x[I_WY])
    wz = float(x[I_WZ])
    gamma = float(x[I_GAMMA])
    psi = float(x[I_PSI])
    theta = float(x[I_THETA])
    stab = float(x[I_STAB])
    dstab = float(x[I_DSTAB])
    ail = float(x[I_AIL])
    dail = float(x[I_DAIL])
    direc = float(x[I_DIR])
    ddir = float(x[I_DDIR])

    # Unpack control
    stab_act = float(u[0])
    ail_act = float(u[1])
    dir_act = float(u[2])

    # ---- Altitude-aware mode (track_altitude) ----
    track_altitude = x.size >= 16
    if track_altitude:
        h_state = float(x[14])
        V_state = float(x[15])
        q_now = _isa_dynamic_pressure(h_state, V_state, p.g)
    else:
        h_state = p.Oy
        V_state = p.V
        q_now = p.q

    # ----------------------------------------------------------------
    # Aerodynamic coefficients
    # (argument order mirrors matlab: GetCx/Cy/Mz use stab as fi)
    # ----------------------------------------------------------------
    cx = get_cx(alpha, beta, stab, p.lef, wz, V_state, p.bA, p.sb)
    cy = get_cy(alpha, beta, stab, p.lef, wz, V_state, p.bA, p.sb)
    cz = get_cz(alpha, beta, direc, ail, p.lef, wx, wy, V_state, p.l)
    mx_ = get_mx(alpha, beta, stab, direc, ail, p.lef, wx, wy, V_state, p.l)
    my_ = get_my(alpha, beta, stab, direc, ail, p.lef, wx, wy, V_state, p.l)
    mz_ = get_mz(alpha, beta, stab, p.lef, wz, V_state, p.bA, p.sb)

    # ----------------------------------------------------------------
    # Apply damage corrections (no-op if damage_state is None)
    # ----------------------------------------------------------------
    damage_state = p.damage_state
    damage_geo = p.damage_geometry
    if damage_state is not None and damage_geo is not None:
        from ..damage import aero_corrections as _ac
        cy = cy + _ac.delta_cy(alpha, beta, damage_geo, damage_state)
        cx = cx + _ac.delta_cx(alpha, beta, damage_geo, damage_state)
        cz = cz + _ac.delta_cz(alpha, beta, damage_geo, damage_state)
        mx_ = mx_ + _ac.delta_mx(alpha, beta, damage_geo, damage_state)
        my_ = my_ + _ac.delta_my(alpha, beta, damage_geo, damage_state)
        mz_ = mz_ + _ac.delta_mz(alpha, beta, damage_geo, damage_state)

    # ----------------------------------------------------------------
    # Aerodynamic forces (body frame)
    # ----------------------------------------------------------------
    X = -q_now * p.S * cx
    Y = q_now * p.S * cy
    Z = q_now * p.S * cz

    # ----------------------------------------------------------------
    # Aerodynamic moments (body frame)
    # ----------------------------------------------------------------
    # Differential stabilator → roll moment.
    # delta_stab > 0 means left half UP, right DOWN (positive roll =
    # right-wing-down). The gated branch preserves bit-identical behaviour
    # for symmetric/legacy operation (delta_stab == 0.0 → exact same Mx).
    delta_stab = float(p.delta_stab_cmd)
    if delta_stab != 0.0:
        Mx = q_now * p.S * p.l * (mx_ - p.delta_stab_roll_gain * delta_stab)
    else:
        Mx = q_now * p.S * p.l * mx_
    My = q_now * p.S * p.l * my_
    Mz = q_now * p.S * p.bA * mz_

    # ----------------------------------------------------------------
    # Resultant forces and moments (shifted to actual CG)
    # ----------------------------------------------------------------
    Rx = X
    Ry = Y
    Rz = Z
    MRx = Mx
    MRy = My - p.rcgx * Rz
    MRz = Mz + p.rcgx * Ry

    # ----------------------------------------------------------------
    # Angular-momentum equations (Dx.wx, Dx.wy, Dx.wz)
    # ----------------------------------------------------------------
    Gamma = p.Jx * p.Jy - p.Jxy**2

    dwx = (
        p.Jy * MRx
        + p.Jxy * (MRy - p.hEx * wz)
        + p.Jxy * (p.Jz - p.Jx - p.Jy) * wx * wz
        + (p.Jxy**2 + p.Jy * (p.Jy - p.Jz)) * wy * wz
    ) / Gamma

    dwy = (
        p.Jxy * MRx
        + p.Jx * (MRy - p.hEx * wz)
        + (p.Jx * (p.Jz - p.Jx) - p.Jxy**2) * wx * wz
        + p.Jxy * (p.Jx + p.Jy - p.Jz) * wy * wz
    ) / Gamma

    dwz = (MRz + p.hEx * wy + p.Jxy * (wx**2 - wy**2) + (p.Jx - p.Jy) * wx * wy) / p.Jz

    # ----------------------------------------------------------------
    # Gravitational acceleration components in wind/body aerodynamic axes
    # (Dx.alpha, Dx.beta)
    # ----------------------------------------------------------------
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)
    sin_g = np.sin(gamma)
    cos_g = np.cos(gamma)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # gay and gaz as in matlab (gax is computed but unused in alpha/beta eqs)
    gay = p.g * (-sin_th * sin_a - cos_g * cos_th * cos_a)
    gaz = p.g * (
        sin_th * cos_a * sin_b - cos_g * cos_th * sin_a * sin_b + sin_g * cos_th * cos_b
    )

    # Aerodynamic-frame force components
    Ya = -sin_a * Rx + cos_a * Ry
    Za = cos_a * sin_b * Rx + sin_a * sin_b * Ry + cos_b * Rz

    dalpha = (
        wz
        + (wy * sin_a - wx * cos_a) * np.tan(beta)
        - (Ya + p.m * gay) / (p.m * V_state * cos_b)
    )

    dbeta = wx * sin_a + wy * cos_a + (Za + p.m * gaz) / (p.m * V_state)

    # ----------------------------------------------------------------
    # Euler-angle kinematics (Dx.gamma, Dx.theta, Dx.psi)
    # ----------------------------------------------------------------
    dgamma = wx - cos_g * np.tan(theta) * wy + sin_g * np.tan(theta) * wz

    dtheta = sin_g * wy + cos_g * wz

    dpsi = (cos_g / cos_th) * wy - (sin_g / cos_th) * wz

    # ----------------------------------------------------------------
    # Actuator models (second-order with rate and position limiting)
    # ----------------------------------------------------------------
    # Stabilator
    dstab_out = float(np.clip(dstab, -p.maxabsdstab, p.maxabsdstab))
    stab_act_c = float(np.clip(stab_act, -p.maxabsstab, p.maxabsstab))
    ddstab = (-2.0 * p.Tstab * p.Xistab * dstab - stab + stab_act_c) / (p.Tstab**2)

    # Aileron
    dail_out = float(np.clip(dail, -p.maxabsdail, p.maxabsdail))
    ail_act_c = float(np.clip(ail_act, -p.maxabsail, p.maxabsail))
    ddail = (-2.0 * p.Tail * p.Xiail * dail - ail + ail_act_c) / (p.Tail**2)

    # Rudder
    ddir_out = float(np.clip(ddir, -p.maxabsddir, p.maxabsddir))
    dir_act_c = float(np.clip(dir_act, -p.maxabsdir, p.maxabsdir))
    dddir = (-2.0 * p.Tdir * p.Xidir * ddir - direc + dir_act_c) / (p.Tdir**2)

    # ---- Altitude / airspeed dynamics (only when track_altitude) ----
    if track_altitude:
        # Body-frame velocity vector (V along body x, projected by alpha/beta)
        v_body_x = V_state * cos_a * cos_b
        v_body_y = V_state * sin_b
        v_body_z = V_state * sin_a * cos_b
        # Inertial z-component (NED, +z is down). Body→inertial via the
        # standard 3-2-1 Euler matrix (yaw, pitch, roll).
        v_inertial_z = (
            -sin_th * v_body_x
            + cos_th * sin_g * v_body_y
            + cos_th * cos_g * v_body_z
        )
        dh = -v_inertial_z

        # Flight path angle: angle between velocity vector and local horizontal.
        # sin(γ) = -v_inertial_z / V (positive = climbing).
        sin_gamma_path = -v_inertial_z / max(V_state, 1e-3)
        sin_gamma_path = max(-1.0, min(1.0, sin_gamma_path))

        # Drag magnitude (D = q·S·Cx; X = -D in body x convention).
        drag = q_now * p.S * cx

        # Thrust: from runtime override if set, otherwise from constant.
        thrust = p.T_active
        if not (thrust == thrust):  # NaN check
            thrust = p.T_thrust

        # Energy: dV/dt = (T·cos(α)·cos(β) − D) / m − g·sin(γ)
        dV = (thrust * cos_a * cos_b - drag) / p.m - p.g * sin_gamma_path

    base_dx = [
        dalpha, dbeta, dwx, dwy, dwz,
        dgamma, dpsi, dtheta,
        dstab_out, ddstab,
        dail_out, ddail,
        ddir_out, dddir,
    ]
    if track_altitude:
        base_dx.extend([dh, dV])
    return np.array(base_dx, dtype=np.float64)
