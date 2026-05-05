"""Quadrotor 6-DoF nonlinear dynamics — pure-numpy RHS.

Conventions
-----------
- **Earth frame:** NED (north / east / down). Gravity is +g along z.
- **Body frame:** x forward, y right, z down (matches NED when level).
  Thrust is **−z body** (up when level), so the thrust force vector in
  the body frame is ``[0, 0, -T]``.
- **Euler angles:** ZYX (yaw→pitch→roll, "321" convention). Singularity
  at ``|theta| = π/2`` (gimbal lock); the kinematic mapping below uses
  ``1/cos(theta)`` so trajectories that approach vertical pitch are
  numerically unstable — use a quaternion variant for aggressive
  acrobatic regimes.
- **Angular rates:** ``[p, q, r]`` in body frame (roll, pitch, yaw rate).

State (12)
----------
``[x_e, y_e, z_e, u_b, v_b, w_b, phi, theta, psi, p_b, q_b, r_b]``

Control (4)
-----------
``[T, tau_x, tau_y, tau_z]`` — collective thrust + body-frame torques.
"""

from __future__ import annotations

import numpy as np

from .params import QuadrotorParameters


def _rotation_be(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation from **body to earth** frame, ZYX (321) Euler angles.

    Returns 3×3 matrix ``R_eb`` such that ``v_e = R_eb @ v_b``.
    """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [
                cth * cpsi,
                sphi * sth * cpsi - cphi * spsi,
                cphi * sth * cpsi + sphi * spsi,
            ],
            [
                cth * spsi,
                sphi * sth * spsi + cphi * cpsi,
                cphi * sth * spsi - sphi * cpsi,
            ],
            [-sth, sphi * cth, cphi * cth],
        ],
        dtype=np.float64,
    )


def quadrotor_ode_6dof(
    x: np.ndarray,
    u: np.ndarray,
    t: float,
    params: QuadrotorParameters,
) -> np.ndarray:
    """Right-hand side of the 6-DoF quadrotor ODE.

    Args:
        x: state vector (12,) — see module docstring for layout.
        u: control vector (4,) — ``[T, tau_x, tau_y, tau_z]``.
        t: current time, s (unused — model is autonomous).
        params: inertial / geometric parameters.

    Returns:
        ``dx/dt`` of shape ``(12,)``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if x.size != 12:
        raise ValueError(f"state must have 12 elements; got {x.size}")
    if u.size != 4:
        raise ValueError(f"control must have 4 elements; got {u.size}")

    # Unpack state (positions x[0:3] not needed in body-frame dynamics)
    u_b, v_b, w_b = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]

    # Unpack control
    T, tau_x, tau_y, tau_z = u[0], u[1], u[2], u[3]

    m, g = params.m, params.g
    Jx, Jy, Jz = params.Jx, params.Jy, params.Jz
    kdx, kdy, kdz = params.kdx, params.kdy, params.kdz

    # 1. Position kinematics (body velocity → earth-frame position rate)
    R_eb = _rotation_be(phi, theta, psi)
    pos_dot = R_eb @ np.array([u_b, v_b, w_b], dtype=np.float64)

    # 2. Body-frame velocity dynamics
    #    m·v̇_b = −ω × m·v_b  +  R_be · F_gravity_e  +  F_thrust_b  −  D · v_b
    #    F_gravity_e (NED, down is +z) = [0, 0, m·g]
    #    F_thrust_b  (thrust along −z body) = [0, 0, −T]
    R_be = R_eb.T
    F_grav_b = R_be @ np.array([0.0, 0.0, m * g], dtype=np.float64)
    F_thrust_b = np.array([0.0, 0.0, -T], dtype=np.float64)
    F_drag_b = np.array([kdx * u_b, kdy * v_b, kdz * w_b], dtype=np.float64)

    # ω × m·v_b
    omega = np.array([p, q, r], dtype=np.float64)
    v_b = np.array([u_b, v_b, w_b], dtype=np.float64)
    coriolis_b = np.cross(omega, m * v_b)

    v_b_dot = (F_grav_b + F_thrust_b - F_drag_b - coriolis_b) / m

    # 3. Euler-angle kinematics  (ZYX 321: phi=roll, theta=pitch, psi=yaw)
    cth = np.cos(theta)
    if abs(cth) < 1e-9:
        # Gimbal lock — clamp to avoid 1/0 blow-up; trajectory is
        # already invalid at this attitude with Euler angles.
        cth = np.copysign(1e-9, cth)
    tth = np.tan(theta)
    sphi, cphi = np.sin(phi), np.cos(phi)

    phi_dot = p + sphi * tth * q + cphi * tth * r
    theta_dot = cphi * q - sphi * r
    psi_dot = (sphi / cth) * q + (cphi / cth) * r

    # 4. Angular-rate dynamics (Newton-Euler, diagonal inertia tensor)
    p_dot = (tau_x + (Jy - Jz) * q * r) / Jx
    q_dot = (tau_y + (Jz - Jx) * p * r) / Jy
    r_dot = (tau_z + (Jx - Jy) * p * q) / Jz

    return np.array(
        [
            pos_dot[0],
            pos_dot[1],
            pos_dot[2],
            v_b_dot[0],
            v_b_dot[1],
            v_b_dot[2],
            phi_dot,
            theta_dot,
            psi_dot,
            p_dot,
            q_dot,
            r_dot,
        ],
        dtype=np.float64,
    )
