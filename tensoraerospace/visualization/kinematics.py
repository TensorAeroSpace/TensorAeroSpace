"""Position reconstruction kinematics for F-16 flight visualization.

The pure-numpy F-16 models track aerodynamic angles, body rates and (for
the 6-DoF model) Euler angles, but no inertial position. To draw a
trajectory we integrate position from a constant assumed airspeed using
the standard body→inertial direction-cosine matrix.

Inertial frame convention used here:
    +x — forward (north / body x at heading 0)
    +y — right  (east)
    +z — up

Pitch up therefore moves the aircraft toward +z. This is opposite to the
NED (z down) convention common in flight dynamics literature, but
matches the natural Plotly 3D scene orientation (z up) and is purely a
visualization choice — kinematics code does not commit to either.
"""

from __future__ import annotations

import numpy as np


def _body_to_inertial_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Direction-cosine matrix mapping body-frame vectors to inertial.

    ZYX (yaw-pitch-roll) Tait-Bryan rotation, with the inertial z-axis
    pointing UP (so positive pitch raises the nose toward +z, matching
    the Plotly 3D scene orientation used by the visualization layer):

        R = Rz(yaw) @ Ry(-pitch) @ Rx(roll)

    Identity at (roll, pitch, yaw) = (0, 0, 0). Pure pitch +π/4 maps
    body x = (1, 0, 0) to inertial (cos π/4, 0, sin π/4). Pure roll
    leaves the body x-axis invariant in inertial coordinates.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, -cy * sp * sr - sy * cr, -cy * sp * cr + sy * sr],
        [sy * cp, -sy * sp * sr + cy * cr, -sy * sp * cr - cy * sr],
        [sp,       cp * sr,                 cp * cr],
    ])


def _body_velocity(airspeed: float, alpha: float, beta: float) -> np.ndarray:
    """Velocity vector in the body frame for given airspeed, AoA, sideslip."""
    return airspeed * np.array([
        np.cos(alpha) * np.cos(beta),
        np.sin(beta),
        np.sin(alpha) * np.cos(beta),
    ])


def reconstruct_position_6dof(
    state_history: np.ndarray,
    *,
    airspeed: float,
    dt: float,
    initial_position: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct (T, 3) inertial position from 6-DoF state history.

    Args:
        state_history: shape (T, n>=8) with cols
            [alpha, beta, wx, wy, wz, gamma, psi, theta, ...].
            Only cols 0, 1, 5, 6, 7 are read.
        airspeed: True airspeed (m/s) — assumed constant.
        dt: Time step between consecutive state rows (s).
        initial_position: (3,) inertial position at t=0; defaults to origin.

    Returns:
        Positions (T, 3) in metres, inertial frame.
    """
    state = np.asarray(state_history, dtype=np.float64)
    T = state.shape[0]
    pos = np.zeros((T, 3), dtype=np.float64)
    if initial_position is not None:
        pos[0] = np.asarray(initial_position, dtype=np.float64).reshape(3)
    for t in range(T - 1):
        alpha, beta = state[t, 0], state[t, 1]
        roll, yaw, pitch = state[t, 5], state[t, 6], state[t, 7]  # gamma, psi, theta
        v_body = _body_velocity(airspeed, alpha, beta)
        v_inertial = _body_to_inertial_matrix(roll, pitch, yaw) @ v_body
        pos[t + 1] = pos[t] + v_inertial * dt
    return pos


def reconstruct_position_longitudinal(
    state_history: np.ndarray,
    *,
    airspeed: float,
    dt: float,
    initial_pitch: float = 0.0,
    initial_position: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct positions AND attitudes for the longitudinal model.

    The longitudinal model state is [alpha, wz, stab, dstab]; pitch
    angle θ is integrated from wz. Roll and yaw are zero.

    Args:
        state_history: shape (T, n>=2) with cols [alpha, wz, ...].
        airspeed: True airspeed (m/s).
        dt: Time step (s).
        initial_pitch: θ(t=0) in radians.
        initial_position: (3,) inertial position at t=0.

    Returns:
        Tuple ``(positions (T, 3) with y=0, attitudes (T, 3))`` where
        ``attitudes[:, 1]`` is θ and the other two columns are zero.
    """
    state = np.asarray(state_history, dtype=np.float64)
    T = state.shape[0]
    pos = np.zeros((T, 3), dtype=np.float64)
    att = np.zeros((T, 3), dtype=np.float64)
    if initial_position is not None:
        pos[0] = np.asarray(initial_position, dtype=np.float64).reshape(3)
    att[0, 1] = initial_pitch
    # Integrate pitch first. ``wz`` from the longitudinal F-16 model is
    # consumed here as a per-step pitch increment (radians/step), not a
    # rate (radians/second), so the time step is intentionally NOT
    # multiplied in: theta[t+1] = theta[t] + wz[t]. The position
    # integration below still uses dt.
    for t in range(T - 1):
        wz = state[t, 1]
        att[t + 1, 1] = att[t, 1] + wz
    # Then integrate position with the pitch trajectory
    for t in range(T - 1):
        alpha = state[t, 0]
        pitch = att[t, 1]
        v_body = _body_velocity(airspeed, alpha, beta=0.0)
        v_inertial = _body_to_inertial_matrix(0.0, pitch, 0.0) @ v_body
        pos[t + 1] = pos[t] + v_inertial * dt
    return pos, att
