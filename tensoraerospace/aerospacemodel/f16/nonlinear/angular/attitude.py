"""Singularity-free attitude kinematics for the angular F-16 model.

The state keeps Euler angles ``gamma`` (roll), ``psi`` (heading, positive left)
and ``theta`` (pitch) in the Russian body axes (x forward, y up, z right). Their
rate equations divide by ``cos(theta)`` and break down at ``theta = ±90°``
(gimbal lock): near a vertical pass the ``psi`` and ``gamma`` rates grow without
bound and a fixed integration step corrupts the attitude.

The model therefore integrates a unit quaternion and converts it back to Euler
angles after every step. Converting Euler angles to a quaternion is always
defined, and the gravity terms of the dynamics only use ``sin(theta)``,
``cos(theta)cos(gamma)`` and ``cos(theta)sin(gamma)``, which are rotation-matrix
entries and stay well defined at the vertical.

Convention: the quaternion rotates the conventional body frame (x forward,
y right, z down) into a north-east-down frame with ZYX angles
``yaw = -psi``, ``pitch = theta`` and ``roll = gamma``. Body rates map as
``p = wx``, ``q = wz`` and ``r = -wy``. These are the same kinematics as the
Euler-angle equations in ``dynamics.f16_ode_6dof``.
"""

from __future__ import annotations

import math

import numpy as np


def euler_to_quaternion(gamma: float, psi: float, theta: float) -> np.ndarray:
    """Return the unit quaternion ``[w, x, y, z]`` for model Euler angles."""
    half_roll, half_pitch, half_yaw = 0.5 * gamma, 0.5 * theta, -0.5 * psi
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _wrap_near(angle: float, reference: float) -> float:
    """``angle`` shifted by whole turns to lie within π of ``reference``."""
    return reference + math.remainder(angle - reference, 2.0 * math.pi)


def quaternion_to_euler(
    q: np.ndarray, reference: tuple[float, float] | None = None
) -> tuple[float, float, float]:
    """Return ``(gamma, psi, theta)`` with ``theta`` in ``[-π/2, π/2]``.

    ``reference`` is the previous ``(gamma, psi)``. Roll and heading are kept
    continuous with it (no artificial ±2π jumps), so unwrapped headings keep
    accumulating as before. Passing the vertical changes both by about π:
    that is the same attitude in the canonical pitch range.
    """
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r10 = 2.0 * (x * y + w * z)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    # atan2 keeps pitch well conditioned next to the vertical (asin does not).
    theta = math.atan2(-r20, math.hypot(r00, r10))
    if math.hypot(r21, r22) > 1e-12:
        gamma = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        # Exactly vertical: only gamma ∓ yaw is defined. Keep the previous
        # roll and put the whole rotation into heading.
        gamma = 0.0 if reference is None else reference[0]
        r01 = 2.0 * (x * y - w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        sign = 1.0 if theta > 0 else -1.0
        yaw = math.atan2(-r01, r11) + sign * gamma
    psi = -yaw
    if reference is not None:
        gamma = _wrap_near(gamma, reference[0])
        psi = _wrap_near(psi, reference[1])
    return gamma, psi, theta


def quaternion_rate(q: np.ndarray, wx: float, wy: float, wz: float) -> np.ndarray:
    """``dq/dt = ½ q ⊗ (0, p, q, r)`` for model body rates ``wx, wy, wz``."""
    p, qr, r = wx, wz, -wy
    w, x, y, z = q
    return 0.5 * np.array(
        [
            -x * p - y * qr - z * r,
            w * p + y * r - z * qr,
            w * qr - x * r + z * p,
            w * r + x * qr - y * p,
        ],
        dtype=np.float64,
    )
