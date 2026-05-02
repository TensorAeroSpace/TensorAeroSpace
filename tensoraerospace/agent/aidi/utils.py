"""AIDI utility helpers — body-frame load-factor reconstruction etc.

For envs that do not expose ``n_z`` directly we reconstruct it from
``(α, α̇, q, V, θ, φ)`` using the standard small-β relation::

    n_z ≈ (V/g)·(q·cos α − α̇) + cos θ · cos φ

The cosine term encodes the gravity component along the body-z axis at
the current attitude. Inverted level flight (φ = π) gives n_z ≈ −1.
"""

from __future__ import annotations

import math

GRAVITY = 9.80665  # m/s²


def reconstruct_n_z(
    alpha: float,
    alpha_dot: float,
    q: float,
    V: float,
    theta: float,
    phi: float,
) -> float:
    """Reconstruct the body-frame load factor.

    Args:
        alpha: Angle of attack (rad).
        alpha_dot: Time derivative of α (rad/s).
        q: Pitch rate (rad/s).
        V: True airspeed (m/s).
        theta: Pitch attitude (rad).
        phi: Roll attitude (rad).
    """
    aero = (float(V) / GRAVITY) * (float(q) * math.cos(float(alpha)) - float(alpha_dot))
    grav = math.cos(float(theta)) * math.cos(float(phi))
    return aero + grav
