"""Shared body-rate to ZYX Euler-rate conversion for rigid-body models."""

import numpy as np


def body_rates_to_euler_rates(
    phi: float, theta: float, p: float, q: float, r: float
) -> tuple[float, float, float]:
    """Return roll, pitch and yaw angle rates from body rates (rad/s).

    Preserve the sign of cos(theta) outside the principal pitch interval.
    The denominator floor only regularizes the Euler singularity; it does
    not make Euler angles suitable for trajectories through gimbal lock.
    """
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    cos_theta = float(np.cos(theta))
    safe_cos = float(np.copysign(max(abs(cos_theta), 1e-9), cos_theta))
    yaw_numerator = q * sin_phi + r * cos_phi
    return (
        float(p + yaw_numerator * np.sin(theta) / safe_cos),
        float(q * cos_phi - r * sin_phi),
        float(yaw_numerator / safe_cos),
    )
