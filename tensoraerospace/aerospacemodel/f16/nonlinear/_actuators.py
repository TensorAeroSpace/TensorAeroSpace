"""Second-order servos with physical rate limits and hard travel stops.

Inside the limits the original linear servo equation is unchanged. At a stop,
only outward motion is removed; inward motion remains possible. Models project
accepted integration states to remove numerical overshoot and rate windup.
"""

from __future__ import annotations

import numpy as np


def actuator_derivatives(
    position, rate, command, time_constant, damping, limit, rate_limit
):
    """Return actual surface velocity and acceleration in radians and seconds."""
    velocity = float(np.clip(rate, -rate_limit, rate_limit))
    command = float(np.clip(command, -limit, limit))
    if (position >= limit and velocity > 0) or (position <= -limit and velocity < 0):
        velocity = 0.0
    acceleration = (
        -2.0 * time_constant * damping * velocity - position + command
    ) / time_constant**2
    if (rate >= rate_limit and acceleration > 0) or (
        rate <= -rate_limit and acceleration < 0
    ):
        acceleration = 0.0
    return velocity, acceleration


def project_actuators(state, params, channels):
    """Bound accepted state pairs (position, rate), stopping at travel limits."""
    result = np.array(state, dtype=np.float64, copy=True)
    for index, name in channels:
        limit = getattr(params, "maxabs" + name)
        rate_limit = getattr(params, "maxabsd" + name)
        position = float(np.clip(result[index], -limit, limit))
        rate = float(np.clip(result[index + 1], -rate_limit, rate_limit))
        if (position >= limit and rate > 0) or (position <= -limit and rate < 0):
            rate = 0.0
        result[index : index + 2] = position, rate
    return result


def surface_limits(params, split_stab=False):
    """Physical command limits for the angular model, in action-vector order."""
    limits = [params.maxabsstab, params.maxabsail, params.maxabsdir]
    if split_stab:
        limits.insert(0, params.maxabsstab)
    return np.asarray(limits, dtype=np.float64)
