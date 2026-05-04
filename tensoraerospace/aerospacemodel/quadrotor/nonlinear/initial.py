"""Default initial state for the nonlinear quadrotor model."""

from __future__ import annotations

import numpy as np

# Hover at altitude 0 (NED z=0 means at-the-reference-altitude), all
# velocities, attitude, and rates zero. The hover thrust that keeps
# this state stationary is T = m·g (computed by the env layer).
initial_state = np.zeros((12, 1), dtype=np.float64)


initial_state_dict = {
    "x_e":   0.0,
    "y_e":   0.0,
    "z_e":   0.0,
    "u_b":   0.0,
    "v_b":   0.0,
    "w_b":   0.0,
    "phi":   0.0,
    "theta": 0.0,
    "psi":   0.0,
    "p":     0.0,
    "q":     0.0,
    "r":     0.0,
}


def set_initial_state(**overrides) -> np.ndarray:
    """Build a new ``(12, 1)`` initial-state vector with overrides.

    Example:
        ``set_initial_state(z_e=-5.0, theta=0.05)`` returns hover at
        5 m altitude with a 0.05 rad pitch.
    """
    layout = list(initial_state_dict.keys())
    state = np.zeros((12, 1), dtype=np.float64)
    for k, v in overrides.items():
        if k not in initial_state_dict:
            raise KeyError(
                f"Unknown state component {k!r}; valid keys are {layout}"
            )
        state[layout.index(k), 0] = float(v)
    return state
