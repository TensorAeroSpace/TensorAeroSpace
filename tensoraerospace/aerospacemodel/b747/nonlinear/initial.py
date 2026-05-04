"""Initial-state helpers for the B-747 nonlinear 6-DoF model.

State layout (12 elements):

    [u, v, w, p, q, r, phi, theta, psi, x_e, y_e, z_e]

Translational velocity is in the **body axis** in ft/s; angles are in
radians; positions in ft (NED, ``z_e`` positive down ⇒ altitude is
``-z_e``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .flight_conditions import B747FlightCondition, get_flight_condition

STATE_LIST = [
    "u", "v", "w",
    "p", "q", "r",
    "phi", "theta", "psi",
    "x_e", "y_e", "z_e",
]


def default_state() -> np.ndarray:
    """Zero-everything initial state — useful for synthetic tests only."""
    return np.zeros(12, dtype=np.float64)


def initial_state_from_fc(
    fc: B747FlightCondition,
    *,
    psi: float = 0.0,
    bank_deg: float = 0.0,
) -> np.ndarray:
    """Trimmed initial state at flight condition ``fc``.

    Body-axis velocity is decomposed from V via α₀ (β = 0). Pitch is
    set equal to α₀ so that the climb angle γ is zero (level flight).
    Heading and bank can be overridden via keyword arguments.

    Args:
        fc: Flight condition to anchor the trim point at.
        psi: Initial heading angle, radians.
        bank_deg: Initial bank angle (φ), degrees. Default is wings level.
    """
    alpha = np.deg2rad(fc.alpha0_deg)
    V = fc.V_ft_s
    state = np.zeros(12, dtype=np.float64)
    state[0] = V * np.cos(alpha)        # u
    state[1] = 0.0                      # v (β = 0)
    state[2] = V * np.sin(alpha)        # w
    state[6] = np.deg2rad(bank_deg)     # phi
    state[7] = alpha                     # theta = α₀ for γ = 0
    state[8] = psi
    state[11] = -fc.altitude_ft         # NED z (positive down)
    return state


def set_initial_state(
    state: Optional[np.ndarray] = None,
    **overrides: float,
) -> np.ndarray:
    """Construct an initial state by overriding individual entries by name.

    >>> set_initial_state(u=600.0, theta=np.deg2rad(2.0), z_e=-30000.0)

    Unspecified entries default to zero (or to ``state`` if supplied).
    """
    base = default_state() if state is None else np.asarray(state, dtype=np.float64).copy()
    for k, v in overrides.items():
        if k not in STATE_LIST:
            raise ValueError(f"unknown state key: {k!r} (allowed: {STATE_LIST})")
        base[STATE_LIST.index(k)] = float(v)
    return base


def initial_state_dict(state: np.ndarray) -> dict[str, float]:
    """Return a dict view of a 12-element state vector."""
    if state.size != 12:
        raise ValueError(f"state must have 12 elements; got {state.size}")
    return {name: float(state[i]) for i, name in enumerate(STATE_LIST)}
