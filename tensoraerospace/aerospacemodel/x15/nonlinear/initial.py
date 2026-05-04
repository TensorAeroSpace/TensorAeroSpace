"""State vector layout and initialisation helpers for the X-15 model.

The X-15 nonlinear state has **13 elements** — the standard 12-D
rigid-body state plus an extra propellant-mass channel:

    [u, v, w,           # body velocity, ft/s
     p, q, r,           # body angular rates, rad/s
     phi, theta, psi,   # Euler angles, rad
     x_e, y_e, z_e,     # NED position, ft  (z_e positive down)
     m_prop]            # remaining propellant, lb

The propellant channel decreases monotonically from the configuration's
full load to zero. Once it reaches zero, the engine flames out
naturally (returns zero thrust regardless of throttle command).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .flight_conditions import X15FlightCondition, get_flight_condition
from .params import X15Configuration, X15Parameters, default_parameters


STATE_LIST = [
    "u", "v", "w",
    "p", "q", "r",
    "phi", "theta", "psi",
    "x_e", "y_e", "z_e",
    "m_prop",
]
STATE_DIM = len(STATE_LIST)


def default_state() -> np.ndarray:
    """Zero state vector with full propellant (13 elements)."""
    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[12] = 13_000.0   # default to BASIC config full load
    return x


def initial_state_from_fc(
    fc: X15FlightCondition,
    *,
    config: X15Configuration = X15Configuration.BASIC,
) -> np.ndarray:
    """Build a 13-state from a published anchor flight condition.

    The flight condition's α₀ is interpreted as the body-axis angle
    of attack at level flight (γ = 0), so:

    .. math::
       u_b = V \\cos α_0,\\qquad w_b = V \\sin α_0,
       \\qquad \\theta = α_0,\\qquad m_{prop} = \\text{fc.propellant\\_lb}.
    """
    V = fc.V_ft_s
    alpha = math.radians(fc.alpha0_deg)
    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[0] = V * math.cos(alpha)         # u_b
    x[1] = 0.0                          # v_b
    x[2] = V * math.sin(alpha)          # w_b
    x[7] = alpha                        # theta (γ = 0)
    x[11] = -fc.altitude_ft             # z_e (NED: positive down)
    x[12] = fc.propellant_lb
    return x


def set_initial_state(
    *,
    config: X15Configuration = X15Configuration.BASIC,
    altitude_ft: float = 45_000.0,
    V_ft_s: float = 800.0,
    alpha_deg: float = 4.0,
    theta_deg: Optional[float] = None,
    propellant_lb: Optional[float] = None,
) -> np.ndarray:
    """Free-form initial state for non-anchor scenarios.

    If ``theta_deg`` is None it defaults to ``alpha_deg`` (i.e. level
    flight). If ``propellant_lb`` is None the configuration's full
    propellant load is used.
    """
    if theta_deg is None:
        theta_deg = alpha_deg
    if propellant_lb is None:
        propellant_lb = default_parameters(config).propellant_full_lb
    alpha = math.radians(alpha_deg)
    theta = math.radians(theta_deg)
    x = np.zeros(STATE_DIM, dtype=np.float64)
    x[0] = V_ft_s * math.cos(alpha)
    x[1] = 0.0
    x[2] = V_ft_s * math.sin(alpha)
    x[7] = theta
    x[11] = -float(altitude_ft)
    x[12] = float(propellant_lb)
    return x
