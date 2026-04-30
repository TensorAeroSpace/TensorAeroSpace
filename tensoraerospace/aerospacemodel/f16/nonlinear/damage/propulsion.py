"""Engine state effects.

The angular F-16 ODE in this codebase does not currently consume thrust
directly — airspeed is held constant. This module exposes a small utility
so downstream consumers (gym envs, RL agents, future dynamics extensions)
can read the effective thrust given the engine's damage state.
"""

from __future__ import annotations

from .state import DamageState


def effective_thrust(base_thrust: float, state: DamageState) -> float:
    """Apply engine.thrust_factor and hard_failure to a base thrust value."""
    if state.engine.hard_failure:
        return 0.0
    return float(base_thrust * state.engine.thrust_factor)
