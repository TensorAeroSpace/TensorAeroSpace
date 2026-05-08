"""Engine state effects.

The angular F-16 ODE in this codebase does not currently consume thrust
directly — airspeed is held constant. This module exposes a small utility
so downstream consumers (gym envs, RL agents, future dynamics extensions)
can read the effective thrust given the engine's damage state.
"""

from __future__ import annotations

from .state import DamageState


def effective_thrust(base_thrust: float, state: DamageState) -> float:
    """Apply engine damage factors to a base thrust value.

    Combines the discrete-failure multiplier ``thrust_factor`` (used by
    ENGINE_FLAMEOUT and similar abrupt presets) with the slow-drift
    multiplier ``thrust_scale`` (Phase 2 Task 12 ENGINE_THRUST_DRIFT).
    Both default to 1.0 in healthy state, so this is bit-identical to
    the legacy single-factor behaviour when no drift is in effect.
    """
    if state.engine.hard_failure:
        return 0.0
    return float(
        base_thrust * state.engine.thrust_factor * state.engine.thrust_scale
    )
