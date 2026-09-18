"""Engine state effects.

The angular F-16 ODE applies this correction to thrust when altitude and
airspeed are integrated. The resulting body-axis force contributes to speed,
angle-of-attack and sideslip acceleration. Fixed-speed reduced models do not
model the translational response to engine failure.
"""

from __future__ import annotations

from .state import DamageState


def effective_thrust(base_thrust: float, state: DamageState) -> float:
    """Apply engine.thrust_factor and hard_failure to a base thrust value."""
    if state.engine.hard_failure:
        return 0.0
    return float(base_thrust * state.engine.thrust_factor)
