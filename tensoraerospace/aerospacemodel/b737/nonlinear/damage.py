"""Parametric elevator-authority loss for the nonlinear B737.

The aerodynamic table receives effectiveness times the physical elevator
angle. Encoder feedback remains the physical angle; mass/inertia are unchanged.
This is a parameter study, not a calibrated structural-damage model.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElevatorEffectiveness:
    """Piecewise-constant elevator authority, applied from ``time`` seconds."""

    time: float = 30.0
    effectiveness: float = 0.5

    def __post_init__(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError("fault time must be finite and nonnegative")
        if not np.isfinite(self.effectiveness) or not 0 <= self.effectiveness <= 1:
            raise ValueError("effectiveness must be in [0, 1]")

    def at(self, time):
        """Aerodynamic effectiveness at the given physical time."""
        return self.effectiveness if time >= self.time else 1.0

    def apply(self, action, time):
        """Return an aerodynamic input without modifying physical feedback."""
        result = np.array(action, dtype=float, copy=True)
        result[0] *= self.at(time)
        return result
