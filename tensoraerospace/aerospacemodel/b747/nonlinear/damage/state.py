"""Damage state for the B-747 nonlinear simulation.

Three groups of failures are modelled:

1. **Per-surface actuator damage** (elevator / aileron / rudder /
   throttle):

   * ``mu`` ∈ [0, 1] — multiplicative effectiveness (1 = healthy).
   * ``jam_value`` — float | None: when not None, the surface is
     locked at this commanded value (rad), regardless of the agent's
     command.
   * ``tau`` — exponential decay time constant; 0 means no decay.
   * ``mu_floor`` — asymptote of the decay.

2. **Per-engine propulsion damage** (1..4): ``engines_mu[i]`` ∈ [0, 1]
   scales engine *i*'s thrust independently. Asymmetric values
   produce a yaw moment computed by the engine model from the
   spanwise engine positions.

3. **Configuration jamming** (mechanical flap / gear failure):
   ``flap_jam_config`` overrides :attr:`B747Parameters.config` for the
   aerodynamic build, simulating leading-edge or trailing-edge
   high-lift devices stuck at one position.

The state object is mutable; events update it in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..params import B747Configuration

_SURFACES = ("elevator", "aileron", "rudder", "throttle")
_ENGINES = (1, 2, 3, 4)


@dataclass
class B747DamageState:
    """Combined actuator + propulsion + configuration damage state."""

    mu: dict[str, float] = field(default_factory=lambda: {s: 1.0 for s in _SURFACES})
    jam: dict[str, Optional[float]] = field(
        default_factory=lambda: {s: None for s in _SURFACES}
    )
    tau: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in _SURFACES})
    mu_floor: dict[str, float] = field(
        default_factory=lambda: {s: 0.0 for s in _SURFACES}
    )
    engines_mu: dict[int, float] = field(
        default_factory=lambda: {i: 1.0 for i in _ENGINES}
    )
    flap_jam_config: Optional[B747Configuration] = None

    def __post_init__(self) -> None:
        for s in _SURFACES:
            if s not in self.mu:
                self.mu[s] = 1.0
            if not 0.0 <= self.mu[s] <= 1.0:
                raise ValueError(f"mu[{s}] must be in [0, 1]; got {self.mu[s]}")
            if s not in self.tau:
                self.tau[s] = 0.0
            if s not in self.mu_floor:
                self.mu_floor[s] = 0.0
            if s not in self.jam:
                self.jam[s] = None
        for i in _ENGINES:
            if i not in self.engines_mu:
                self.engines_mu[i] = 1.0
            if not 0.0 <= self.engines_mu[i] <= 1.0:
                raise ValueError(
                    f"engines_mu[{i}] must be in [0, 1]; got {self.engines_mu[i]}"
                )

    @classmethod
    def healthy(cls) -> "B747DamageState":
        """Return a fresh state with all surfaces at full effectiveness."""
        return cls()

    def step_decay(self, dt: float) -> None:
        """Advance time-decay surfaces by one Euler step of size ``dt``."""
        for s, tau in self.tau.items():
            if tau > 0.0:
                floor = self.mu_floor[s]
                self.mu[s] = floor + (self.mu[s] - floor) * math.exp(-dt / tau)
                # Clamp to [floor, 1.0] in case of numerical noise
                self.mu[s] = float(np.clip(self.mu[s], floor, 1.0))

    def apply(self, u_virtual: np.ndarray) -> np.ndarray:
        """Apply effectiveness multipliers + jam holds to a virtual command.

        Args:
            u_virtual: ``[δ_e, δ_a, δ_r, δ_T]`` as commanded by the agent.

        Returns:
            Effective command after damage. Jammed surfaces ignore the
            commanded value; healthy ones pass through.
        """
        u = np.asarray(u_virtual, dtype=np.float64).copy()
        for i, s in enumerate(_SURFACES):
            if self.jam[s] is not None:
                u[i] = self.jam[s]
            else:
                u[i] *= self.mu[s]
        return u

    def snapshot(self) -> dict:
        """Frozen dict view, JSON-friendly for logging."""
        return {
            "mu": dict(self.mu),
            "jam": {s: (None if v is None else float(v)) for s, v in self.jam.items()},
            "tau": dict(self.tau),
            "mu_floor": dict(self.mu_floor),
            "engines_mu": dict(self.engines_mu),
            "flap_jam_config": (
                None if self.flap_jam_config is None else self.flap_jam_config.value
            ),
        }


SURFACES = _SURFACES
ENGINES = _ENGINES
