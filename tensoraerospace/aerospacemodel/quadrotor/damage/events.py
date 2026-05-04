"""Damage-event types for the rotor-level subsystem.

Three concrete event subclasses:

- :class:`RotorDamageEvent` — multiplicative effectiveness loss applied
  instantaneously at ``trigger_time`` (Lu 2019).
- :class:`RotorLossEvent` — complete rotor stop, ``mu = 0`` (Lanzon 2015).
- :class:`MotorEfficiencyDecay` — sets a finite time-constant ``tau`` so
  the rotor's effectiveness exponentially decays to ``mu_floor``
  (gradual wear).

Events compose into :class:`DamageProfile`, replayed by the
:class:`~tensoraerospace.aerospacemodel.quadrotor.damage.manager.RotorDamageManager`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .state import RotorDamageState


@dataclass(frozen=True)
class DamageEvent:
    """Base class for rotor damage events."""

    trigger_time: float
    rotor_id: int
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be >= 0; got {self.trigger_time}")
        if not 0 <= self.rotor_id <= 3:
            raise ValueError(
                f"rotor_id must be in {{0, 1, 2, 3}}; got {self.rotor_id}"
            )

    def apply(self, state: RotorDamageState) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class RotorDamageEvent(DamageEvent):
    """Instantaneous multiplicative effectiveness loss on rotor ``rotor_id``.

    ``mu`` is the surviving fraction (1.0 = healthy, 0.0 = total loss).
    Mirrors the convention in Lu 2019 (ISMC FTC):
    :math:`\\omega^2_{i,\\text{eff}} = \\mu \\cdot \\omega^2_{i,\\text{cmd}}`.
    """

    mu: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.mu <= 1.0:
            raise ValueError(f"mu must be in [0, 1]; got {self.mu}")

    def apply(self, state: RotorDamageState) -> None:
        state.mu[self.rotor_id] = self.mu


@dataclass(frozen=True)
class RotorLossEvent(DamageEvent):
    """Complete rotor stop — ``mu = 0`` (Lanzon 2015 catastrophic case)."""

    def apply(self, state: RotorDamageState) -> None:
        state.mu[self.rotor_id] = 0.0
        # Cancel any active decay on this rotor since it's already gone
        state.tau[self.rotor_id] = 0.0


@dataclass(frozen=True)
class MotorEfficiencyDecay(DamageEvent):
    """Exponential effectiveness decay starting from ``trigger_time``.

    From the trigger onward, ``mu[i]`` evolves as

    .. math::
        \\dot \\mu_i = -(1/\\tau)(\\mu_i - \\mu_\\text{floor}),

    so ``mu`` exponentially approaches ``mu_floor`` with time-constant
    ``tau``. Applies a one-shot setter at trigger time; the actual
    decay is integrated by :meth:`RotorDamageState.step_decay` on each
    env step.
    """

    tau: float = 1.0
    mu_floor: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError(f"tau must be positive; got {self.tau}")
        if not 0.0 <= self.mu_floor <= 1.0:
            raise ValueError(f"mu_floor must be in [0, 1]; got {self.mu_floor}")

    def apply(self, state: RotorDamageState) -> None:
        state.tau[self.rotor_id] = self.tau
        state.mu_floor[self.rotor_id] = self.mu_floor


@dataclass
class DamageProfile:
    """Ordered collection of damage events to apply over an episode."""

    events: list[DamageEvent] = field(default_factory=list)
    seed: Optional[int] = None

    def get_pending_events(
        self, t_current: float, t_previous: float
    ) -> list[DamageEvent]:
        """Events triggering in the half-open window ``(t_previous, t_current]``."""
        return [
            e for e in self.events if t_previous < e.trigger_time <= t_current
        ]
