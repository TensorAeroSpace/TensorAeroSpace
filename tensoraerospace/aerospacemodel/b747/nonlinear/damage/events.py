"""Damage events for the B-747 nonlinear model.

Five concrete event classes:

* :class:`SurfaceEffectivenessEvent` — multiplicative effectiveness loss
  on a control surface (elevator / aileron / rudder / throttle), e.g.
  partial actuator failure or icing.
* :class:`SurfaceJamEvent` — surface is mechanically locked at a
  commanded deflection; the agent's command is ignored on that channel.
* :class:`SurfaceEffectivenessDecay` — exponentially decaying
  effectiveness with a finite time constant. Useful for fluid leakage,
  thermal degradation, or progressive jamming scenarios.
* :class:`EngineFailureEvent` — single (or partial) engine flameout.
  Engines are numbered 1–4 from left wing to right wing; an asymmetric
  failure produces a yaw moment because the surviving engines pull on
  one side only.
* :class:`FlapJamEvent` — high-lift devices mechanically locked in one
  position. Re-routes the aerodynamic-derivative selection in
  :func:`~tensoraerospace.aerospacemodel.b747.nonlinear.aero.b747_aero`
  to the jammed configuration regardless of the requested config.

Events compose into a :class:`DamageProfile`, replayed by the
:class:`~tensoraerospace.aerospacemodel.b747.nonlinear.damage.\
manager.B747DamageManager`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from ..params import B747Configuration
from .state import ENGINES, SURFACES, B747DamageState


@dataclass(frozen=True)
class DamageEvent:
    """Base class for B-747 damage events."""

    trigger_time: float
    surface: str = "elevator"
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be ≥ 0; got {self.trigger_time}")
        if self.surface not in SURFACES:
            raise ValueError(
                f"surface must be one of {SURFACES}; got {self.surface!r}"
            )

    def apply(self, state: B747DamageState) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class SurfaceEffectivenessEvent(DamageEvent):
    """Set effectiveness ``mu`` of one control surface to a fixed value.

    ``mu = 0`` ⇒ total loss of effectiveness on that channel
    (the surface still moves but has no aerodynamic effect).
    """

    mu: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.mu <= 1.0:
            raise ValueError(f"mu must be in [0, 1]; got {self.mu}")

    def apply(self, state: B747DamageState) -> None:
        state.mu[self.surface] = float(self.mu)


@dataclass(frozen=True)
class SurfaceJamEvent(DamageEvent):
    """Mechanically lock a surface at a fixed deflection (radians).

    For ``surface="throttle"``, ``jam_value`` is in [0, 1].
    """

    jam_value: float = 0.0

    def apply(self, state: B747DamageState) -> None:
        state.jam[self.surface] = float(self.jam_value)


@dataclass(frozen=True)
class SurfaceEffectivenessDecay(DamageEvent):
    """Schedule exponential effectiveness decay starting at ``trigger_time``.

    Until the trigger fires, ``mu`` stays at its current value. Once
    fired, the manager advances ``mu(t) = floor + (mu - floor)·exp(-dt/τ)``
    on every integrator tick.
    """

    tau: float = 5.0
    mu_floor: float = 0.3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0:
            raise ValueError(f"tau must be > 0; got {self.tau}")
        if not 0.0 <= self.mu_floor <= 1.0:
            raise ValueError(f"mu_floor must be in [0, 1]; got {self.mu_floor}")

    def apply(self, state: B747DamageState) -> None:
        state.tau[self.surface] = float(self.tau)
        state.mu_floor[self.surface] = float(self.mu_floor)


@dataclass(frozen=True)
class EngineFailureEvent:
    """Single-engine flameout (or partial failure) on a numbered engine.

    Engines are numbered 1–4 left-to-right (1, 2 = left wing; 3, 4 =
    right wing). When ``thrust_fraction = 0``, the engine is fully
    flamed out. Asymmetric failures produce a yaw moment because the
    surviving engines pull on one side; the moment is computed at
    runtime by :func:`~tensoraerospace.aerospacemodel.b747.nonlinear.\
engine.jt9d_thrust_with_asymmetry` from the spanwise engine
    positions.
    """

    trigger_time: float
    engine_id: int = 1
    thrust_fraction: float = 0.0
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be ≥ 0; got {self.trigger_time}")
        if self.engine_id not in ENGINES:
            raise ValueError(
                f"engine_id must be one of {ENGINES}; got {self.engine_id}"
            )
        if not 0.0 <= self.thrust_fraction <= 1.0:
            raise ValueError(
                f"thrust_fraction must be in [0, 1]; got {self.thrust_fraction}"
            )

    def apply(self, state: B747DamageState) -> None:
        state.engines_mu[self.engine_id] = float(self.thrust_fraction)


@dataclass(frozen=True)
class FlapJamEvent:
    """High-lift devices mechanically locked in a fixed configuration.

    Sets ``B747DamageState.flap_jam_config`` to one of the three
    discrete CR-2144 configurations (NOMINAL = retracted clean, 20° =
    ``POWER_APPROACH``, 30° = ``LANDING``). Once set, the aero build
    selects derivatives for *that* configuration regardless of the
    plane's requested config — capturing the scenario "pilot wants to
    retract flaps but they are stuck at 30°".
    """

    trigger_time: float
    jammed_config: B747Configuration = B747Configuration.LANDING
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trigger_time < 0:
            raise ValueError(f"trigger_time must be ≥ 0; got {self.trigger_time}")
        if not isinstance(self.jammed_config, B747Configuration):
            raise ValueError(
                f"jammed_config must be a B747Configuration; got "
                f"{type(self.jammed_config).__name__}"
            )

    def apply(self, state: B747DamageState) -> None:
        state.flap_jam_config = self.jammed_config


# Anything with a ``trigger_time`` and ``apply(state)`` method qualifies as
# a damage event. We model that as a Union so DamageProfile can hold any
# subset.
AnyDamageEvent = Union[DamageEvent, EngineFailureEvent, FlapJamEvent]


@dataclass
class DamageProfile:
    """Collection of damage events scheduled by absolute time."""

    events: list[AnyDamageEvent] = field(default_factory=list)

    def get_pending_events(
        self, t_current: float, t_previous: float
    ) -> list[AnyDamageEvent]:
        """Events that fall in the half-open window ``(t_previous, t_current]``."""
        return [
            ev for ev in self.events if t_previous < ev.trigger_time <= t_current
        ]
