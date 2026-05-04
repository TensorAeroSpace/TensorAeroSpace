"""B-747 control-surface, propulsion and configuration damage subsystem.

Mirrors the F-16 damage subsystem in
:mod:`tensoraerospace.aerospacemodel.f16.nonlinear.damage` and the
quadrotor rotor-damage subsystem in
:mod:`tensoraerospace.aerospacemodel.quadrotor.damage`.

Five event types:

* :class:`SurfaceEffectivenessEvent` — instantaneous multiplicative
  authority loss on a control surface.
* :class:`SurfaceJamEvent` — surface mechanically locked at a
  commanded value.
* :class:`SurfaceEffectivenessDecay` — exponentially decaying
  authority with finite time constant.
* :class:`EngineFailureEvent` — single-engine flameout (or partial
  failure) with asymmetric-thrust yaw moment.
* :class:`FlapJamEvent` — high-lift devices mechanically locked in
  one configuration.

Ready-made presets in :mod:`.presets`.
"""

from .events import (
    AnyDamageEvent,
    DamageEvent,
    DamageProfile,
    EngineFailureEvent,
    FlapJamEvent,
    SurfaceEffectivenessDecay,
    SurfaceEffectivenessEvent,
    SurfaceJamEvent,
)
from .manager import B747DamageManager
from .presets import (
    AILERON_TOTAL_LOSS,
    ELEVATOR_50PCT_LOSS,
    ELEVATOR_JAMMED_NOSE_UP,
    ENGINE_FLAMEOUT,
    FLAPS_JAMMED_LANDING,
    FLAPS_JAMMED_RETRACTED,
    LEFT_OUTER_ENGINE_FAILURE,
    LEFT_TWO_ENGINES_OUT,
    RUDDER_HYDRAULIC_LEAK,
)
from .state import ENGINES, SURFACES, B747DamageState

__all__ = [
    "AILERON_TOTAL_LOSS",
    "AnyDamageEvent",
    "B747DamageManager",
    "B747DamageState",
    "DamageEvent",
    "DamageProfile",
    "ELEVATOR_50PCT_LOSS",
    "ELEVATOR_JAMMED_NOSE_UP",
    "ENGINE_FLAMEOUT",
    "ENGINES",
    "EngineFailureEvent",
    "FLAPS_JAMMED_LANDING",
    "FLAPS_JAMMED_RETRACTED",
    "FlapJamEvent",
    "LEFT_OUTER_ENGINE_FAILURE",
    "LEFT_TWO_ENGINES_OUT",
    "RUDDER_HYDRAULIC_LEAK",
    "SURFACES",
    "SurfaceEffectivenessDecay",
    "SurfaceEffectivenessEvent",
    "SurfaceJamEvent",
]
