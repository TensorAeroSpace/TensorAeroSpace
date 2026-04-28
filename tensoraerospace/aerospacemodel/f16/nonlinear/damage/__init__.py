"""Damage modeling subsystem for the nonlinear F-16 model."""

from .events import DamageEvent, DamageProfile
from .geometry import AeroSection, BaseGeometry
from .manager import DamageManager
from .presets import (
    BIRDSTRIKE_COMPOUND,
    ELEVATOR_JAM_NEUTRAL,
    ELEVATOR_JAM_PITCH_UP,
    ENGINE_FLAMEOUT,
    RUDDER_LOST,
    WING_STRIKE_LEFT_HALF,
    WING_STRIKE_LEFT_TIP,
    load_f16_geometry,
)
from .state import (
    ControlFailure,
    DamageState,
    EngineState,
    StructuralState,
)

__all__ = [
    "AeroSection", "BaseGeometry",
    "DamageState", "ControlFailure", "EngineState", "StructuralState",
    "DamageEvent", "DamageProfile",
    "DamageManager",
    "load_f16_geometry",
    "WING_STRIKE_LEFT_TIP", "WING_STRIKE_LEFT_HALF",
    "ELEVATOR_JAM_NEUTRAL", "ELEVATOR_JAM_PITCH_UP",
    "RUDDER_LOST", "ENGINE_FLAMEOUT", "BIRDSTRIKE_COMPOUND",
]
