"""Mutable runtime state describing what is currently damaged."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .geometry import BaseGeometry

ControlMode = Literal[
    "healthy", "efficiency_loss", "jam", "free_floating", "lost"
]
_VALID_CONTROL_MODES = (
    "healthy", "efficiency_loss", "jam", "free_floating", "lost",
)


@dataclass
class ControlFailure:
    mode: ControlMode = "healthy"
    efficiency: float = 1.0
    jam_position_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in _VALID_CONTROL_MODES:
            raise ValueError(
                f"ControlFailure.mode must be one of "
                f"{_VALID_CONTROL_MODES}; got {self.mode!r}"
            )
        if not (0.0 <= self.efficiency <= 1.0):
            raise ValueError(
                f"efficiency must be in [0,1]; got {self.efficiency}"
            )


@dataclass
class EngineState:
    thrust_factor: float = 1.0
    hard_failure: bool = False


@dataclass
class StructuralState:
    extra_mass_delta_kg: float = 0.0
    extra_cg_shift_m: tuple = (0.0, 0.0, 0.0)
    extra_inertia_delta: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class DamageState:
    section_loss: dict[str, float]
    control_failures: dict[str, ControlFailure]
    engine: EngineState = field(default_factory=EngineState)
    structural: StructuralState = field(default_factory=StructuralState)

    @classmethod
    def healthy(cls, geometry: BaseGeometry) -> "DamageState":
        return cls(
            section_loss={name: 0.0 for name in geometry.section_names()},
            control_failures={},
            engine=EngineState(),
            structural=StructuralState(),
        )

    def set_section_loss(self, section_name: str, fraction: float) -> None:
        clamped = max(0.0, min(1.0, float(fraction)))
        self.section_loss[section_name] = clamped

    def set_control_failure(self, surface: str, failure: ControlFailure) -> None:
        if failure.mode == "healthy":
            self.control_failures.pop(surface, None)
        else:
            self.control_failures[surface] = failure

    def snapshot(self) -> dict:
        """Return a deep-copied, JSON-friendly view."""
        return {
            "section_loss": dict(self.section_loss),
            "control_failures": {
                name: {
                    "mode": cf.mode,
                    "efficiency": cf.efficiency,
                    "jam_position_rad": cf.jam_position_rad,
                }
                for name, cf in self.control_failures.items()
            },
            "engine": {
                "thrust_factor": self.engine.thrust_factor,
                "hard_failure": self.engine.hard_failure,
            },
            "structural": {
                "extra_mass_delta_kg": self.structural.extra_mass_delta_kg,
                "extra_cg_shift_m": tuple(self.structural.extra_cg_shift_m),
                "extra_inertia_delta": tuple(self.structural.extra_inertia_delta),
            },
        }
