"""Parametric geometry primitives for damage modeling.

Each aircraft is described as a list of AeroSection objects. Sections aggregate
into a BaseGeometry, which serves as the source of truth for mass, inertia,
and aerodynamic contributions. Damage is applied by scaling per-section loss
fractions (see state.py and recompute.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Inertia4 = Tuple[float, float, float, float]  # Ixx, Iyy, Izz, Ixz


@dataclass(frozen=True)
class AeroSection:
    """A discrete piece of the aircraft.

    Coordinates are in the body-fixed frame:
      x — forward (positive towards the nose)
      y — right (positive towards the right wingtip)
      z — down

    `span_position` is the y-coordinate of the section's aero centre
    (signed: negative=left, positive=right).
    """

    name: str
    side: Literal["left", "right", "center"]
    type: Literal["wing", "stab", "vtail", "control", "fuselage"]

    area: float                     # m², projected area
    span_position: float            # m (signed, see above)
    chord: float                    # m, mean chord
    sweep: float                    # rad

    mass: float                     # kg
    cg_local: Vec3                  # body-frame cg of the section, m
    inertia_local: Inertia4         # Ixx, Iyy, Izz, Ixz about section cg

    cl_alpha_contribution: float    # 1/rad, this section's share of ∂Cy/∂α
    cd0_contribution: float         # this section's share of Cx0

    controls_input: Optional[str] = None      # "stab_left", "rudder", ...
    control_effectiveness: float = 1.0
    aero_x_arm: float = 0.0          # x-arm from aircraft CG to section's aero centre


@dataclass
class BaseGeometry:
    """Aggregate description of the un-damaged aircraft."""

    sections: list[AeroSection]

    def __post_init__(self) -> None:
        names = [s.name for s in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate section names")
        self._index: dict[str, AeroSection] = {s.name: s for s in self.sections}

    def section(self, name: str) -> AeroSection:
        try:
            return self._index[name]
        except KeyError as e:
            raise KeyError(f"No section named {name!r}") from e

    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]

    def total_wing_area(self) -> float:
        return sum(s.area for s in self.sections if s.type == "wing")

    def total_mass(self) -> float:
        return sum(s.mass for s in self.sections)

    def center_of_mass(self) -> np.ndarray:
        m_total = self.total_mass()
        if m_total <= 0:
            raise ValueError("Total mass is non-positive")
        x = sum(s.mass * s.cg_local[0] for s in self.sections) / m_total
        y = sum(s.mass * s.cg_local[1] for s in self.sections) / m_total
        z = sum(s.mass * s.cg_local[2] for s in self.sections) / m_total
        return np.array([x, y, z])
