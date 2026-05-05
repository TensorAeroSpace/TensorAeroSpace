"""Nonlinear 6-DoF Skywalker X8 small UAV model.

Aerodynamic data: peer-reviewed flight-test identification from
Løw-Hansen et al., CEAS Aeronautical Journal (2025).
DOI: 10.1007/s13272-025-00816-3
"""

from .aero import AeroForces, AeroState, x8_aero
from .dynamics import x8_ode_6dof
from .engine import X8Propeller, x8_thrust
from .initial import STATE_DIM, STATE_LIST, default_state, set_initial_state
from .model import NonlinearSkywalkerX8
from .params import (
    SkywalkerX8Parameters,
    default_parameters,
    isa_density_kg_m3,
    isa_speed_of_sound_m_s,
)
from .trim import TrimResult, trim

__all__ = [
    "AeroForces",
    "AeroState",
    "NonlinearSkywalkerX8",
    "STATE_DIM",
    "STATE_LIST",
    "SkywalkerX8Parameters",
    "TrimResult",
    "X8Propeller",
    "default_parameters",
    "default_state",
    "isa_density_kg_m3",
    "isa_speed_of_sound_m_s",
    "set_initial_state",
    "trim",
    "x8_aero",
    "x8_ode_6dof",
    "x8_thrust",
]
