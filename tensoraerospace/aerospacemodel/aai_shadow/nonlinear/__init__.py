"""Nonlinear 6-DoF AAI RQ-7 Shadow tactical UAV model.

Class-II surveillance UAV: ~ 170 kg, 4.27 m wingspan, 38 hp rotary
pusher engine, V-tail. Aerodynamic derivatives synthesised from
class-II small-UAV literature (Beard & McLain Aerosonde +
NASA TM-2014-218686 + Roskam Vol VI V-tail mixing).
"""

from .aero import AeroForces, AeroState, shadow_aero
from .dynamics import shadow_ode_6dof
from .engine import shadow_thrust
from .initial import STATE_DIM, STATE_LIST, default_state, set_initial_state
from .model import NonlinearAAIShadow
from .params import (
    AAIShadowParameters,
    default_parameters,
    isa_density_kg_m3,
    isa_speed_of_sound_m_s,
)
from .trim import TrimResult, trim

__all__ = [
    "AAIShadowParameters",
    "AeroForces",
    "AeroState",
    "NonlinearAAIShadow",
    "STATE_DIM",
    "STATE_LIST",
    "TrimResult",
    "default_parameters",
    "default_state",
    "isa_density_kg_m3",
    "isa_speed_of_sound_m_s",
    "set_initial_state",
    "shadow_aero",
    "shadow_ode_6dof",
    "shadow_thrust",
    "trim",
]
