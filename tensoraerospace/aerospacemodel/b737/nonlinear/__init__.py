"""Nonlinear 6-DoF Boeing 737 model.

Aerodynamic data sourced from the JSBSim 737 model, which builds on
Roskam Vol VI Appendix B (737-100 reference) and the original Boeing
wind-tunnel data published as NASA CR-114494 (Hanke 1971). Engine
model wraps a Mach/altitude-derated cluster compatible with both
Pratt & Whitney JT8D-9 (737-100/200) and CFM International CFM56-7B
(737-NG).

Two configurations:

* :attr:`B737Configuration.B737_100` — original 737, JSBSim defaults.
* :attr:`B737Configuration.B737_800` — 737-NG, CFM56-7B engines.
"""

from .aero import AeroForces, AeroState, b737_aero
from .dynamics import b737_ode_6dof
from .engine import (
    ENGINE_Y_POSITIONS_FT,
    B737Engine,
    b737_thrust,
    b737_thrust_with_asymmetry,
)
from .initial import STATE_DIM, STATE_LIST, default_state, set_initial_state
from .model import NonlinearB737
from .params import (
    B737Configuration,
    B737Parameters,
    B737ParametersSI,
    default_parameters,
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
    to_si,
)
from .trim import TrimResult, trim

__all__ = [
    "AeroForces",
    "AeroState",
    "B737Configuration",
    "B737Engine",
    "B737Parameters",
    "B737ParametersSI",
    "ENGINE_Y_POSITIONS_FT",
    "NonlinearB737",
    "STATE_DIM",
    "STATE_LIST",
    "TrimResult",
    "b737_aero",
    "b737_ode_6dof",
    "b737_thrust",
    "b737_thrust_with_asymmetry",
    "default_parameters",
    "default_state",
    "isa_density_slug_ft3",
    "isa_speed_of_sound_ft_s",
    "set_initial_state",
    "to_si",
    "trim",
]
