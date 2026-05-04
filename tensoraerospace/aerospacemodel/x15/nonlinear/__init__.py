"""Nonlinear hypersonic 6-DoF X-15 model.

Mach-tabulated aerodynamics from NASA TM X-1669 (Walker & Wolowicz),
XLR99 rocket-engine model with variable mass, valid envelope:

    M ∈ [0.4, 6.7],  h ∈ [0, 250 000] ft.

See :mod:`tensoraerospace.aerospacemodel.b747.nonlinear` for the
companion airliner model and the parent module docs for the full
list of aerospaceplane models.
"""

from .aero import AeroForces, AeroState, x15_aero
from .dynamics import x15_ode_6dof
from .engine import XLR99Engine, xlr99_thrust
from .flight_conditions import (
    X15_FLIGHT_CONDITIONS,
    X15FlightCondition,
    get_flight_condition,
)
from .initial import (
    STATE_DIM,
    STATE_LIST,
    default_state,
    initial_state_from_fc,
    set_initial_state,
)
from .model import NonlinearX15
from .params import (
    X15Configuration,
    X15Parameters,
    X15ParametersSI,
    default_parameters,
    isa_density_slug_ft3,
    isa_speed_of_sound_ft_s,
    to_si,
)
from .trim import TrimResult, level_trim, trim

__all__ = [
    "AeroForces",
    "AeroState",
    "NonlinearX15",
    "STATE_DIM",
    "STATE_LIST",
    "TrimResult",
    "X15Configuration",
    "X15FlightCondition",
    "X15Parameters",
    "X15ParametersSI",
    "X15_FLIGHT_CONDITIONS",
    "XLR99Engine",
    "default_parameters",
    "default_state",
    "get_flight_condition",
    "initial_state_from_fc",
    "isa_density_slug_ft3",
    "isa_speed_of_sound_ft_s",
    "level_trim",
    "set_initial_state",
    "to_si",
    "trim",
    "x15_aero",
    "x15_ode_6dof",
    "xlr99_thrust",
]
