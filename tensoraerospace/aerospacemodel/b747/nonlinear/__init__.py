"""Pure-numpy Boeing 747 nonlinear 6-DoF model.

State (NED frame, ZYX 321 Euler):
    [u, v, w,        # body-frame velocity, ft/s
     p, q, r,        # body-frame angular rates, rad/s
     phi, theta, psi,  # Euler angles, rad
     x_e, y_e, z_e]  # earth-frame position, ft

Control:
    [de, da, dr, dT]
        de — elevator deflection, rad (positive: trailing edge down)
        da — aileron deflection, rad (positive: right roll)
        dr — rudder deflection, rad (positive: right yaw)
        dT — throttle setting [0, 1] (mapped to nominal cruise thrust by
              a configurable engine model)

Aerodynamic and inertial parameters are sourced from NASA CR-2144 §IX
(Heffley & Jewell 1972) — see :mod:`.params`, :mod:`.flight_conditions`,
:mod:`.derivatives`.
"""

from .flight_conditions import B747_FLIGHT_CONDITIONS, B747FlightCondition
from .initial import default_state, initial_state_from_fc, set_initial_state
from .model import NonlinearB747
from .params import B747Configuration, B747Parameters, default_parameters
from .trim import TrimResult, trim

__all__ = [
    "B747_FLIGHT_CONDITIONS",
    "B747Configuration",
    "B747FlightCondition",
    "B747Parameters",
    "NonlinearB747",
    "TrimResult",
    "default_parameters",
    "default_state",
    "initial_state_from_fc",
    "set_initial_state",
    "trim",
]
