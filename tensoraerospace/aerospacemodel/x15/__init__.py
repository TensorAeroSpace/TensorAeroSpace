"""North American X-15 aerospaceplane models.

This package exposes both the **linear** and **nonlinear** X-15 models:

* :class:`tensoraerospace.aerospacemodel.x15.LongitudinalX15` — the
  classic single-trim-point linear longitudinal state-space model in
  FPS units, originally distributed as
  ``tensoraerospace/aerospacemodel/x15.py``. Re-exported at the
  package root for backward compatibility.
* :class:`tensoraerospace.aerospacemodel.x15.NonlinearX15` — the new
  Mach-tabulated 6-DoF model with XLR99 rocket engine and
  variable-mass dynamics. Valid for M ∈ [0.4, 6.7], h ∈ [0, 250 kft].
"""

from .linear import LongitudinalX15
from .nonlinear import (
    X15_FLIGHT_CONDITIONS,
    NonlinearX15,
    X15Configuration,
    X15FlightCondition,
    X15Parameters,
    initial_state_from_fc,
    set_initial_state,
    trim,
)

__all__ = [
    "LongitudinalX15",
    "NonlinearX15",
    "X15Configuration",
    "X15FlightCondition",
    "X15Parameters",
    "X15_FLIGHT_CONDITIONS",
    "initial_state_from_fc",
    "set_initial_state",
    "trim",
]
