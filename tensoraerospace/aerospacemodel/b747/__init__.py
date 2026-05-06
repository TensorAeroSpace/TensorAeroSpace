"""Boeing 747 aerospace models.

* :class:`tensoraerospace.aerospacemodel.b747.linear.LongitudinalB747` —
  legacy single-trim-point linear state-space model (longitudinal).
* :class:`tensoraerospace.aerospacemodel.b747.nonlinear.NonlinearB747` —
  full 6-DoF nonlinear model from NASA CR-2144 (Heffley & Jewell 1972).
"""

from .linear import LongitudinalB747
from .nonlinear import NonlinearB747

__all__ = ["LongitudinalB747", "NonlinearB747"]
