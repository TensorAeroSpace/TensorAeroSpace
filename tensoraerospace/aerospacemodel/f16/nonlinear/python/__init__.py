"""Pure-numpy F-16 nonlinear models (longitudinal + angular).

Side-by-side replacement for the matlab.engine-backed implementations in
`tensoraerospace.aerospacemodel.f16.nonlinear.{longitudinal,angular}`.
"""

from .angular import AngularF16
from .longitudinal import LongitudinalF16

__all__ = ["LongitudinalF16", "AngularF16"]
