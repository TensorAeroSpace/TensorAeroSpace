"""F-16 Fighting Falcon Gymnasium environments.

This package bundles Gymnasium-compatible environments built on top of the
F-16 models provided in :mod:`tensoraerospace.aerospacemodel`.
"""

from tensoraerospace.envs.f16.linear_longitudial import (
    LinearLongitudinalF16 as LinearLongitudinalF16,
)
from tensoraerospace.envs.f16.nonlinear_angular import (
    NonlinearAngularF16 as NonlinearAngularF16,
)
from tensoraerospace.envs.f16.nonlinear_longitudinal import (
    NonlinearLongitudinalF16 as NonlinearLongitudinalF16,
)
