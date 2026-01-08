"""Gymnasium environments shipped with TensorAeroSpace.

This package exposes a set of Gymnasium-compatible environments for aerospace
control tasks (aircraft, rockets, satellites). Environments are re-exported
here for convenience, and are also registered in the top-level
``tensoraerospace`` package.
"""

from .b747 import ImprovedB747Env as ImprovedB747Env  # noqa: F401
from .b747 import LinearLongitudinalB747 as LinearLongitudinalB747
from .b747_vec_torch import (  # noqa: F401
    ImprovedB747VecEnvTorch as ImprovedB747VecEnvTorch,
)
from .comsat import ComSatEnv as ComSatEnv  # noqa: F401
from .comsat import ImprovedComSatEnv as ImprovedComSatEnv  # noqa: F401
from .elv import ImprovedELVEnv as ImprovedELVEnv
from .elv import (  # noqa: F401
    LinearLongitudinalELVRocket as LinearLongitudinalELVRocket,
)
from .f4c import F4CPitchEnvNormalized as F4CPitchEnvNormalized  # noqa: F401
from .f4c import LinearLongitudinalF4C as LinearLongitudinalF4C
from .f16.linear_longitudial import (  # noqa: F401
    LinearLongitudinalF16 as LinearLongitudinalF16,
)
from .geostat import GeoSatEnv as GeoSatEnv  # noqa: F401
from .lapan import ImprovedLAPANEnv as ImprovedLAPANEnv  # noqa: F401
from .lapan import LinearLongitudinalLAPAN as LinearLongitudinalLAPAN  # noqa: F401
from .rocket import ImprovedMissileEnv as ImprovedMissileEnv
from .rocket import (  # noqa: F401
    LinearLongitudinalMissileModel as LinearLongitudinalMissileModel,
)
from .uav import LinearLongitudinalUAV as LinearLongitudinalUAV  # noqa: F401
from .ultrastick import ImprovedUltrastickEnv as ImprovedUltrastickEnv  # noqa: F401
from .ultrastick import (  # noqa: F401
    LinearLongitudinalUltrastick as LinearLongitudinalUltrastick,
)
from .x15 import ImprovedX15Env as ImprovedX15Env  # noqa: F401
from .x15 import LinearLongitudinalX15 as LinearLongitudinalX15

__all__ = [
    "ImprovedB747Env",
    "ImprovedB747VecEnvTorch",
    "LinearLongitudinalB747",
    "ComSatEnv",
    "ImprovedComSatEnv",
    "LinearLongitudinalELVRocket",
    "ImprovedELVEnv",
    "F4CPitchEnvNormalized",
    "LinearLongitudinalF4C",
    "LinearLongitudinalF16",
    "GeoSatEnv",
    "LinearLongitudinalLAPAN",
    "LinearLongitudinalMissileModel",
    "ImprovedMissileEnv",
    "LinearLongitudinalUAV",
    "LinearLongitudinalUltrastick",
    "ImprovedUltrastickEnv",
    "ImprovedX15Env",
    "LinearLongitudinalX15",
]

# from .unity_env import get_plane_env, unity_discrete_env
