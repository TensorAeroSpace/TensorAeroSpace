"""Deprecated compatibility alias for :mod:`tensoraerospace.envs.geosat`."""

from __future__ import annotations

import warnings

from tensoraerospace.envs.geosat import GeoSatEnv as GeoSatEnv

warnings.warn(
    "tensoraerospace.envs.geostat is deprecated; "
    "use tensoraerospace.envs.geosat instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["GeoSatEnv"]
