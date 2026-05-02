"""Deprecated compatibility alias for the corrected linear longitudinal F-16 path."""

from __future__ import annotations

import warnings

from tensoraerospace.envs.f16.linear_longitudinal import (
    MODEL_STATE_ORDER as MODEL_STATE_ORDER,
)
from tensoraerospace.envs.f16.linear_longitudinal import (
    LinearLongitudinalF16 as LinearLongitudinalF16,
)

warnings.warn(
    "tensoraerospace.envs.f16.linear_longitudial is deprecated; "
    "use tensoraerospace.envs.f16.linear_longitudinal instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LinearLongitudinalF16", "MODEL_STATE_ORDER"]
