"""Deprecated compatibility alias for the corrected longitudinal F-16 initial module."""

from __future__ import annotations

import warnings

from .initial import initial_state as initial_state
from .initial import initial_state_dict as initial_state_dict
from .initial import set_initial_state as set_initial_state

warnings.warn(
    "tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.inital is deprecated; "
    "use tensoraerospace.aerospacemodel.f16.nonlinear.longitudinal.initial instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["initial_state", "initial_state_dict", "set_initial_state"]
