"""Boeing 737 aerospace models.

Currently exposes only the **nonlinear** 6-DoF model. (Unlike B-747,
the legacy linear-state-space module did not pre-exist for the 737.)
"""

from .nonlinear import (
    B737Configuration,
    B737Parameters,
    NonlinearB737,
    set_initial_state,
    trim,
)

__all__ = [
    "B737Configuration",
    "B737Parameters",
    "NonlinearB737",
    "set_initial_state",
    "trim",
]
