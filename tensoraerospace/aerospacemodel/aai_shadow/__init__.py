"""AAI RQ-7 Shadow class-II tactical UAV models."""

from .nonlinear import (
    AAIShadowParameters,
    NonlinearAAIShadow,
    set_initial_state,
    trim,
)

__all__ = [
    "AAIShadowParameters",
    "NonlinearAAIShadow",
    "set_initial_state",
    "trim",
]
