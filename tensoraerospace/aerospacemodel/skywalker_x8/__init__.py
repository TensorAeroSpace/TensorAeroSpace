"""Skywalker X8 small fixed-wing UAV models."""

from .nonlinear import (
    NonlinearSkywalkerX8,
    SkywalkerX8Parameters,
    set_initial_state,
    trim,
)

__all__ = [
    "NonlinearSkywalkerX8",
    "SkywalkerX8Parameters",
    "set_initial_state",
    "trim",
]
