"""Quadrotor / multirotor UAV plant models."""

from .allocation import XConfigAllocator, default_allocator

__all__ = ["NonlinearQuadrotor", "XConfigAllocator", "default_allocator"]


def __getattr__(name):
    if name == "NonlinearQuadrotor":
        from .nonlinear import NonlinearQuadrotor

        return NonlinearQuadrotor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
