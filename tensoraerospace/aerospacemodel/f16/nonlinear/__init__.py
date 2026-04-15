"""Pure-numpy F-16 nonlinear models (longitudinal + angular)."""

__all__ = ["LongitudinalF16", "AngularF16"]


def __getattr__(name):
    if name == "LongitudinalF16":
        from .longitudinal import LongitudinalF16
        return LongitudinalF16
    if name == "AngularF16":
        from .angular import AngularF16
        return AngularF16
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
