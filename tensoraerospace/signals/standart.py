"""Deprecated compatibility alias for :mod:`tensoraerospace.signals.standard`."""

from __future__ import annotations

import warnings

from .standard import chirp as chirp
from .standard import constant_line as constant_line
from .standard import damped_sinusoid as damped_sinusoid
from .standard import doublet as doublet
from .standard import exponential as exponential
from .standard import gaussian_pulse as gaussian_pulse
from .standard import multi_step as multi_step
from .standard import multisine as multisine
from .standard import pulse as pulse
from .standard import ramp as ramp
from .standard import sawtooth as sawtooth
from .standard import sinusoid as sinusoid
from .standard import sinusoid_vertical_shift as sinusoid_vertical_shift
from .standard import square_wave as square_wave
from .standard import triangular_wave as triangular_wave
from .standard import unit_step as unit_step

warnings.warn(
    "tensoraerospace.signals.standart is deprecated; "
    "use tensoraerospace.signals.standard instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "chirp",
    "constant_line",
    "damped_sinusoid",
    "doublet",
    "exponential",
    "gaussian_pulse",
    "multi_step",
    "multisine",
    "pulse",
    "ramp",
    "sawtooth",
    "sinusoid",
    "sinusoid_vertical_shift",
    "square_wave",
    "triangular_wave",
    "unit_step",
]
