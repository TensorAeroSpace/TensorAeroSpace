"""Default parameters for the F-16 longitudinal nonlinear model.

Mirrors longitudinal/matlab_code/airplane_parameters.m line by line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_L = 0.0065
_R = 287.0531
_T0 = 288.15
_RHO0 = 1.225


def _isa_dynamic_pressure(altitude_m: float, velocity_mps: float, g: float) -> float:
    T = _T0 - _L * altitude_m
    rho = _RHO0 * (T / _T0) ** (g / (_L * _R) - 1.0)
    return 0.5 * rho * velocity_mps ** 2


@dataclass
class F16LongParameters:
    m: float = 9295.44
    S: float = 27.87
    bA: float = 3.45
    Jz: float = 75673.6
    rcgx: float = field(init=False)
    Tstab: float = 0.03
    Xistab: float = 0.707
    maxabsstab: float = field(default_factory=lambda: math.radians(25))
    maxabsdstab: float = field(default_factory=lambda: math.radians(60))
    lef: float = 0.0
    sb: float = 0.0
    g: float = 9.80665
    Oy: float = 3000.0
    V: float = 150.0
    q: float = field(init=False)

    def __post_init__(self) -> None:
        self.rcgx = -0.05 * self.bA
        self.q = _isa_dynamic_pressure(self.Oy, self.V, self.g)


def default_parameters() -> F16LongParameters:
    return F16LongParameters()
