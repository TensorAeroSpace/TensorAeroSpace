"""Default parameters for the F-16 6-DoF angular model.

Mirrors angular/matlab_code/airplane_parameters.m line by line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..damage.geometry import BaseGeometry
    from ..damage.state import DamageState

_L = 0.0065
_R = 287.0531
_T0 = 288.15
_RHO0 = 1.225


def _isa_dynamic_pressure(altitude_m: float, velocity_mps: float, g: float) -> float:
    T = _T0 - _L * altitude_m
    rho = _RHO0 * (T / _T0) ** (g / (_L * _R) - 1.0)
    return float(0.5 * rho * velocity_mps**2)


@dataclass
class F16AngularParameters:
    m: float = 9295.44
    l: float = 9.144  # noqa: E741 - legacy MATLAB notation used by callers
    S: float = 27.87
    bA: float = 3.45
    # Aerodynamic reference geometry used by the coefficient tables. Damage
    # may change the effective geometry above, but the F-16 aero tables and
    # damage coefficient deltas are normalized to the intact aircraft.
    S_ref: float = field(init=False)
    l_ref: float = field(init=False)
    bA_ref: float = field(init=False)
    Jx: float = 12874.8
    Jy: float = 85552.1
    Jz: float = 75673.6
    Jxy: float = 1331.4
    Jyz: float = 0.0
    Jxz: float = 0.0
    rcgx: float = field(init=False)
    hEx: float = 0.0

    # Split-stab differential command (rad). Set by AngularF16.run_step
    # when split_stab=True; 0.0 means symmetric (legacy) operation.
    delta_stab_cmd: float = 0.0
    # Differential-stabilator roll-moment gain (1/rad). Calibrated for the
    # F-16 stabilator to give realistic roll rates ~30 deg/s for δ=10 deg.
    delta_stab_roll_gain: float = 0.6

    # Thrust (N). When the model is configured with thrust_mode="constant"
    # (default), T_thrust is the fixed propulsive force applied along the
    # body x-axis. When thrust_mode="control", the runtime thrust input is
    # clipped to [0, T_max_thrust]. Default ≈ 10 kN balances drag at the
    # default trim point (V=120 m/s, Oy=3000 m, alpha≈5°). Real F-16
    # F100 engine: ~129 kN max with afterburner.
    T_thrust: float = 10000.0
    T_max_thrust: float = 130000.0
    # Runtime override set by AngularF16.run_step when thrust_mode is
    # "control"; the ODE reads this in place of T_thrust. Defaults to NaN
    # so we can detect "not set yet" → fall back to T_thrust.
    T_active: float = field(init=False, default=float("nan"))

    # Damage subsystem hooks (set by AngularF16.run_step before each
    # integrator step). None = healthy aircraft. The damage_state and
    # damage_geometry are passed through params (rather than as ODE args)
    # to keep the ODE arity stable for legacy callers.
    damage_state: DamageState | None = None
    damage_geometry: BaseGeometry | None = None

    Tstab: float = 0.03
    Xistab: float = 0.707
    maxabsstab: float = field(default_factory=lambda: math.radians(25))
    maxabsdstab: float = field(default_factory=lambda: math.radians(60))

    Tail: float = 0.02
    Xiail: float = 0.707
    maxabsail: float = field(default_factory=lambda: math.radians(21.5))
    maxabsdail: float = field(default_factory=lambda: math.radians(80))

    Tdir: float = 0.03
    Xidir: float = 0.707
    maxabsdir: float = field(default_factory=lambda: math.radians(30))
    maxabsddir: float = field(default_factory=lambda: math.radians(120))

    lef: float = 0.0
    sb: float = 0.0
    g: float = 9.80665
    Oy: float = 3000.0
    V: float = 120.0
    q: float = field(init=False)

    def __post_init__(self) -> None:
        self.S_ref = float(self.S)
        self.l_ref = float(self.l)
        self.bA_ref = float(self.bA)
        self.rcgx = -0.05 * self.bA
        self.q = _isa_dynamic_pressure(self.Oy, self.V, self.g)


def default_parameters() -> F16AngularParameters:
    return F16AngularParameters()
