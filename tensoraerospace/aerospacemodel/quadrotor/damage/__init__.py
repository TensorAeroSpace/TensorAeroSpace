"""Rotor-level damage subsystem for the nonlinear quadrotor.

Reproduces the failure modes used in the canonical FTC-quadrotor
literature (Lu 2019 ISMC, Wang 2019 terminal SMC, Lanzon 2015 complete
rotor failure):

- :class:`RotorDamageEvent` — instantaneous multiplicative effectiveness
  loss (Lu 2019).
- :class:`RotorLossEvent` — complete rotor stop (Lanzon 2015 — the
  "extreme" case).
- :class:`MotorEfficiencyDecay` — exponential degradation over a time
  constant (gradual wear / partial blade damage).

Damage acts at the **rotor level**: each motor has an effectiveness
coefficient :math:`\\mu_i \\in [0, 1]` such that the actual rotor speed²
delivered is :math:`\\omega^2_{i,\\text{eff}} = \\mu_i \\cdot \\omega^2_{i,\\text{cmd}}`.
After the multiplicative scaling, the env mixes back to virtual
:math:`(T_{\\text{eff}}, \\tau_{\\text{eff}})` and integrates the ODE.
"""

from .events import (
    DamageEvent,
    DamageProfile,
    MotorEfficiencyDecay,
    RotorDamageEvent,
    RotorLossEvent,
)
from .manager import RotorDamageManager
from .presets import (
    LANZON_M1_LOSS,
    LU_M1_50PCT_LOSS,
    WEAR_DEGRADATION_M3,
)
from .state import RotorDamageState

__all__ = [
    "DamageEvent",
    "DamageProfile",
    "MotorEfficiencyDecay",
    "RotorDamageEvent",
    "RotorLossEvent",
    "RotorDamageManager",
    "RotorDamageState",
    "LANZON_M1_LOSS",
    "LU_M1_50PCT_LOSS",
    "WEAR_DEGRADATION_M3",
]
