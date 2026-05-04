"""Ready-made damage profiles matching canonical FTC scenarios."""

from __future__ import annotations

from .events import (
    DamageProfile,
    MotorEfficiencyDecay,
    RotorDamageEvent,
    RotorLossEvent,
)

# Lanzon 2015: complete loss of motor 1 at t = 5 s.
LANZON_M1_LOSS = DamageProfile(
    events=[RotorLossEvent(trigger_time=5.0, rotor_id=0, label="m1_full_loss")]
)

# Lu 2019: 50% effectiveness loss on motor 1 at t = 5 s.
LU_M1_50PCT_LOSS = DamageProfile(
    events=[RotorDamageEvent(trigger_time=5.0, rotor_id=0, mu=0.5, label="m1_50pct")]
)

# Gradual wear: motor 3 starts decaying at t = 2 s with tau = 8 s,
# eventually reaches mu_floor = 0.3 (70% effectiveness loss).
WEAR_DEGRADATION_M3 = DamageProfile(
    events=[
        MotorEfficiencyDecay(
            trigger_time=2.0,
            rotor_id=2,
            tau=8.0,
            mu_floor=0.3,
            label="m3_wear_decay",
        )
    ]
)
