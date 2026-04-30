"""Damage presets matching the fault scenarios used in Ul Haq et al. 2026.

The paper sweeps an inboard / outboard elevon effectiveness from 100 % down
to 0 % in five increments (1.0, 0.75, 0.5, 0.25, 0). On the F-16 the
closest analogue is the symmetric stabilator and the ailerons; we provide
stand-alone presets for both, plus a complete-loss preset on the rudder.
"""

from __future__ import annotations

from typing import Sequence

from .events import DamageEvent, DamageProfile


def stab_efficiency_step(
    t_inject: float = 5.0,
    mu: float = 0.25,
    surface: str = "stab_left",
) -> DamageProfile:
    """Single step efficiency-loss event on the (left) stabilator."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must be in [0, 1]")
    return DamageProfile(
        events=[
            DamageEvent(
                trigger_time=float(t_inject),
                event_type="control_failure",
                payload={
                    "surface": surface,
                    "mode": "efficiency_loss",
                    "efficiency": float(mu),
                },
                label=f"stab_eff_loss_{int(round(mu * 100))}",
            ),
        ]
    )


def aileron_efficiency_loss_schedule(
    t_start: float = 2.0,
    dt_between: float = 1.0,
    levels: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    surface: str = "aileron_left",
) -> DamageProfile:
    """Schedule of progressive efficiency loss matching the paper sweep."""
    events = []
    for k, mu in enumerate(levels):
        events.append(
            DamageEvent(
                trigger_time=float(t_start + k * dt_between),
                event_type="control_failure",
                payload={
                    "surface": surface,
                    "mode": "efficiency_loss",
                    "efficiency": float(mu),
                },
                label=f"aileron_eff_{int(round(mu * 100))}",
            )
        )
    return DamageProfile(events=events)


def rudder_total_loss(t_inject: float = 10.0) -> DamageProfile:
    """Complete loss of rudder — common worst-case in the paper."""
    return DamageProfile(
        events=[
            DamageEvent(
                trigger_time=float(t_inject),
                event_type="control_failure",
                payload={"surface": "rudder", "mode": "lost"},
                label="rudder_lost",
            ),
        ]
    )
