"""Ready-made damage profiles matching canonical FTC scenarios for B-747."""

from __future__ import annotations

from ..params import B747Configuration
from .events import (
    DamageProfile,
    EngineFailureEvent,
    FlapJamEvent,
    SurfaceEffectivenessDecay,
    SurfaceEffectivenessEvent,
    SurfaceJamEvent,
)


# 50% loss of elevator effectiveness at t = 5 s — Lu 2019 / Wang 2019 style.
ELEVATOR_50PCT_LOSS = DamageProfile(events=[
    SurfaceEffectivenessEvent(
        trigger_time=5.0, surface="elevator", mu=0.5,
        label="elevator_50pct_loss",
    ),
])


# Hard-over: elevator jammed at a small nose-up deflection.
ELEVATOR_JAMMED_NOSE_UP = DamageProfile(events=[
    SurfaceJamEvent(
        trigger_time=10.0, surface="elevator", jam_value=-0.0349,  # -2 deg
        label="elevator_jammed_-2deg",
    ),
])


# Aileron failure: total loss at t = 8 s.
AILERON_TOTAL_LOSS = DamageProfile(events=[
    SurfaceEffectivenessEvent(
        trigger_time=8.0, surface="aileron", mu=0.0,
        label="aileron_total_loss",
    ),
])


# Hydraulic leak: rudder authority decays exponentially toward 30%.
RUDDER_HYDRAULIC_LEAK = DamageProfile(events=[
    SurfaceEffectivenessDecay(
        trigger_time=2.0, surface="rudder",
        tau=8.0, mu_floor=0.3,
        label="rudder_hydraulic_decay",
    ),
])


# Engine cluster failure: throttle stuck at idle (τ=0 because the throttle
# channel passes directly to the engine model — jamming the input is enough).
ENGINE_FLAMEOUT = DamageProfile(events=[
    SurfaceJamEvent(
        trigger_time=15.0, surface="throttle", jam_value=0.0,
        label="engine_flameout",
    ),
])


# Single-engine failure: outer-left engine #1 dies at t = 10 s. The remaining
# three engines produce 75% of nominal thrust plus a strong asymmetric yaw
# moment toward the dead engine (a classic V_MC engine-out scenario).
LEFT_OUTER_ENGINE_FAILURE = DamageProfile(events=[
    EngineFailureEvent(
        trigger_time=10.0, engine_id=1, thrust_fraction=0.0,
        label="left_outer_engine_flameout",
    ),
])


# Two-engines-out on the same wing: both left engines fail simultaneously at
# t = 10 s — worst-case asymmetry (≈ 50% thrust, large yaw moment toward
# the dead side). Used for FTC controller stress tests.
LEFT_TWO_ENGINES_OUT = DamageProfile(events=[
    EngineFailureEvent(
        trigger_time=10.0, engine_id=1, thrust_fraction=0.0,
        label="left_outer_engine_flameout",
    ),
    EngineFailureEvent(
        trigger_time=10.0, engine_id=2, thrust_fraction=0.0,
        label="left_inner_engine_flameout",
    ),
])


# Flap retraction failure: aircraft commanded back to NOMINAL (clean) but
# flaps mechanically stuck at LANDING (30°). At t = 5 s the aerodynamic
# build switches to the LANDING configuration regardless of params.config —
# higher CL and CD, lower V_max.
FLAPS_JAMMED_LANDING = DamageProfile(events=[
    FlapJamEvent(
        trigger_time=5.0, jammed_config=B747Configuration.LANDING,
        label="flaps_jammed_at_30deg_landing",
    ),
])


# Flap extension failure: pilot commands flaps for approach but they fail to
# deploy past the clean position. Useful inverse-scenario to FLAPS_JAMMED_LANDING
# (cruise → approach with no flap extension).
FLAPS_JAMMED_RETRACTED = DamageProfile(events=[
    FlapJamEvent(
        trigger_time=5.0, jammed_config=B747Configuration.NOMINAL,
        label="flaps_jammed_retracted",
    ),
])
