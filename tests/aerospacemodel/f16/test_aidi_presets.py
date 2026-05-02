"""AIDI-specific damage presets for the F-16."""

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageProfile,
    aileron_efficiency_loss_schedule,
    rudder_total_loss,
    stab_efficiency_step,
)


def test_stab_efficiency_step_returns_profile():
    profile = stab_efficiency_step(t_inject=5.0, mu=0.25)
    assert isinstance(profile, DamageProfile)
    assert len(profile.events) == 1
    ev = profile.events[0]
    assert ev.trigger_time == 5.0
    assert ev.event_type == "control_failure"
    assert ev.payload["mode"] == "efficiency_loss"
    assert ev.payload["efficiency"] == 0.25


def test_aileron_schedule_emits_five_decreasing_events():
    profile = aileron_efficiency_loss_schedule(
        t_start=2.0,
        dt_between=1.0,
        levels=(1.0, 0.75, 0.5, 0.25, 0.0),
    )
    assert isinstance(profile, DamageProfile)
    assert len(profile.events) == 5
    times = [e.trigger_time for e in profile.events]
    assert times == [2.0, 3.0, 4.0, 5.0, 6.0]
    effs = [e.payload["efficiency"] for e in profile.events]
    assert effs == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_rudder_total_loss():
    profile = rudder_total_loss(t_inject=10.0)
    assert profile.events[0].payload["mode"] == "lost"
    assert profile.events[0].payload["surface"] == "rudder"
