"""DamageEvent / DamageProfile: scheduling and triggering."""

from __future__ import annotations

import pytest


def test_event_is_frozen():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
    )
    e = DamageEvent(
        trigger_time=5.0, event_type="section_loss",
        payload={"section": "left_tip", "loss_fraction": 1.0},
    )
    with pytest.raises((AttributeError, Exception)):
        e.trigger_time = 99.0


def test_profile_returns_pending_in_window():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    e1 = DamageEvent(1.0, "section_loss", {"section": "x", "loss_fraction": 1.0})
    e2 = DamageEvent(5.5, "engine_failure", {"thrust_factor": 0.0})
    e3 = DamageEvent(10.0, "section_loss", {"section": "y", "loss_fraction": 0.5})
    p = DamageProfile(events=[e1, e2, e3])
    pending = p.get_pending_events(t_current=6.0, t_previous=1.0)
    assert e2 in pending
    # e1 is at t=1.0, window is (1.0, 6.0], so e1 is NOT in (exclusive at t_prev)
    assert e1 not in pending
    assert e3 not in pending


def test_profile_inclusive_at_current_exclusive_at_previous():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    e_at_5 = DamageEvent(5.0, "engine_failure", {"thrust_factor": 0.5})
    p = DamageProfile(events=[e_at_5])
    # (4.99, 5.0] should contain it
    assert e_at_5 in p.get_pending_events(5.0, 4.99)
    # (5.0, 5.5] should NOT
    assert e_at_5 not in p.get_pending_events(5.5, 5.0)


def test_invalid_event_type_raises():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
    )
    with pytest.raises(ValueError):
        DamageEvent(5.0, "not_a_type", {})  # type: ignore[arg-type]


def test_negative_trigger_time_raises():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent,
    )
    with pytest.raises(ValueError):
        DamageEvent(-1.0, "section_loss", {"section": "x", "loss_fraction": 0.5})
