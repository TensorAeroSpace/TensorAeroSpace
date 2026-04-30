"""Built-in damage scenarios."""

from __future__ import annotations


def test_wing_strike_left_tip_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets

    p = presets.WING_STRIKE_LEFT_TIP
    assert len(p.events) == 1
    e = p.events[0]
    assert e.event_type == "section_loss"
    assert e.payload["section"] == "left_tip"
    assert e.payload["loss_fraction"] == 1.0


def test_engine_flameout_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets

    p = presets.ENGINE_FLAMEOUT
    e = p.events[0]
    assert e.event_type == "engine_failure"


def test_birdstrike_compound_preset():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets

    p = presets.BIRDSTRIKE_COMPOUND
    assert len(p.events) >= 2  # multiple events bundled
