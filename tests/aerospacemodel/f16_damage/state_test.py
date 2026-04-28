"""DamageState behaviour: defaults, mutation, snapshot."""

from __future__ import annotations

import pytest


def test_default_state_is_healthy():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    geo = load_f16_geometry()
    s = DamageState.healthy(geo)
    assert all(v == 0.0 for v in s.section_loss.values())
    assert s.engine.thrust_factor == 1.0
    assert not s.engine.hard_failure
    assert s.structural.extra_mass_delta_kg == 0.0
    assert s.control_failures == {}


def test_section_loss_clamped_to_unit_interval():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    s = DamageState(section_loss={"x": 0.0}, control_failures={})
    s.set_section_loss("x", 1.5)
    assert s.section_loss["x"] == 1.0
    s.set_section_loss("x", -0.2)
    assert s.section_loss["x"] == 0.0


def test_control_failure_validates_mode():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        ControlFailure,
    )
    cf = ControlFailure(mode="jam", jam_position_rad=0.1)
    assert cf.mode == "jam"
    with pytest.raises(ValueError):
        ControlFailure(mode="not_a_mode")  # type: ignore[arg-type]


def test_snapshot_is_independent_copy():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    geo = load_f16_geometry()
    s = DamageState.healthy(geo)
    snap = s.snapshot()
    s.set_section_loss(geo.section_names()[0], 0.5)
    first = geo.section_names()[0]
    assert snap["section_loss"][first] == 0.0
