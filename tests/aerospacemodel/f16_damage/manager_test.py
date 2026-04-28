"""DamageManager: orchestrates events → state mutations → param updates."""

from __future__ import annotations

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
    DamageEvent, DamageProfile,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


def test_manager_default_state_is_healthy(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    m = DamageManager(geometry=geo, params=p, profile=DamageProfile(events=[]))
    assert all(v == 0.0 for v in m.state.section_loss.values())


def test_manager_triggers_section_loss(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    base_m = p.m
    profile = DamageProfile(events=[
        DamageEvent(5.0, "section_loss",
                    {"section": "left_tip", "loss_fraction": 1.0}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    triggered = mgr.update(t_current=4.0, t_previous=0.0)
    assert triggered == []
    assert p.m == pytest.approx(base_m, rel=0.001)
    triggered = mgr.update(t_current=6.0, t_previous=4.0)
    assert len(triggered) == 1
    assert mgr.state.section_loss["left_tip"] == 1.0
    # Params updated: mass less than baseline
    assert p.m < base_m


def test_manager_triggers_control_failure(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    profile = DamageProfile(events=[
        DamageEvent(2.0, "control_failure",
                    {"surface": "rudder", "mode": "jam",
                     "jam_position_rad": 0.05}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    mgr.update(t_current=2.5, t_previous=1.5)
    assert "rudder" in mgr.state.control_failures
    cf = mgr.state.control_failures["rudder"]
    assert cf.mode == "jam"
    assert cf.jam_position_rad == 0.05


def test_manager_inject_event_runtime(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    mgr = DamageManager(geometry=geo, params=p, profile=DamageProfile(events=[]))
    mgr.inject_event(DamageEvent(
        trigger_time=3.0, event_type="engine_failure",
        payload={"thrust_factor": 0.0, "hard_failure": True},
    ))
    mgr.update(t_current=3.5, t_previous=2.5)
    assert mgr.state.engine.hard_failure is True


def test_manager_reset_clears_state(geo):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    p = F16AngularParameters()
    profile = DamageProfile(events=[
        DamageEvent(1.0, "section_loss",
                    {"section": "left_tip", "loss_fraction": 1.0}),
    ])
    mgr = DamageManager(geometry=geo, params=p, profile=profile)
    mgr.update(t_current=2.0, t_previous=0.0)
    assert mgr.state.section_loss["left_tip"] == 1.0
    mgr.reset()
    assert mgr.state.section_loss["left_tip"] == 0.0
