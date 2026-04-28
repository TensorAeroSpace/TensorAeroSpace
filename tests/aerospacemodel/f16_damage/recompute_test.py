"""Recompute aircraft parameters from DamageState + BaseGeometry."""

from __future__ import annotations

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
    load_f16_geometry,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
    DamageState,
)


@pytest.fixture
def geo():
    return load_f16_geometry()


@pytest.fixture
def healthy(geo):
    return DamageState.healthy(geo)


def test_healthy_recompute_matches_baseline(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    out = recompute_mass_geometry(geo, healthy)
    p = F16AngularParameters()
    assert out["m"] == pytest.approx(p.m, rel=0.01)
    assert out["S"] == pytest.approx(p.S, rel=0.01)
    assert out["bA"] == pytest.approx(p.bA, rel=0.05)
    # CG of healthy aircraft is at origin (per fuselage_main cg=0)
    assert abs(out["cg"][1]) < 0.01


def test_left_tip_full_loss_reduces_mass_and_area(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    healthy.set_section_loss("left_tip", 1.0)
    out = recompute_mass_geometry(geo, healthy)
    base = recompute_mass_geometry(geo, DamageState.healthy(geo))
    tip = geo.section("left_tip")
    assert out["m"] == pytest.approx(base["m"] - tip.mass, rel=0.001)
    assert out["S"] == pytest.approx(base["S"] - tip.area, rel=0.001)
    # CG must shift to the right (positive y) when left tip is lost
    assert out["cg"][1] > 0.005


def test_symmetric_loss_keeps_cg_centered(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    healthy.set_section_loss("left_tip", 0.5)
    healthy.set_section_loss("right_tip", 0.5)
    out = recompute_mass_geometry(geo, healthy)
    assert abs(out["cg"][1]) < 1e-6


def test_b_eff_uses_outermost_surviving_section(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_mass_geometry,
    )
    # Lose left tip entirely → effective half-span on left is left_mid (≈ 2.80 m)
    healthy.set_section_loss("left_tip", 1.0)
    out = recompute_mass_geometry(geo, healthy)
    # b_eff = max(left_mid abs y) + max(right_tip abs y) = 2.80 + 4.30 = 7.10
    assert out["b"] == pytest.approx(2.80 + 4.30, abs=0.01)


def test_healthy_inertia_matches_baseline(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia,
    )
    import numpy as np
    out = recompute_inertia(geo, healthy, cg=geo.center_of_mass())
    p = F16AngularParameters()
    # Baseline inertias: tolerance 5% (geometry quantisation)
    assert out["Jx"] == pytest.approx(p.Jx, rel=0.05)
    assert out["Jy"] == pytest.approx(p.Jy, rel=0.05)
    assert out["Jz"] == pytest.approx(p.Jz, rel=0.05)


def test_steiner_known_two_point_masses():
    """Validate Huygens-Steiner: two equal point masses at ±d on x-axis,
    each m, give Iyy = Izz = 2 m d² about the centre."""
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia,
    )
    import numpy as np
    sections = [
        AeroSection(
            name="a", side="center", type="fuselage",
            area=0.0, span_position=0.0, chord=0.0, sweep=0.0,
            mass=2.0, cg_local=(3.0, 0.0, 0.0), inertia_local=(0, 0, 0, 0),
            cl_alpha_contribution=0.0, cd0_contribution=0.0,
        ),
        AeroSection(
            name="b", side="center", type="fuselage",
            area=0.0, span_position=0.0, chord=0.0, sweep=0.0,
            mass=2.0, cg_local=(-3.0, 0.0, 0.0), inertia_local=(0, 0, 0, 0),
            cl_alpha_contribution=0.0, cd0_contribution=0.0,
        ),
    ]
    geo = BaseGeometry(sections=sections)
    state = DamageState.healthy(geo)
    out = recompute_inertia(geo, state, cg=np.zeros(3))
    # Iyy = Izz = 2*2*9 = 36; Ixx = 0
    assert out["Jx"] == pytest.approx(0.0, abs=1e-9)
    assert out["Jy"] == pytest.approx(36.0, rel=1e-9)
    assert out["Jz"] == pytest.approx(36.0, rel=1e-9)


def test_left_tip_loss_reduces_jx(geo, healthy):
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia, recompute_mass_geometry,
    )
    base = recompute_inertia(geo, healthy, cg=geo.center_of_mass())
    healthy.set_section_loss("left_tip", 1.0)
    new_cg = recompute_mass_geometry(geo, healthy)["cg"]
    out = recompute_inertia(geo, healthy, cg=new_cg)
    # Jx primarily comes from y-distance of mass; losing a tip reduces it
    assert out["Jx"] < base["Jx"]
