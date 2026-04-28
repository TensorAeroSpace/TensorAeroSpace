"""AeroSection and BaseGeometry primitives."""

from __future__ import annotations

import pytest


def test_aero_section_is_frozen():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection,
    )
    s = AeroSection(
        name="left_tip", side="left", type="wing",
        area=2.0, span_position=-3.5, chord=1.5, sweep=0.0,
        mass=80.0, cg_local=(0.0, -3.5, 0.0),
        inertia_local=(50.0, 100.0, 60.0, 0.0),
        cl_alpha_contribution=0.4, cd0_contribution=0.005,
    )
    with pytest.raises((AttributeError, Exception)):
        s.area = 99.0


def test_base_geometry_aggregates_mass_and_area():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    # Sections are arranged in symmetric left/right pairs:
    # (sec_0 at y=-1, sec_1 at y=+1) and (sec_2 at y=-3, sec_3 at y=+3)
    span_ys = [-1.0, 1.0, -3.0, 3.0]
    sections = [
        AeroSection(
            name=f"sec_{i}", side="left" if span_ys[i] < 0 else "right",
            type="wing",
            area=1.0, span_position=span_ys[i],
            chord=1.0, sweep=0.0,
            mass=10.0,
            cg_local=(0.0, span_ys[i], 0.0),
            inertia_local=(1.0, 1.0, 1.0, 0.0),
            cl_alpha_contribution=0.1,
            cd0_contribution=0.001,
        )
        for i in range(4)
    ]
    g = BaseGeometry(sections=sections)
    assert g.total_wing_area() == pytest.approx(4.0)
    assert g.total_mass() == pytest.approx(40.0)
    # CG should be at y=0 for symmetric layout
    assert g.center_of_mass()[1] == pytest.approx(0.0, abs=1e-9)


def test_base_geometry_lookup_by_name():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.geometry import (
        AeroSection, BaseGeometry,
    )
    s = AeroSection(
        name="left_tip", side="left", type="wing",
        area=1.0, span_position=-3.0, chord=1.0, sweep=0.0,
        mass=10.0, cg_local=(0, -3, 0), inertia_local=(1, 1, 1, 0),
        cl_alpha_contribution=0.1, cd0_contribution=0.001,
    )
    g = BaseGeometry(sections=[s])
    assert g.section("left_tip") is s
    with pytest.raises(KeyError):
        g.section("nonexistent")
