"""F-16 base geometry calibration: aggregate properties must match
F16AngularParameters within ~1%.
"""

from __future__ import annotations

import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    F16AngularParameters,
)


@pytest.fixture
def base_geo():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )

    return load_f16_geometry()


def test_total_mass_matches_params(base_geo):
    p = F16AngularParameters()
    assert base_geo.total_mass() == pytest.approx(p.m, rel=0.01)


def test_total_wing_area_matches_params(base_geo):
    p = F16AngularParameters()
    assert base_geo.total_wing_area() == pytest.approx(p.S, rel=0.01)


def test_required_section_names_present(base_geo):
    names = set(base_geo.section_names())
    required = {
        "left_root",
        "left_mid",
        "left_tip",
        "right_root",
        "right_mid",
        "right_tip",
        "stab_left",
        "stab_right",
        "vtail",
        "rudder",
        "aileron_left",
        "aileron_right",
        "fuselage_main",
    }
    missing = required - names
    assert not missing, f"Missing required sections: {missing}"


def test_section_sides_consistent(base_geo):
    for s in base_geo.sections:
        if s.name.startswith("left_") or s.name.endswith("_left"):
            assert s.side == "left", f"{s.name} side={s.side}"
        if s.name.startswith("right_") or s.name.endswith("_right"):
            assert s.side == "right", f"{s.name} side={s.side}"


def test_aircraft_cy_alpha_in_realistic_range(base_geo):
    """The aggregate Cy_α from per-section cl_alpha_contribution × area
    should give the aircraft-level lift-curve slope of ~4.5 /rad."""
    s_base = base_geo.total_wing_area()
    cy_alpha = sum(s.cl_alpha_contribution * s.area for s in base_geo.sections) / s_base
    # F-16 baseline is around 4.0–5.0 /rad depending on Mach
    assert 3.5 <= cy_alpha <= 5.5, f"Cy_α = {cy_alpha:.3f} /rad not in [3.5, 5.5]"
