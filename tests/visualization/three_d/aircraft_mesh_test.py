"""Static checks on the procedural F-16 builder (viewer.js).

We can't easily render WebGL inside pytest, so these tests inspect the
generated HTML / JS string for required functions, names, and
type-handling branches.
"""

from __future__ import annotations

import re

import numpy as np
import pytest


def _make_html():
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    from tensoraerospace.visualization.three_d import build_flight_log, build_html
    env = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=20, dt=0.01,
        airspeed=200.0,
    )
    env.reset()
    for _ in range(5):
        env.step(np.zeros(3))
    return build_html(build_flight_log(env))


def test_html_no_longer_has_placeholder_cube():
    html = _make_html()
    # The Phase B placeholder used a BoxGeometry(8, 2, 8); this exact
    # call should be gone.
    assert "BoxGeometry(8, 2, 8)" not in html


def test_html_has_buildaircraft():
    html = _make_html()
    assert "function buildAircraft(" in html
    # And it's actually invoked
    assert "buildAircraft(log.geometry)" in html


def test_html_has_body_to_three_helper():
    html = _make_html()
    assert "function bodyToThree" in html
    # The mapping must be exactly [bx, -bz, by]
    assert re.search(r"bodyToThree\([^)]*\)\s*\{[^}]*\[bx,\s*-bz,\s*by\]",
                     html) is not None


def test_html_uses_lathe_geometry_for_fuselage():
    html = _make_html()
    assert "LatheGeometry" in html


def test_html_has_canopy_intake_pitot_nozzle():
    html = _make_html()
    for piece in ("fuselage_canopy", "fuselage_intake", "nozzle"):
        assert piece in html


def test_html_has_named_wing_sections():
    """Each YAML wing section name must appear as a mesh.name in the
    rendered viewer.js for damage visualization to find them."""
    html = _make_html()
    for s in ("left_root", "left_mid", "left_tip",
              "right_root", "right_mid", "right_tip",
              "stab_left", "stab_right",
              "vtail", "rudder",
              "aileron_left", "aileron_right",
              "fuselage_main"):
        assert s in html, f"section name missing: {s}"


def test_html_has_right_wing_polygons():
    """The hand-tuned right-wing planform must be in the bundled JS."""
    html = _make_html()
    assert "RIGHT_WING_POLYGONS" in html
    assert "right_root" in html
    assert "right_mid" in html
    assert "right_tip" in html


def test_camera_distance_increased_for_aircraft_scale():
    """3D camera preset uses an offset around aircraft.position. The
    scene-init position is closer than the original placeholder so the
    larger / more detailed F-16 fills the viewport."""
    html = _make_html()
    assert "camera.position.set(20, 14, 20)" in html


def test_section_meshes_have_names():
    """Mesh names are set so damage viz can find sections by name."""
    html = _make_html()
    # The new builder uses mesh.name = name for flat panels
    assert 'mesh.name = name' in html
