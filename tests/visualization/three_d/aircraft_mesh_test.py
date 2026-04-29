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


def test_html_handles_all_section_types():
    html = _make_html()
    for t in ("wing", "stab", "vtail", "control", "fuselage"):
        assert f'"{t}"' in html or f"'{t}'" in html, f"missing branch for {t}"


def test_aircraft_uses_log_geometry_sections():
    """The mesh builder must iterate over log.geometry.sections so the
    visual stays in sync with the YAML / DamageState."""
    html = _make_html()
    assert "geometry.sections" in html


def test_camera_distance_increased_for_aircraft_scale():
    html = _make_html()
    # Phase C bumps initial camera position so the ~10 m fuselage fits
    assert "camera.position.set(45, 30, 45)" in html


def test_section_meshes_named_after_section_name():
    """mesh.name is set so Phase E can find sections by name."""
    html = _make_html()
    assert "mesh.name = s.name" in html
