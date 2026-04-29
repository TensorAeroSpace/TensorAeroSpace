"""Tests for Phase D-E damage visualization and camera preset additions."""

from __future__ import annotations

import re

import numpy as np


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


def test_html_has_damage_state_machinery():
    html = _make_html()
    assert "function damageStateAt(" in html
    assert "function applyDamageState(" in html
    assert "log.damage_state_history" in html


def test_html_has_engine_exhaust():
    html = _make_html()
    assert "ConeGeometry" in html
    assert 'name = "exhaust"' in html


def test_html_has_camera_presets():
    html = _make_html()
    assert "function preset3DCamera" in html
    assert "function presetTopDown" in html
    assert "function presetLeftSide" in html
    assert "function presetRightSide" in html
    for btn in ("btn-cam-3d", "btn-cam-top", "btn-cam-left", "btn-cam-right"):
        assert btn in html


def test_html_has_events_hud():
    html = _make_html()
    assert 'id="hud-events"' in html


def test_damage_state_applied_each_frame():
    """applyDamageState must be invoked from inside setFrame."""
    html = _make_html()
    # Find the setFrame function block
    m = re.search(r"function setFrame\([^)]*\)\s*\{(.*?)\n    \}", html, re.DOTALL)
    assert m is not None
    body = m.group(1)
    assert "applyDamageState(" in body


def test_camera_follows_aircraft_each_frame():
    """updateCamera() must be invoked from setFrame() so all preset
    modes (3D / Top / Left / Right) track aircraft.position as the
    plane moves through the scene."""
    html = _make_html()
    assert "function updateCamera" in html
    m = re.search(r"function setFrame\([^)]*\)\s*\{(.*?)\n    \}", html, re.DOTALL)
    assert m is not None
    assert "updateCamera()" in m.group(1)


def test_camera_follow_uses_aircraft_rotation():
    """updateCamera() must rotate the offset by aircraft.rotation so the
    camera follows in body frame, not world frame."""
    html = _make_html()
    assert "applyEuler(aircraft.rotation)" in html
    # camera.up is also rotated for proper roll-tracking
    assert "camera.up" in html


def test_html_uses_tube_geometry_for_trail():
    html = _make_html()
    assert "TubeGeometry" in html
    assert "CatmullRomCurve3" in html
    # Make sure the old per-frame Float32Array hairline is gone
    assert "trailGeom.attributes.position.needsUpdate" not in html


def test_damage_breakaway_animation_present():
    """Sections that get damaged should animate (translate + rotate
    over BREAKAWAY_DURATION) before fading out."""
    html = _make_html()
    assert "function advanceDamageAnimations" in html
    assert "BREAKAWAY_DURATION" in html
    assert "damageAnim" in html


def test_camera_3d_mode_moves_with_aircraft():
    """In 3D mode, both controls.target AND camera.position move by
    the same delta so the orbit ring travels with the aircraft."""
    html = _make_html()
    # Look for the delta-shift pattern
    assert "p.clone().sub(controls.target)" in html
    assert "camera.position.add(delta)" in html


def test_control_surfaces_use_hinge_groups():
    """Stabilators / rudder / ailerons must be hinge-pivoted Groups so
    their deflection can be animated about the correct axis."""
    html = _make_html()
    assert "_stabHingeGroup" in html
    assert "_rudderHingeGroup" in html
    assert "_aileronHingeGroup" in html
    # Hinge body coordinates declared
    assert "STAB_RIGHT_HINGE_BODY" in html
    assert "AILERON_RIGHT_HINGE_BODY" in html
    assert "RUDDER_HINGE_BODY" in html


def test_control_surface_deflections_applied_each_frame():
    """setFrame() must rotate stab / aileron / rudder groups by the
    trajectory deflection values."""
    import re
    html = _make_html()
    m = re.search(r"function setFrame\([^)]*\)\s*\{(.*?)\n    \}", html, re.DOTALL)
    assert m is not None
    body = m.group(1)
    assert "traj.stab[idx]" in body
    assert "traj.ail[idx]" in body
    assert "traj.dir[idx]" in body
    assert 'getObjectByName("stab_left")' in body
    assert 'getObjectByName("rudder")' in body
