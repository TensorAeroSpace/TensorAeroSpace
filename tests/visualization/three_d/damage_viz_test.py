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


def test_html_has_keyboard_shortcuts():
    """Keyboard handler must be installed and cover the documented keys."""
    html = _make_html()
    assert "addEventListener(\"keydown\"" in html or \
        "addEventListener('keydown'" in html
    for case in ("Space", "ArrowLeft", "ArrowRight", "Home", "End",
                 "Digit1", "Digit2", "Digit3", "Digit4"):
        assert case in html, f"missing keybinding for {case}"


def test_html_has_keyboard_help_overlay():
    html = _make_html()
    assert 'id="keyboard-help"' in html
    assert "Keyboard shortcuts" in html


def test_html_has_lerx_and_realistic_features():
    """The upgraded F-16 mesh ships LERX, ventral fins, wingtip launchers."""
    html = _make_html()
    for feature in ("lerx_right", "lerx_left",
                    "ventral_fin_right", "ventral_fin_left",
                    "launcher_right", "launcher_left"):
        assert feature in html, f"missing F-16 feature: {feature}"


def test_deflection_visual_gain_present():
    """The visual amplification factor that makes stab/ail/dir
    deflections readable on the small surfaces."""
    html = _make_html()
    assert "DEFLECTION_VISUAL_GAIN" in html


def test_hud_has_deflection_readouts():
    html = _make_html()
    for hud_id in ("hud-stab", "hud-ail", "hud-dir"):
        assert hud_id in html


def test_html_has_f16_hud_overlay():
    """The fighter-style SVG HUD overlay must be present with all
    documented anchors (heading tape, KIAS, altitude, pitch ladder,
    bank scale, FPM, AOA/G, caution)."""
    html = _make_html()
    assert 'id="hud-svg"' in html
    for anchor in (
        "hud-hdg-value", "hud-kias-value", "hud-mach-value",
        "hud-alt-value", "hud-vsi-value", "hud-aoa-value",
        "hud-g-value", "hud-beta-value",
        "hud-bank-pointer", "hud-bank-rot", "hud-pitch-trans",
        "hud-pitch-ladder", "hud-reticle", "hud-fpm",
        "hud-caution", "hud-time-value", "hud-speed-value",
    ):
        assert anchor in html, f"missing HUD anchor: {anchor}"


def test_hud_pitch_ladder_built_at_init():
    """The pitch ladder rungs are appended to #hud-pitch-ladder by an
    IIFE during scene init (rather than hardcoded in the HTML)."""
    html = _make_html()
    assert "hud-pitch-ladder" in html
    assert "PITCH_DEG_PER_PX" in html


def test_hud_reads_params_from_flight_log():
    """The HUD must source airspeed and altitude from log.metadata.params,
    not from hardcoded constants."""
    html = _make_html()
    assert "log.metadata.params" in html
    # The old hardcoded constants must be gone
    assert "TRIM_ALT_M = 3000.0" not in html
    # params.V is read for airspeed, params.Oy for trim altitude
    assert "params.V" in html
    assert "params.Oy" in html


def test_hud_caution_responds_to_damage_state():
    """The CAUTION badge visibility must be set from damageStateAt()
    each frame."""
    import re
    html = _make_html()
    m = re.search(r"function setFrame\([^)]*\)\s*\{(.*?)\n    \}", html, re.DOTALL)
    assert m is not None
    body = m.group(1)
    assert 'id="hud-caution"' in html or '"hud-caution"' in html
    assert "anyDamage" in body
