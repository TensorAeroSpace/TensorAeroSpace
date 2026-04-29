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
    assert "function presetFree" in html
    assert "function presetChase" in html
    assert "function presetTopDown" in html
    for btn in ("btn-cam-free", "btn-cam-chase", "btn-cam-top"):
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
