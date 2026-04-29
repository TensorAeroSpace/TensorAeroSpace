"""Builder tests: HTML structure, vendor bundles inlined, flight log embedded."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest


def _make_flight_log():
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    from tensoraerospace.visualization.three_d import build_flight_log
    env = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=20, dt=0.01,
        airspeed=200.0,
    )
    env.reset()
    for _ in range(5):
        env.step(np.zeros(3))
    return build_flight_log(env)


def test_build_html_returns_nonempty_string():
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log())
    assert isinstance(html, str)
    assert len(html) > 100_000  # three.js alone is ~600 KB


def test_html_has_required_dom_anchors():
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log())
    for sel in ('id="scene"', 'id="hud"', 'id="controls"',
                'id="btn-play"', 'id="timeline"', 'id="speed"',
                'id="hud-time"', 'id="hud-alpha"', 'id="hud-beta"',
                'id="hud-wx"', 'id="hud-wz"'):
        assert sel in html, f"missing dom anchor: {sel}"


def test_html_inlines_three_js_and_orbit_controls():
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log())
    # three.js global
    assert "THREE.WebGLRenderer" in html or "WebGLRenderer" in html
    # OrbitControls UMD wrapper
    assert "OrbitControls" in html


def test_html_embeds_flight_log_json():
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log())
    assert "window.FLIGHT_LOG" in html
    # Extract the JSON literal
    m = re.search(r"window\.FLIGHT_LOG\s*=\s*(\{.*?\});", html, re.DOTALL)
    assert m is not None
    parsed = json.loads(m.group(1))
    assert parsed["version"] == 1
    assert "trajectory" in parsed
    assert "geometry" in parsed


def test_save_html_writes_file(tmp_path):
    from tensoraerospace.visualization.three_d import save_html
    out = save_html(_make_flight_log(), tmp_path / "flight.html")
    assert out.exists()
    contents = out.read_text(encoding="utf-8")
    assert contents.startswith("<!DOCTYPE html>")
    assert "window.FLIGHT_LOG" in contents


def test_html_with_custom_title():
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log(), title="Custom Test Title")
    assert "<title>Custom Test Title</title>" in html


def test_html_self_contained_no_external_urls():
    """No CDN refs, no localhost — fully offline-capable."""
    from tensoraerospace.visualization.three_d import build_html
    html = build_html(_make_flight_log())
    forbidden_patterns = (
        "https://cdn", "https://unpkg", "http://localhost",
        "<script src=", "<link rel=\"stylesheet\" href=",
    )
    for p in forbidden_patterns:
        assert p not in html, f"Found forbidden external reference: {p!r}"
