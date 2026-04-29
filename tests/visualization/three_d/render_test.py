"""Tests for the high-level render() entrypoint and env integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_env_after_run(render_mode=None, damage_profile=None, n_steps=10):
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    env = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=20, dt=0.01,
        airspeed=200.0,
        split_stab=damage_profile is not None,
        damage_profile=damage_profile,
        render_mode=render_mode,
    )
    env.reset()
    n_inputs = 4 if env.split_stab else 3
    for _ in range(n_steps):
        env.step(np.zeros(n_inputs))
    return env


def test_render_3d_web_mode_registered():
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    assert "3d_web" in NonlinearAngularF16.metadata["render_modes"]


def test_render_save_to_writes_html(tmp_path):
    from tensoraerospace.visualization.three_d import render
    env = _make_env_after_run()
    out = render(env, open_in_browser=False, inline=False,
                 save_to=tmp_path / "flight.html")
    assert isinstance(out, Path)
    assert out.exists()
    contents = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in contents
    assert "window.FLIGHT_LOG" in contents


def test_env_render_dispatches_to_3d_web(tmp_path, monkeypatch):
    """env.render() with render_mode='3d_web' must produce an HTML, not
    open a browser, when redirected via patched webbrowser.open."""
    import webbrowser
    env = _make_env_after_run(render_mode="3d_web")

    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)

    out = env.render()
    assert isinstance(out, Path)
    assert out.exists()
    assert len(opened) == 1
    assert opened[0].startswith("file://")


def test_render_inline_returns_iframe_html():
    """inline=True returns an IPython.display.HTML wrapping an iframe."""
    pytest.importorskip("IPython")
    from IPython.display import HTML
    from tensoraerospace.visualization.three_d import render

    env = _make_env_after_run()
    out = render(env, inline=True, open_in_browser=False)
    assert isinstance(out, HTML)
    # The rendered IPython HTML wraps an iframe with srcdoc
    s = out.data
    assert "<iframe" in s
    assert "srcdoc=" in s


def test_render_no_browser_when_disabled(tmp_path, monkeypatch):
    """open_in_browser=False must not call webbrowser.open."""
    import webbrowser
    from tensoraerospace.visualization.three_d import render

    env = _make_env_after_run()
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)

    out = render(env, inline=False, open_in_browser=False,
                 save_to=tmp_path / "flight.html")
    assert opened == []
    assert out.exists()
