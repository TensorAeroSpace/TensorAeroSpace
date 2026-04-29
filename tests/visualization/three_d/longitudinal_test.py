"""Tests for the 3D viewer integration on the longitudinal env."""

from __future__ import annotations

import numpy as np
import pytest


def _make_longitudinal_env(damage_profile=None, render_mode=None):
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )
    env = NonlinearLongitudinalF16(
        initial_state=np.array([0.0, 0.0]),  # alpha, wz in state_space
        reference_signal=np.zeros((1, 50)),
        number_time_steps=20,
        dt=0.01,
        damage_profile=damage_profile,
        render_mode=render_mode,
    )
    env.reset()
    for _ in range(5):
        env.step(np.zeros(1))
    return env


def test_longitudinal_env_has_damage_log_attributes():
    env = _make_longitudinal_env()
    assert hasattr(env, "damage_events_log")
    assert hasattr(env, "damage_state_log")
    assert env.damage_events_log == []


def test_longitudinal_env_records_damage_events():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.events import (
        DamageEvent, DamageProfile,
    )
    profile = DamageProfile(events=[
        DamageEvent(0.02, "section_loss",
                    {"section": "left_tip", "loss_fraction": 0.5}),
    ])
    env = _make_longitudinal_env(damage_profile=profile)
    assert len(env.damage_events_log) == 1
    e = env.damage_events_log[0]
    assert e["payload"]["section"] == "left_tip"
    assert 0 < e["time"] <= 0.10


def test_longitudinal_env_3d_web_in_metadata():
    from tensoraerospace.envs.f16.nonlinear_longitudinal import (
        NonlinearLongitudinalF16,
    )
    assert "3d_web" in NonlinearLongitudinalF16.metadata["render_modes"]


def test_build_flight_log_for_longitudinal_env():
    from tensoraerospace.visualization.three_d import build_flight_log
    env = _make_longitudinal_env()
    log = build_flight_log(env)
    traj = log["trajectory"]
    n = len(traj["time"])
    # All channels present, channels not in the longitudinal ODE are zero
    assert all(v == 0.0 for v in traj["beta"])
    assert all(v == 0.0 for v in traj["wx"])
    assert all(v == 0.0 for v in traj["wy"])
    assert all(v == 0.0 for v in traj["ail"])
    assert all(v == 0.0 for v in traj["dir"])
    # Channels that ARE modelled have data
    assert len(traj["alpha"]) == n
    assert len(traj["wz"]) == n
    assert len(traj["stab"]) == n


def test_longitudinal_env_render_3d_web(tmp_path, monkeypatch):
    """env.render() with render_mode='3d_web' on the longitudinal env
    must produce an HTML and (in script mode) call webbrowser.open."""
    import webbrowser
    env = _make_longitudinal_env(render_mode="3d_web")
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    out = env.render()
    from pathlib import Path
    assert isinstance(out, Path)
    assert out.exists()
    assert opened and opened[0].startswith("file://")


def test_build_flight_log_unsupported_state_dimension():
    """Sanity check: build_flight_log should reject other state sizes."""
    from tensoraerospace.visualization.three_d.exporter import build_flight_log
    # Construct a minimal mock env with a wrong-shape x_history
    class _MockModel:
        x_history = [np.zeros((7, 1))]  # 7-element state — not 4 or 14
    class _MockEnv:
        model = _MockModel()
        position_history = np.zeros((1, 3))
        attitude_history = np.zeros((1, 3))
        time_history = np.zeros(1)
        damage_events_log = []
        damage_state_log = []
        dt = 0.01
        airspeed = 200.0
        split_stab = False
        _geo_for_obs = None
        _geo_for_damage = None
    with pytest.raises(ValueError, match="state dimension"):
        build_flight_log(_MockEnv())
