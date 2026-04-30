"""Tests that the 3D viewer correctly handles altitude-tracking envs."""

from __future__ import annotations

import numpy as np


def _make_altitude_env(steps=10):
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=20,
        dt=0.01,
        airspeed=200.0,
        track_altitude=True,
    )
    env.reset()
    for _ in range(steps):
        env.step(np.zeros(3))
    return env


def test_exporter_handles_16_state():
    from tensoraerospace.visualization.three_d import build_flight_log

    env = _make_altitude_env()
    log = build_flight_log(env)
    traj = log["trajectory"]
    assert "altitude_m" in traj
    assert "airspeed_mps" in traj
    assert len(traj["altitude_m"]) == len(traj["time"])


def test_viewer_uses_real_altitude_when_available():
    from tensoraerospace.visualization.three_d import build_flight_log, build_html

    env = _make_altitude_env()
    log = build_flight_log(env)
    html = build_html(log)
    # Viewer JS must reference the altitude_m channel
    assert "traj.altitude_m" in html
    assert "traj.airspeed_mps" in html
