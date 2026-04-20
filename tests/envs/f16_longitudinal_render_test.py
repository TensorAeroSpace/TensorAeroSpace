"""NonlinearLongitudinalF16 render + position tracking."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def env_factory():
    def _make(**overrides):
        from tensoraerospace.envs.f16.nonlinear_longitudinal import (
            NonlinearLongitudinalF16,
        )

        defaults = dict(
            initial_state=np.zeros(4),
            reference_signal=np.zeros((1, 100)),
            number_time_steps=10,
            dt=0.01,
            airspeed=200.0,
        )
        defaults.update(overrides)
        return NonlinearLongitudinalF16(**defaults)

    return _make


def test_position_history_grows(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(3):
        env.step(np.array([0.0]))
    assert env.unwrapped.position_history.shape == (4, 3)


def test_attitude_history_pitch_only(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(3):
        env.step(np.array([0.0]))
    att = env.unwrapped.attitude_history
    np.testing.assert_allclose(att[:, 0], 0.0)  # roll = 0
    np.testing.assert_allclose(att[:, 2], 0.0)  # yaw = 0


def test_chart_history_default_states(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(3):
        env.step(np.array([0.0]))
    ch = env.unwrapped.chart_history
    assert "alpha" in ch and "wz" in ch and "stab" in ch
    assert len(ch["alpha"]) == 4


def test_render_human_returns_figure(env_factory):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    env = env_factory(render_mode="human")
    env.reset()
    for _ in range(5):
        env.step(np.array([0.0]))
    fig = env.render()
    assert isinstance(fig, go.Figure)


def test_render_none_returns_none(env_factory):
    env = env_factory(render_mode=None)
    env.reset()
    env.step(np.array([0.0]))
    assert env.render() is None
