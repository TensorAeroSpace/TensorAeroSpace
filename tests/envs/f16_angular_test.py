"""NonlinearAngularF16 env basic behaviour."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def env_factory():
    def _make(**overrides):
        from tensoraerospace.envs.f16.nonlinear_angular import (
            NonlinearAngularF16,
        )

        defaults = dict(
            initial_state=np.zeros(14),
            number_time_steps=10,
            dt=0.01,
            airspeed=200.0,
        )
        defaults.update(overrides)
        return NonlinearAngularF16(**defaults)

    return _make


def test_env_constructs(env_factory):
    env = env_factory()
    assert env is not None


def test_observation_space_shape(env_factory):
    env = env_factory()
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_action_space_shape(env_factory):
    env = env_factory()
    assert env.action_space.shape == (3,)  # stab, ail, dir


def test_step_advances_position_history(env_factory):
    env = env_factory()
    env.reset()
    initial_len = len(env.unwrapped.position_history)
    env.step(np.zeros(3))
    assert len(env.unwrapped.position_history) == initial_len + 1


def test_position_history_is_3d(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(5):
        env.step(np.zeros(3))
    pos = env.unwrapped.position_history
    assert pos.ndim == 2 and pos.shape[1] == 3


def test_attitude_history_is_3d(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(5):
        env.step(np.zeros(3))
    att = env.unwrapped.attitude_history
    assert att.ndim == 2 and att.shape[1] == 3


def test_chart_history_contains_default_states(env_factory):
    env = env_factory()
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    ch = env.unwrapped.chart_history
    assert "alpha" in ch and "beta" in ch
    assert len(ch["alpha"]) == 4  # initial + 3 steps


def test_time_history_grows(env_factory):
    env = env_factory()
    env.reset()
    env.step(np.zeros(3))
    env.step(np.zeros(3))
    t = env.unwrapped.time_history
    assert len(t) == 3
    np.testing.assert_allclose(t, [0.0, 0.01, 0.02])


def test_render_none_returns_none(env_factory):
    env = env_factory(render_mode=None)
    env.reset()
    env.step(np.zeros(3))
    assert env.render() is None


def test_render_human_returns_figure(env_factory):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    env = env_factory(render_mode="human")
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    fig = env.render()
    assert isinstance(fig, go.Figure)


def test_render_rgb_array_returns_ndarray(env_factory):
    pytest.importorskip("kaleido")
    pytest.importorskip("PIL")
    env = env_factory(render_mode="rgb_array")
    env.reset()
    for _ in range(3):
        env.step(np.zeros(3))
    arr = env.render()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3 and arr.shape[-1] in (3, 4)


def test_render_live_returns_figurewidget(env_factory):
    pytest.importorskip("anywidget")
    import plotly.graph_objects as go

    env = env_factory(render_mode="live")
    env.reset()
    fig = env.render()
    assert isinstance(fig, go.FigureWidget)
    # Subsequent step + render should not raise
    env.step(np.zeros(3))
    env.render()
