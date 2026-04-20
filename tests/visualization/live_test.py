"""LivePlotlyRenderer behaviour."""

from __future__ import annotations

import numpy as np
import pytest


def _seed_data(T: int = 5):
    positions = np.zeros((T, 3))
    positions[:, 0] = np.arange(T)
    attitudes = np.zeros((T, 3))
    time = np.arange(T, dtype=float) * 0.1
    chart_data = {"alpha": np.zeros(T), "wz": np.zeros(T)}
    return positions, attitudes, time, chart_data


def test_init_from_returns_figurewidget():
    pytest.importorskip("ipywidgets")
    from tensoraerospace.visualization.live import LivePlotlyRenderer

    r = LivePlotlyRenderer()
    fig = r.init_from(*_seed_data())
    import plotly.graph_objects as go

    assert isinstance(fig, go.FigureWidget)


def test_extend_grows_trail():
    pytest.importorskip("ipywidgets")
    from tensoraerospace.visualization.live import LivePlotlyRenderer

    r = LivePlotlyRenderer()
    r.init_from(*_seed_data())
    initial_trail_len = len(r._fig.data[0].x)
    r.extend(
        position_row=np.array([10.0, 0.0, 0.0]),
        attitude_row=np.array([0.0, 0.05, 0.0]),
        t=0.6,
        chart_row={"alpha": 0.05, "wz": 0.0},
    )
    assert len(r._fig.data[0].x) == initial_trail_len + 1


def test_extend_appends_to_chart_traces():
    pytest.importorskip("ipywidgets")
    from tensoraerospace.visualization.live import LivePlotlyRenderer

    r = LivePlotlyRenderer()
    r.init_from(*_seed_data())
    chart_traces_before = [len(t.y) for t in r._fig.data if t.type == "scatter"]
    r.extend(
        position_row=np.array([10.0, 0.0, 0.0]),
        attitude_row=np.array([0.0, 0.05, 0.0]),
        t=0.6,
        chart_row={"alpha": 0.05, "wz": 0.0},
    )
    chart_traces_after = [len(t.y) for t in r._fig.data if t.type == "scatter"]
    assert chart_traces_after == [n + 1 for n in chart_traces_before]
