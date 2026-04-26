"""Plotly figure builder for 3D flight visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest


def _toy_data(T: int = 50):
    positions = np.column_stack(
        [
            np.linspace(0, 100, T),
            np.zeros(T),
            np.linspace(0, 10, T),
        ]
    )
    attitudes = np.zeros((T, 3))
    attitudes[:, 1] = np.linspace(0, 0.1, T)  # pitch
    time = np.linspace(0, 1.0, T)
    chart_data = {
        "alpha": np.linspace(0, 0.05, T),
        "wz": np.zeros(T),
    }
    return positions, attitudes, time, chart_data


def test_build_returns_figure():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data()
    fig = build_flight_3d_figure(positions, attitudes, time, chart_data)
    assert isinstance(fig, go.Figure)


def test_figure_has_3d_scene():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data()
    fig = build_flight_3d_figure(positions, attitudes, time, chart_data)
    # The scene-typed subplot is registered under fig.layout.scene
    assert fig.layout.scene is not None


def test_trail_trace_present():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data()
    fig = build_flight_3d_figure(positions, attitudes, time, chart_data)
    scatter3d_traces = [t for t in fig.data if t.type == "scatter3d"]
    assert len(scatter3d_traces) >= 1
    trail = scatter3d_traces[0]
    assert len(trail.x) == 50


def test_chart_traces_present_per_chart_state():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data()
    fig = build_flight_3d_figure(positions, attitudes, time, chart_data)
    scatter2d_traces = [t for t in fig.data if t.type == "scatter"]
    # one trace per chart_data entry (alpha + wz)
    assert len(scatter2d_traces) == 2


def test_trail_length_clipping():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data(T=100)
    fig = build_flight_3d_figure(
        positions,
        attitudes,
        time,
        chart_data,
        trail_length=30,
    )
    trail = next(t for t in fig.data if t.type == "scatter3d")
    assert len(trail.x) == 30


def test_aircraft_glyph_at_final_pose():
    from tensoraerospace.visualization.flight_3d import build_flight_3d_figure

    positions, attitudes, time, chart_data = _toy_data()
    fig = build_flight_3d_figure(positions, attitudes, time, chart_data)
    mesh3d_traces = [t for t in fig.data if t.type == "mesh3d"]
    # one glyph for the aircraft
    assert len(mesh3d_traces) >= 1
