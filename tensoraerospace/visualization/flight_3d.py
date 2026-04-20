"""Plotly figure builder for 3D flight visualization with chart strip."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_GLYPH_BODY = np.array(
    [
        [3.0, 0.0, 0.0],  # nose
        [-1.5, 0.5, 0.0],  # tail-right
        [-1.5, -0.5, 0.0],  # tail-left
    ]
)

_GLYPH_WING = np.array(
    [
        [0.0, 0.0, 0.0],  # root
        [0.0, 3.0, 0.0],  # right tip
        [0.0, -3.0, 0.0],  # left tip
    ]
)

_GLYPH_TAIL = np.array(
    [
        [-1.5, 0.0, 0.0],  # base
        [-1.5, 0.0, 1.0],  # top
        [-2.0, 0.0, 0.0],  # back
    ]
)


def _make_glyph_mesh(
    position: np.ndarray,
    attitude: np.ndarray,
    *,
    scale: float = 1.0,
) -> go.Mesh3d:
    """Build a small triangular glyph oriented by attitude and placed at position.

    attitude: (3,) (roll, pitch, yaw) radians.
    position: (3,)
    scale: multiplier for the glyph size; canonical body length is 3m, so
        ``scale=10`` produces a 30m body, ``scale=100`` a 300m body. Use
        a scale proportional to the trail extent so the glyph stays
        visible regardless of how big the flight envelope is.
    """
    from .kinematics import _body_to_inertial_matrix

    R = _body_to_inertial_matrix(attitude[0], attitude[1], attitude[2])
    vertices = np.vstack([_GLYPH_BODY, _GLYPH_WING, _GLYPH_TAIL]) * float(scale)
    transformed = vertices @ R.T + position
    # Indices into the 9-vertex array: body=0..2, wing=3..5, tail=6..8
    i = [0, 3, 6]
    j = [1, 4, 7]
    k = [2, 5, 8]
    return go.Mesh3d(
        x=transformed[:, 0],
        y=transformed[:, 1],
        z=transformed[:, 2],
        i=i,
        j=j,
        k=k,
        color="crimson",
        opacity=0.95,
        name="aircraft",
        hoverinfo="skip",
        showlegend=False,
    )


def _autoscale_glyph(positions: np.ndarray) -> float:
    """Pick a glyph scale proportional to the trail's largest dimension.

    Targets ~5% of the bounding-box diagonal so the aircraft is clearly
    visible without dominating the scene. Falls back to scale=1.0 for
    near-stationary trails.
    """
    if len(positions) < 2:
        return 1.0
    extent = positions.max(axis=0) - positions.min(axis=0)
    diag = float(np.linalg.norm(extent))
    if diag < 1e-6:
        return 1.0
    # Canonical body length is 3 m; scale so that body length == 5% of diag.
    target_body = 0.05 * diag
    return target_body / 3.0


def build_flight_3d_figure(
    positions: np.ndarray,
    attitudes: np.ndarray,
    time: np.ndarray,
    chart_data: dict[str, np.ndarray],
    *,
    trail_length: int | None = None,
    height: int = 800,
) -> go.Figure:
    """Build a Plotly figure: 3D trail + glyph on top, time-series strip below.

    Args:
        positions: (T, 3) inertial position in metres.
        attitudes: (T, 3) Euler angles (roll, pitch, yaw) in radians.
        time: (T,) seconds.
        chart_data: mapping of state name → (T,) values; one chart per entry.
        trail_length: keep only the last N points of the trail. None = all.
        height: total figure height in pixels.

    Returns:
        plotly.graph_objects.Figure ready to .show() or .to_image().
    """
    n_charts = len(chart_data)
    chart_row_height = 0.12
    scene_row_height = max(0.4, 1.0 - chart_row_height * n_charts - 0.05)
    row_heights = [scene_row_height] + [chart_row_height] * n_charts

    specs = [[{"type": "scene"}]] + [[{"type": "xy"}]] * n_charts
    fig = make_subplots(
        rows=1 + n_charts,
        cols=1,
        row_heights=row_heights,
        specs=specs,
        vertical_spacing=0.02,
        subplot_titles=[None] + list(chart_data.keys()),
    )

    # Trail
    if trail_length is not None and trail_length < len(positions):
        trail_positions = positions[-trail_length:]
        trail_time = time[-trail_length:]
    else:
        trail_positions = positions
        trail_time = time

    fig.add_trace(
        go.Scatter3d(
            x=trail_positions[:, 0],
            y=trail_positions[:, 1],
            z=trail_positions[:, 2],
            mode="lines+markers",
            marker=dict(
                size=2, color=trail_time, colorscale="Viridis", showscale=False
            ),
            line=dict(width=4, color="rgba(70, 100, 200, 0.7)"),
            name="trail",
        ),
        row=1,
        col=1,
    )

    # Aircraft glyph at final pose, sized proportionally to the trail extent
    glyph_scale = _autoscale_glyph(trail_positions)
    fig.add_trace(
        _make_glyph_mesh(positions[-1], attitudes[-1], scale=glyph_scale),
        row=1,
        col=1,
    )

    # Chart strip
    for i, (name, values) in enumerate(chart_data.items(), start=2):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=values,
                mode="lines",
                name=name,
                line=dict(width=2),
                hovertemplate=f"{name}: %{{y:.4f}}<br>t=%{{x:.2f}}s<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=i, col=1)

    fig.update_layout(
        height=height,
        scene=dict(
            xaxis_title="x (m, north)",
            yaxis_title="y (m, east)",
            zaxis_title="z (m, up)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    if n_charts > 0:
        fig.update_xaxes(title_text="time (s)", row=1 + n_charts, col=1)

    return fig
