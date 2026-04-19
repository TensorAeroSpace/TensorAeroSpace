"""Per-step live updater for 3D flight visualization in Jupyter.

Wraps `build_flight_3d_figure` output as a `plotly.graph_objects.FigureWidget`
and provides incremental update via `extend()`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import plotly.graph_objects as go

from .flight_3d import build_flight_3d_figure


class LivePlotlyRenderer:
    """Per-step live updater for 3D flight visualization in a Jupyter notebook.

    Usage::

        r = LivePlotlyRenderer()
        widget = r.init_from(positions, attitudes, time, chart_data)
        display(widget)
        for step in rollout:
            r.extend(
                position_row=...,
                attitude_row=...,
                t=...,
                chart_row=...,
            )
    """

    def __init__(self, *, height: int = 800, trail_length: int | None = None):
        self._height = height
        self._trail_length = trail_length
        self._fig: go.FigureWidget | None = None
        self._chart_names: list[str] = []

    def init_from(
        self,
        positions: np.ndarray,
        attitudes: np.ndarray,
        time: np.ndarray,
        chart_data: Mapping[str, np.ndarray],
    ) -> go.FigureWidget:
        """Build the initial figure and return it as a FigureWidget for display."""
        base_fig = build_flight_3d_figure(
            positions, attitudes, time, dict(chart_data),
            trail_length=self._trail_length,
            height=self._height,
        )
        self._fig = go.FigureWidget(base_fig)
        self._chart_names = list(chart_data.keys())
        return self._fig

    def extend(
        self,
        *,
        position_row: np.ndarray,
        attitude_row: np.ndarray,
        t: float,
        chart_row: Mapping[str, float],
    ) -> None:
        """Append a single timestep to all traces and update the aircraft glyph."""
        if self._fig is None:
            raise RuntimeError("Call init_from(...) before extend(...).")

        from .flight_3d import _make_glyph_mesh

        with self._fig.batch_update():
            # Trail = first scatter3d trace
            trail = self._fig.data[0]
            trail.x = list(trail.x) + [float(position_row[0])]
            trail.y = list(trail.y) + [float(position_row[1])]
            trail.z = list(trail.z) + [float(position_row[2])]
            if self._trail_length is not None and len(trail.x) > self._trail_length:
                trail.x = list(trail.x)[-self._trail_length:]
                trail.y = list(trail.y)[-self._trail_length:]
                trail.z = list(trail.z)[-self._trail_length:]

            # Aircraft glyph = first mesh3d trace; rebuild it in place
            new_glyph = _make_glyph_mesh(position_row, attitude_row)
            for mesh in self._fig.data:
                if mesh.type == "mesh3d":
                    mesh.x = new_glyph.x
                    mesh.y = new_glyph.y
                    mesh.z = new_glyph.z
                    break

            # Chart traces = the scatter (2D) ones in order
            scatter_traces = [tr for tr in self._fig.data if tr.type == "scatter"]
            for trace, name in zip(scatter_traces, self._chart_names):
                trace.x = list(trace.x) + [float(t)]
                trace.y = list(trace.y) + [float(chart_row[name])]
