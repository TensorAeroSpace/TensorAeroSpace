"""3D WebGL visualization for F-16 envs.

Public API:
    build_flight_log(env)              # → JSON-serializable dict
    build_html(flight_log)             # → self-contained HTML string
    save_html(flight_log, path)        # → write HTML to file
    render(env, *, open_in_browser=True, save_to=None, inline=None)
        # high-level: build log + html and either open in browser
        # (script) or return an IPython.display.HTML (Jupyter)

Used by gym envs through ``render_mode="3d_web"``.
"""

from .builder import build_html, save_html
from .exporter import build_flight_log
from .render import render

__all__ = ["build_flight_log", "build_html", "save_html", "render"]
