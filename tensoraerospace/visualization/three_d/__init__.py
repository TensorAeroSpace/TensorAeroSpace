"""3D WebGL visualization for F-16 envs (in development).

Public API:
    build_flight_log(env) -> dict   # JSON-serializable trajectory + damage log

Subsequent phases will add:
  - HTML/JS template generation
  - Auto-open browser / Jupyter inline display
  - render_mode="3d_web" integration on the gym envs
"""

from .exporter import build_flight_log

__all__ = ["build_flight_log"]
