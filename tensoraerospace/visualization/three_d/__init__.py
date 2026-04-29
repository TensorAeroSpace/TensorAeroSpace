"""3D WebGL visualization for F-16 envs (in development).

Public API:
    build_flight_log(env) -> dict       # JSON-serializable trajectory + damage log
    build_html(flight_log) -> str       # Self-contained HTML string
    save_html(flight_log, path) -> Path # Render to a file

Subsequent phases will add:
  - Procedural F-16 mesh (Phase C)
  - Damage event visualization (Phase E)
  - render_mode="3d_web" integration (Phase F)
"""

from .builder import build_html, save_html
from .exporter import build_flight_log

__all__ = ["build_flight_log", "build_html", "save_html"]
