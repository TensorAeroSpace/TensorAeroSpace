"""Render a flight log into a self-contained HTML viewer.

The HTML inlines:
  - three.js (UMD bundle, vendored at static/vendor/three.min.js)
  - OrbitControls (UMD wrapper, vendored at static/vendor/OrbitControls.js)
  - viewer.js, viewer.css (our scene + UI code)
  - the flight log dict as a JSON literal in window.FLIGHT_LOG

So the resulting .html file is fully portable: open it in any modern
browser, no network or local server required.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_NAME = "viewer.html.j2"


def _read_static(filename: str) -> str:
    static = resources.files("tensoraerospace.visualization.three_d.static")
    return static.joinpath(filename).read_text(encoding="utf-8")


def _read_vendor(filename: str) -> str:
    static = resources.files("tensoraerospace.visualization.three_d.static")
    return static.joinpath("vendor").joinpath(filename).read_text(encoding="utf-8")


def _template_dir() -> Path:
    pkg = resources.files("tensoraerospace.visualization.three_d")
    return Path(str(pkg.joinpath("templates")))


def build_html(flight_log: dict[str, Any], *, title: str | None = None) -> str:
    """Return a self-contained HTML string rendering the flight log."""
    env = Environment(
        loader=FileSystemLoader(str(_template_dir())),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
    )
    tmpl = env.get_template(_TEMPLATE_NAME)
    return tmpl.render(
        title=title or "F-16 flight viewer",
        flight_log_json=json.dumps(flight_log),
        three_js=_read_vendor("three.min.js"),
        orbit_controls_js=_read_vendor("OrbitControls.js"),
        viewer_js=_read_static("viewer.js"),
        css=_read_static("viewer.css"),
    )


def save_html(flight_log: dict[str, Any], path: str | Path,
              *, title: str | None = None) -> Path:
    """Render the flight log to ``path`` and return the absolute path."""
    html = build_html(flight_log, title=title)
    out = Path(path).expanduser().resolve()
    out.write_text(html, encoding="utf-8")
    return out
