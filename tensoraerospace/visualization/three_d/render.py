"""High-level renderer entry point: build flight log → display.

Auto-detects Jupyter vs. terminal and picks an appropriate display:
  - Jupyter / IPython kernel: returns IPython.display.HTML so the
    WebGL viewer renders inline in the notebook cell.
  - Regular Python: writes ``flight_3d_viewer.html`` to the current
    working directory and opens it in the user's default browser via
    ``webbrowser.open()``. Always prints the absolute path so the user
    can copy-paste it manually if the auto-open fails (a common case
    on Linux distros where Firefox runs from a Snap/Flatpak sandbox
    and cannot read ``/tmp``).

Forced overrides via keyword args:
  - inline=True / inline=False
  - open_in_browser=False (suppress browser open in script mode)
  - save_to=path (override the output path; useful for tests / CI)
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Any, Optional

from .builder import build_html, save_html
from .exporter import build_flight_log

DEFAULT_FILENAME = "flight_3d_viewer.html"


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        # ZMQInteractiveShell → Jupyter; TerminalInteractiveShell → ipython REPL
        return "Kernel" in type(ip).__name__ or "ZMQ" in type(ip).__name__
    except Exception:
        return False


def _default_output_path() -> Path:
    """Pick a default location for the generated HTML.

    Defaults to the current working directory. Can be overridden via
    the ``TENSORAEROSPACE_3D_OUT`` environment variable.

    Avoids ``/tmp`` because Firefox-as-snap (the default on recent
    Ubuntu) and Chrome-as-flatpak run inside sandboxes that cannot
    read ``/tmp``. Writing to CWD is universally readable and easy to
    locate ("look in the directory you ran the script from").
    """
    override = os.environ.get("TENSORAEROSPACE_3D_OUT")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / DEFAULT_FILENAME


def render(
    env,
    *,
    open_in_browser: bool = True,
    save_to: Optional[str | Path] = None,
    inline: Optional[bool] = None,
    title: Optional[str] = None,
) -> Any:
    """Render the env's completed episode to the 3D WebGL viewer.

    Parameters
    ----------
    env : NonlinearAngularF16 (or compatible)
    open_in_browser : bool, default True
        In script mode, open the generated HTML in the user's default
        browser. Ignored in Jupyter mode.
    save_to : str | Path, optional
        Override the output path. When None (default), writes
        ``flight_3d_viewer.html`` to the current working directory
        (override globally with ``TENSORAEROSPACE_3D_OUT`` env var).
    inline : bool, optional
        Force Jupyter-inline (True) or script-popup (False). When None,
        auto-detect via IPython.
    title : str, optional
        Browser tab title.

    Returns
    -------
    object
        - Jupyter mode: an ``IPython.display.HTML`` instance
        - Script mode: the ``Path`` to the generated HTML
    """
    log = build_flight_log(env)
    html = build_html(log, title=title)

    use_inline = inline if inline is not None else _is_notebook()

    if use_inline:
        from IPython.display import HTML

        # Wrap in a fixed-height iframe so the page layout doesn't get
        # taken over by the absolute-positioned UI inside.
        # Use srcdoc so the entire viewer is self-contained inside the iframe.
        iframe = (
            '<iframe srcdoc="' + html.replace('"', "&quot;") + '" '
            'width="100%" height="600" style="border:0; border-radius:6px;">'
            "</iframe>"
        )
        if save_to is not None:
            save_html(log, save_to, title=title)
        return HTML(iframe)

    # Script mode
    if save_to is not None:
        out_path = save_html(log, save_to, title=title)
    else:
        out_path = _default_output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

    # Always announce where the file is. The auto-open often fails
    # silently on Linux (no $DISPLAY, $BROWSER misconfigured, snap
    # sandbox), and the user has no way to recover without the path.
    abs_path = out_path.absolute()
    print(f"[viz-3d] Wrote {abs_path}", file=sys.stderr)
    print(f"[viz-3d] Open: file://{abs_path}", file=sys.stderr)

    if open_in_browser:
        try:
            opened = webbrowser.open(out_path.as_uri())
            if not opened:
                print(
                    "[viz-3d] webbrowser.open() returned False — "
                    "open the path above manually.",
                    file=sys.stderr,
                )
        except Exception as e:  # pragma: no cover - best-effort
            print(
                f"[viz-3d] webbrowser.open() failed: {e}\n"
                "[viz-3d] Open the path above manually.",
                file=sys.stderr,
            )

    return out_path
