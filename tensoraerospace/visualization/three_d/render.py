"""High-level renderer entry point: build flight log → display.

Auto-detects Jupyter vs. terminal and picks an appropriate display:
  - Jupyter / IPython kernel: returns IPython.display.HTML so the
    WebGL viewer renders inline in the notebook cell.
  - Regular Python: writes a temp HTML and opens the user's default
    browser via webbrowser.open().

Forced overrides via keyword args:
  - inline=True / inline=False
  - open_in_browser=False (suppress browser open in script mode)
  - save_to=path (additionally save a copy to a chosen location)
"""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Optional

from .builder import build_html, save_html
from .exporter import build_flight_log


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None:
            return False
        # ZMQInteractiveShell → Jupyter; TerminalInteractiveShell → ipython REPL
        return "Kernel" in type(ip).__name__ or "ZMQ" in type(ip).__name__
    except Exception:
        return False


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
        Additionally write the HTML to this path (returns the same path
        object). Useful for archiving runs.
    inline : bool, optional
        Force Jupyter-inline (True) or script-popup (False). When None,
        auto-detect via IPython.
    title : str, optional
        Browser tab title.

    Returns
    -------
    object
        - Jupyter mode: an ``IPython.display.HTML`` instance
        - Script mode: the ``Path`` to the generated (tempfile or save_to) HTML
    """
    log = build_flight_log(env)
    html = build_html(log, title=title)

    use_inline = inline if inline is not None else _is_notebook()

    if use_inline:
        from IPython.display import HTML  # type: ignore
        # Wrap in a fixed-height iframe so the page layout doesn't get
        # taken over by the absolute-positioned UI inside.
        # Use srcdoc so the entire viewer is self-contained inside the iframe.
        iframe = (
            '<iframe srcdoc="' + html.replace('"', "&quot;") + '" '
            'width="100%" height="600" style="border:0; border-radius:6px;">'
            '</iframe>'
        )
        if save_to is not None:
            save_html(log, save_to, title=title)
        return HTML(iframe)

    # Script mode
    if save_to is not None:
        out_path = save_html(log, save_to, title=title)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="tensoraerospace_3d_"))
        out_path = tmpdir / "flight.html"
        out_path.write_text(html, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(out_path.as_uri())

    return out_path
