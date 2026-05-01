"""Small render helpers for environments without graphical viewers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def validate_render_mode(render_mode: str | None, render_modes: Sequence[str]) -> None:
    """Validate Gymnasium-style render mode values."""
    if render_mode is not None and render_mode not in render_modes:
        raise ValueError(
            f"render_mode must be one of {list(render_modes)} or None, "
            f"got {render_mode!r}"
        )


def _format_value(value: Any, max_items: int = 6) -> str:
    if value is None:
        return "None"
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return repr(value)
    shown = ", ".join(f"{item:.4g}" for item in arr[:max_items])
    if arr.size > max_items:
        shown += ", ..."
    return f"[{shown}]"


def telemetry_render(
    env_name: str,
    mode: str | None,
    *,
    step: int,
    total_steps: int,
    state: Any,
    action: Any = None,
    reward: Any = None,
) -> str | None:
    """Render a lightweight text snapshot for envs without a GUI backend."""
    if mode is None:
        return None
    if mode not in ("human", "ansi"):
        raise ValueError("render mode must be 'human', 'ansi', or None")

    snapshot = (
        f"{env_name}(step={step}/{max(total_steps - 1, 0)}, "
        f"state={_format_value(state)}, action={_format_value(action)}, "
        f"reward={_format_value(reward)})"
    )
    if mode == "ansi":
        return snapshot
    print(snapshot)
    return None
