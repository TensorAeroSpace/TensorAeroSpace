"""Supersonic directional linear model package.

`initial.py` depends on an optional `matlab` package. To keep the core model
importable in lightweight environments (CI, users without matlab), we guard
that import.
"""

from .model import DirectionalSuperSonic as DirectionalSuperSonic

try:  # pragma: no cover
    from .initial import initial_state as initial_state
except ModuleNotFoundError:  # pragma: no cover
    # Keep package importable without optional dependency.
    initial_state = None  # type: ignore[assignment]
