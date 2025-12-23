"""Utilities for aerospace models.

This package historically relied on importing LaTeX label dictionaries directly
from `tensoraerospace.aerospacemodel.utils`. Some modules still do:

    from tensoraerospace.aerospacemodel.utils import state_to_latex_eng

In the current layout those dictionaries live in `utils/constant.py`. This
module re-exports them for backwards compatibility.
"""

from .constant import (  # noqa: F401
    control_to_latex_eng,
    control_to_latex_rus,
    ref_state_to_latex_eng,
    ref_state_to_latex_rus,
    state_to_latex_eng,
    state_to_latex_rus,
)

__all__ = [
    "state_to_latex_eng",
    "state_to_latex_rus",
    "ref_state_to_latex_eng",
    "ref_state_to_latex_rus",
    "control_to_latex_eng",
    "control_to_latex_rus",
]
