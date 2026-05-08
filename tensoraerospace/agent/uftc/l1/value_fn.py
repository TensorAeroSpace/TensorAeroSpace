"""HJ value-function protocol used by the L1 shield.

A value function ``V(x)`` is *non-positive inside* the safe set, *zero on the
boundary*, and *positive outside*. The shield uses ``V`` and ``∇V`` to enforce
forward-invariance of the safe set under a CBF-style QP.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HJValueFunction(Protocol):
    """Minimal contract any L1 value function must satisfy."""

    def value(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def lipschitz_const(self) -> float: ...
