"""Per-mode value-function bank with worst-case open-world fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .value_fn import HJValueFunction


@dataclass
class ValueBankConfig:
    fallback: Literal["nominal", "min"] = "min"
    abrupt_lookup_threshold: float = 0.7


class ValueBank:
    """Picks a per-mode V_theta^(h) based on FDDOutput."""

    def __init__(self, value_fns: Mapping[str, HJValueFunction],
                 cfg: ValueBankConfig | None = None) -> None:
        if "nominal" not in value_fns:
            raise ValueError("bank must contain a 'nominal' entry")
        self._vs = dict(value_fns)
        self.cfg = cfg or ValueBankConfig()

    def value(self, x: np.ndarray, fdd: FDDOutput) -> float:
        return self._lookup(fdd).value(x)

    def gradient(self, x: np.ndarray, fdd: FDDOutput) -> np.ndarray:
        return self._lookup(fdd).gradient(x)

    def lipschitz_const(self) -> float:
        return float(max(v.lipschitz_const() for v in self._vs.values()))

    # ----- internal -----
    def _lookup(self, fdd: FDDOutput) -> HJValueFunction:
        if fdd.fault_kind == "none":
            return self._vs["nominal"]
        # MMAE-based class lookup not in Phase 2 — fall through to fallback.
        if self.cfg.fallback == "nominal":
            return self._vs["nominal"]
        # "min" — worst-case open-world shielding: pick the entry whose
        # value at this state is smallest (closest to / past boundary).
        return _MinOverBank(self._vs)


class _MinOverBank:
    """Helper exposing HJValueFunction surface backed by min over a bank."""

    def __init__(self, vs: Mapping[str, HJValueFunction]) -> None:
        self._vs = dict(vs)

    def value(self, x: np.ndarray) -> float:
        return float(min(v.value(x) for v in self._vs.values()))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        # gradient of min is the gradient of the argmin (subgradient choice).
        argmin = min(self._vs.values(), key=lambda v: v.value(x))
        return argmin.gradient(x)

    def lipschitz_const(self) -> float:
        return float(max(v.lipschitz_const() for v in self._vs.values()))
