"""ValueBank lookup logic: nominal/abrupt-with-prob/min-fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.bank import ValueBank, ValueBankConfig


@dataclass
class _Const:
    """Tiny stub HJValueFunction for tests."""

    val: float
    L: float = 1.0

    def value(self, x: np.ndarray) -> float:  # noqa: D401, ARG002
        return float(self.val)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def lipschitz_const(self) -> float:
        return self.L


def _zero_fdd(kind: str = "none") -> FDDOutput:
    return FDDOutput(
        fault_present=(kind != "none"),
        severity=0.0,
        confidence=0.0,
        innovation_norm=0.0,
        time_since_event=0.0,
        fault_kind=kind,
        severity_abrupt=0.0,
        severity_gradual=0.0,
    )


def test_nominal_picks_nominal() -> None:
    bank = ValueBank(
        {"nominal": _Const(0.7), "elev_jam": _Const(-0.3)},
        ValueBankConfig(fallback="min"),
    )
    fdd = _zero_fdd("none")
    assert bank.value(np.zeros(2), fdd) == 0.7


def test_open_world_fallback_min() -> None:
    bank = ValueBank(
        {"nominal": _Const(0.7), "elev_jam": _Const(-0.3)},
        ValueBankConfig(fallback="min"),
    )
    fdd = _zero_fdd("abrupt")  # no MMAE probs available
    assert bank.value(np.zeros(2), fdd) == -0.3


def test_lipschitz_max_over_bank() -> None:
    bank = ValueBank(
        {"nominal": _Const(0.7, L=2.0), "elev_jam": _Const(-0.3, L=5.0)},
        ValueBankConfig(fallback="min"),
    )
    assert bank.lipschitz_const() == 5.0
