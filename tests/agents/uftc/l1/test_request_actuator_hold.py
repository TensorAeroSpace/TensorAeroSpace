"""request_actuator_hold freezes u_safe for exactly one filter() call."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)
from tensoraerospace.agent.uftc.l1.shield import (
    HJReachabilityShield,
    HJShieldConfig,
)


@dataclass
class _Const:
    v: float = 100.0  # always deep inside safe set
    L: float = 1.0

    def value(self, x): return self.v
    def gradient(self, x): return np.zeros_like(x)
    def lipschitz_const(self): return self.L


def _clean_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=False, severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind="none", severity_abrupt=0.0, severity_gradual=0.0,
    )


def _build_shield():
    cm = ConformalMargin(ConformalMarginConfig(), lipschitz_const=1.0)
    return HJReachabilityShield(
        n_state=2, n_control=2, value_fn=_Const(),
        dynamics_fn=lambda x, u: u,
        cfg=HJShieldConfig(h_clear=0.0,
                           u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0])),
        conformal_margin=cm,
    )


def test_hold_repeats_last_u_safe_once() -> None:
    sh = _build_shield()
    out1 = sh.filter(np.zeros(2), np.array([0.5, -0.2]), _clean_fdd())
    sh.request_actuator_hold()
    out2 = sh.filter(np.zeros(2), np.array([0.9, 0.9]), _clean_fdd())
    assert np.allclose(out2.u_safe, out1.u_safe)
    # next tick returns to nominal
    out3 = sh.filter(np.zeros(2), np.array([0.1, 0.1]), _clean_fdd())
    assert np.allclose(out3.u_safe, [0.1, 0.1])


def test_hold_without_prior_filter_is_noop() -> None:
    sh = _build_shield()
    sh.request_actuator_hold()
    out = sh.filter(np.zeros(2), np.array([0.3, -0.1]), _clean_fdd())
    # No prior u_safe, so the hold is silently dropped.
    assert np.allclose(out.u_safe, [0.3, -0.1])
