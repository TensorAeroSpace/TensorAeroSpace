"""HJReachabilityShield QP behaviour and bounds enforcement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)
from tensoraerospace.agent.uftc.l1.shield import (
    HJReachabilityShield,
    HJShieldConfig,
    ShieldOutput,
)


@dataclass
class _Linear:
    """V(x) = a^T x + b — linear value function for analytical tests."""

    a: np.ndarray
    b: float

    def value(self, x: np.ndarray) -> float:
        return float(self.a @ x + self.b)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.a, dtype=np.float64, copy=True)

    def lipschitz_const(self) -> float:
        return float(np.linalg.norm(self.a))


def _clean_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=False, severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind="none", severity_abrupt=0.0, severity_gradual=0.0,
    )


def _affine_dynamics(F: np.ndarray, G: np.ndarray):
    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return F @ x + G @ u
    return f


def test_passthrough_when_deep_inside_safe_set() -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=10.0)  # V(x) very positive
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=vfn.lipschitz_const())
    shield = HJReachabilityShield(
        n_state=2, n_control=2,
        value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=0.5, u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0]), conformal=cm.cfg),
        conformal_margin=cm,
    )
    x = np.array([0.0, 0.0])
    u_nom = np.array([0.7, -0.3])
    out = shield.filter(x, u_nom, _clean_fdd())
    assert isinstance(out, ShieldOutput)
    assert out.active is False
    assert np.allclose(out.u_safe, u_nom)


def test_qp_enforces_u_bounds() -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=0.0)  # V at boundary
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=1.0)
    shield = HJReachabilityShield(
        n_state=2, n_control=2,
        value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=1.0,    # force shield active
                           u_min=np.array([-0.5, -0.5]),
                           u_max=np.array([0.5, 0.5]),
                           conformal=cm.cfg),
        conformal_margin=cm,
    )
    out = shield.filter(np.array([0.0, 0.0]),
                        np.array([2.0, -2.0]),  # outside bounds
                        _clean_fdd())
    assert (out.u_safe >= -0.5 - 1e-6).all()
    assert (out.u_safe <= 0.5 + 1e-6).all()


def test_solver_failure_falls_back_to_nominal(monkeypatch) -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=0.0)
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=1.0)
    shield = HJReachabilityShield(
        n_state=2, n_control=2, value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=1.0,
                           u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0]),
                           conformal=cm.cfg),
        conformal_margin=cm,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(shield, "_solve_qp", boom)
    out = shield.filter(np.array([0.0, 0.0]),
                        np.array([0.4, -0.2]),
                        _clean_fdd())
    assert out.active is False
    assert np.allclose(out.u_safe, [0.4, -0.2])
