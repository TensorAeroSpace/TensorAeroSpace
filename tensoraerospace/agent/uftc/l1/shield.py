"""HJ-Reachability safety shield: QP post-filter on u_indi.

For an affine-in-control surrogate ``f̂(x, u) = F̃ x + G̃ u``, the shield
enforces a CBF-style condition::

    ⟨∇V, F̃ x + G̃ u⟩ + λ V(x) ≥ ε_t

while minimising ``‖u − u_nominal‖²`` subject to ``u_min ≤ u ≤ u_max``.
The solver is OSQP via cvxpy; failures degrade to the nominal control.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .conformal import ConformalMargin, ConformalMarginConfig
from .value_fn import HJValueFunction

LOG = logging.getLogger(__name__)


class _Identity:
    """Always-safe constant value function placeholder.

    Used by :class:`UFTCController` when ``enable_l1_shield=True`` but no
    ``l1_value_fn_path`` is supplied — reports ``V(x) ≡ 1`` (deep inside the
    safe set), zero gradient and unit Lipschitz constant. The shield then
    short-circuits to the nominal control on every call: a no-op shield
    until a real :class:`DeepReachValueFn` is wired in.
    """

    def value(self, x: np.ndarray) -> float:                  # noqa: D401
        return 1.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(x, dtype=np.float64).reshape(-1))

    def lipschitz_const(self) -> float:
        return 1.0


@dataclass
class HJShieldConfig:
    h_clear: float = 0.20
    qp_solver: Literal["OSQP", "ECOS", "MOSEK"] = "OSQP"
    cbf_lambda: float = 1.0
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None
    conformal: ConformalMarginConfig = field(default_factory=ConformalMarginConfig)


@dataclass
class ShieldOutput:
    u_safe: np.ndarray
    intervention_norm: float
    hjb_value: float
    active: bool


class HJReachabilityShield:
    """QP post-filter enforcing forward invariance of the HJ safe set."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        *,
        value_fn: HJValueFunction,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        cfg: HJShieldConfig,
        conformal_margin: ConformalMargin | None = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.value_fn = value_fn
        self.dynamics_fn = dynamics_fn
        self.cfg = cfg
        if conformal_margin is None:
            conformal_margin = ConformalMargin(
                cfg.conformal, lipschitz_const=value_fn.lipschitz_const(),
            )
        self.conformal = conformal_margin
        self._hold_one_tick = False
        self._last_u_safe: np.ndarray | None = None
        self._cached_FG: tuple[np.ndarray, np.ndarray] | None = None

    # ----- macro-action sink -----
    def request_actuator_hold(self) -> None:
        self._hold_one_tick = True

    def set_dynamics_jacobian(self, F: np.ndarray, G: np.ndarray) -> None:
        """UFTCController calls this once per tick with current F̃, G̃ from RLS."""
        self._cached_FG = (np.asarray(F, dtype=np.float64),
                           np.asarray(G, dtype=np.float64))

    def filter(
        self,
        x: np.ndarray,
        u_nominal: np.ndarray,
        fdd: FDDOutput,
        monitor_alarm: str = "OK",
    ) -> ShieldOutput:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        u_nominal = np.asarray(u_nominal, dtype=np.float64).reshape(-1)

        if self._hold_one_tick and self._last_u_safe is not None:
            self._hold_one_tick = False
            return ShieldOutput(
                u_safe=self._last_u_safe.copy(),
                intervention_norm=float(np.linalg.norm(self._last_u_safe - u_nominal)),
                hjb_value=float(self.value_fn.value(x)),
                active=True,
            )

        v_x = float(self.value_fn.value(x))
        eps_t = float(self.conformal.compute(fdd, monitor_alarm))
        h_safe = v_x - eps_t
        if h_safe > self.cfg.h_clear:
            self._last_u_safe = u_nominal.copy()
            return ShieldOutput(u_nominal.copy(), 0.0, v_x, active=False)

        grad_v = self.value_fn.gradient(x)
        try:
            u_safe = self._solve_qp(x, u_nominal, grad_v, v_x, eps_t)
        except Exception as e:                      # pragma: no cover - logged
            LOG.warning("HJ-shield QP failed (%s); falling back to nominal", e)
            self._last_u_safe = u_nominal.copy()
            return ShieldOutput(u_nominal.copy(), 0.0, v_x, active=False)

        self._last_u_safe = u_safe.copy()
        return ShieldOutput(
            u_safe=u_safe,
            intervention_norm=float(np.linalg.norm(u_safe - u_nominal)),
            hjb_value=v_x,
            active=True,
        )

    def reset(self) -> None:
        self._hold_one_tick = False
        self._last_u_safe = None
        self._cached_FG = None

    # ----- internal -----
    def _affine_FG(self, x: np.ndarray, u_nominal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_FG is not None:
            return self._cached_FG
        if self.dynamics_fn is None:
            raise RuntimeError("dynamics_fn is None and no F̃,G̃ cached")
        # Numerical Jacobian (F = ∂f/∂x at u, G = ∂f/∂u at x).
        h = 1e-6
        f0 = self.dynamics_fn(x, u_nominal)
        F = np.zeros((self.n_state, self.n_state))
        for i in range(self.n_state):
            x_p = x.copy(); x_p[i] += h
            F[:, i] = (self.dynamics_fn(x_p, u_nominal) - f0) / h
        G = np.zeros((self.n_state, self.n_control))
        for i in range(self.n_control):
            u_p = u_nominal.copy(); u_p[i] += h
            G[:, i] = (self.dynamics_fn(x, u_p) - f0) / h
        return F, G

    def _solve_qp(
        self,
        x: np.ndarray,
        u_nominal: np.ndarray,
        grad_v: np.ndarray,
        v_x: float,
        eps_t: float,
    ) -> np.ndarray:
        import cvxpy as cp

        F, G = self._affine_FG(x, u_nominal)
        u = cp.Variable(self.n_control)
        objective = cp.Minimize(cp.sum_squares(u - u_nominal))
        constraints = [grad_v @ (F @ x + G @ u) + self.cfg.cbf_lambda * v_x >= eps_t]
        if self.cfg.u_min is not None:
            constraints.append(u >= np.asarray(self.cfg.u_min, dtype=np.float64))
        if self.cfg.u_max is not None:
            constraints.append(u <= np.asarray(self.cfg.u_max, dtype=np.float64))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.cfg.qp_solver, verbose=False)
        if u.value is None:
            raise RuntimeError(f"QP returned no solution; status={prob.status}")
        return np.asarray(u.value, dtype=np.float64).reshape(-1)
