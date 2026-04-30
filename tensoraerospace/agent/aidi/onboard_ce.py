"""Onboard control-effectiveness models used by the AIDI agent.

The agent expects an object that, given the current state ``x`` and the
last applied control ``u``, returns the matrix ``G = ∂ω̇/∂u`` of shape
``(n_state, n_control)`` — where ``n_state`` here means the number of
controlled rate axes (3 for a typical fixed-wing). Two concrete
implementations are provided:

* ``LinearOnboardCE`` — wraps a constant matrix B, useful for linearised
  plants and tests.
* ``F16NonlinearOnboardCE`` — central finite differences on the F-16
  6-DoF angular ODE around the current operating point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OnboardCEModel(Protocol):
    """Duck-typed onboard CE provider."""

    n_state: int
    n_control: int

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...


class LinearOnboardCE:
    """Constant-matrix onboard CE model.

    Args:
        B: Pre-computed control-effectiveness matrix of shape
            ``(n_state, n_control)``.
    """

    def __init__(self, B: np.ndarray) -> None:
        B_arr = np.asarray(B, dtype=np.float64)
        if B_arr.ndim != 2:
            raise ValueError("B must be 2-D")
        self._B = B_arr.copy()
        self.n_state = int(B_arr.shape[0])
        self.n_control = int(B_arr.shape[1])

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del x, u  # constant matrix.
        return self._B.copy()


class F16NonlinearOnboardCE:
    """Finite-difference adapter over the F-16 6-DoF angular ODE.

    Computes ``G_ij = ∂ω̇_i/∂u_j`` by central differencing
    ``f16_ode_6dof`` around the supplied operating point. The 14-element
    state vector is expected (``[α, β, p, q, r, γ, ψ, θ, ...]``), and
    ``G`` is returned in the angular-rate basis ``(p, q, r)`` (rows 2,3,4
    of the ODE).

    Args:
        params: F-16 parameter set (defaults to
            :func:`default_parameters`).
        perturb: Half-width of the central-difference perturbation,
            in the same units the ODE expects on its control vector
            (radians).
    """

    n_state = 3
    n_control = 3
    _RATE_IDX = (2, 3, 4)

    def __init__(self, params=None, perturb: float = 1e-3) -> None:
        from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
            f16_ode_6dof,
        )
        from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
            default_parameters,
        )
        if perturb <= 0.0:
            raise ValueError("perturb must be positive")
        self._ode = f16_ode_6dof
        self._params = params if params is not None else default_parameters()
        self._eps = float(perturb)

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_v = np.asarray(x, dtype=np.float64).reshape(-1)
        u_v = np.asarray(u, dtype=np.float64).reshape(-1)
        if x_v.size != 14:
            raise ValueError(f"x must be 14-element; got {x_v.size}")
        if u_v.size != self.n_control:
            raise ValueError(
                f"u must have length {self.n_control}; got {u_v.size}"
            )
        rate_idx = list(self._RATE_IDX)
        G = np.zeros((self.n_state, self.n_control), dtype=np.float64)
        for j in range(self.n_control):
            u_plus = u_v.copy(); u_plus[j] += self._eps
            u_minus = u_v.copy(); u_minus[j] -= self._eps
            f_plus = self._ode(x_v, u_plus, 0.0, self._params)[rate_idx]
            f_minus = self._ode(x_v, u_minus, 0.0, self._params)[rate_idx]
            G[:, j] = (f_plus - f_minus) / (2.0 * self._eps)
        return G
