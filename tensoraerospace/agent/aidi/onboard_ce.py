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

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        F16AngularParameters,
    )


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

    The F-16 angular ODE applies aero moments through the *actuator
    positions* held in the state vector (indices 8 = stab, 10 = ail,
    12 = dir); the control input ``u`` only feeds the second-order
    actuator dynamics. INDI's control-effectiveness is therefore the
    gain from **actuator deflection** to angular acceleration, with
    time-scale separation handing the actuator dynamics over to the
    inner loop's increment law (``Δu ≡ Δ(deflection)`` on the airframe
    time-scale).

    Axis-ordering note: this F-16 codebase stores the body rates in the
    order ``(wx, wy, wz) = (p, r, q)`` — i.e. ``wy`` is **yaw rate** and
    ``wz`` is **pitch rate**. The adapter remaps so the returned matrix
    rows correspond to the conventional ``(p, q, r)`` order expected by
    the AIDI outer loop (CStar/roll/sideslip).

    We compute ``G_ij = ∂ω̇_i/∂(deflection_j)`` by central differencing
    ``f16_ode_6dof`` around the operating point: perturb state[8/10/12]
    in turn, read rows 2/4/3 of the ODE output (= p, q, r). The
    returned matrix is in the basis ``(p, q, r) × (stab, ail, dir)``.

    Args:
        params: F-16 parameter set (defaults to
            :func:`default_parameters`).
        perturb: Half-width of the central-difference perturbation
            (radians).
    """

    n_state = 3
    n_control = 3
    # State indices for (p, q, r) — note this codebase stores wy=r and wz=q.
    _RATE_IDX = (2, 4, 3)
    _DEFLECTION_IDX = (8, 10, 12)  # stab, ail, dir actuator positions.

    def __init__(
        self, params: "F16AngularParameters | None" = None, perturb: float = 1e-3
    ) -> None:
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
            raise ValueError(f"u must have length {self.n_control}; got {u_v.size}")
        del u  # control input is unused here (CE is wrt deflection state).
        rate_idx = list(self._RATE_IDX)
        defl_idx = list(self._DEFLECTION_IDX)
        G = np.zeros((self.n_state, self.n_control), dtype=np.float64)
        for j_local, j_state in enumerate(defl_idx):
            x_plus = x_v.copy()
            x_plus[j_state] += self._eps
            x_minus = x_v.copy()
            x_minus[j_state] -= self._eps
            f_plus = self._ode(x_plus, u_v, 0.0, self._params)[rate_idx]
            f_minus = self._ode(x_minus, u_v, 0.0, self._params)[rate_idx]
            G[:, j_local] = (f_plus - f_minus) / (2.0 * self._eps)
        return G
