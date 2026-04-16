"""Online incremental system identification via Recursive Least Squares.

Identifies a local linear model in *increment* form::

    Δy_{t+1} ≈ A_t · Δy_t + B_t · Δu_t

where Δy_t = y_t − y_{t−1}, Δu_t = u_t − u_{t−1}. The key advantage over a
full global NN plant model is that only first-order partial derivatives
(F = ∂f/∂x, G = ∂f/∂u) are needed by the critic/actor updates in Dual
Heuristic Programming, and those are exactly the matrices A, B that this
class tracks.

The implementation follows the standard exponentially-weighted RLS with
forgetting factor α ∈ (0, 1] used in the incremental ADP literature
(Sun, van Kampen; Zhou, Chu, van Kampen). A single covariance matrix P is
shared across output dimensions (block RLS).
"""

from __future__ import annotations

import numpy as np


class IncrementalModelRLS:
    """Recursive least squares identifier for the incremental model.

    Maintains a parameter matrix ``theta`` of shape ``(n_y + n_u, n_y)``
    such that the first ``n_y`` rows correspond to ``Aᵀ`` and the last
    ``n_u`` rows correspond to ``Bᵀ``. The identification equation is::

        Δy_{t+1}ᵀ ≈ φ_tᵀ · theta

    with regressor ``φ_t = [Δy_t; Δu_t]`` of length ``n_y + n_u``.

    Args:
        n_y: Dimension of the observed/tracked output vector y.
        n_u: Dimension of the control input vector u.
        forgetting: RLS forgetting factor α ∈ (0, 1]. Values <1 give more
            weight to recent samples; ``1.0`` recovers ordinary least
            squares. Typical flight-control values: 0.995–0.9999.
        cov_init: Initial scale of the covariance matrix P₀ = cov_init · I.
            Larger values encourage faster initial learning at the cost of
            robustness to noisy early samples.
        theta_init_scale: Standard deviation of the zero-mean Gaussian
            used to randomly initialise ``theta``. A small non-zero value
            helps break the symmetry when later matrix operations are
            used (e.g. ``Bᵀ B``).
        seed: Optional random seed for ``theta`` initialisation.
    """

    def __init__(
        self,
        n_y: int,
        n_u: int,
        forgetting: float = 0.999,
        cov_init: float = 1e2,
        theta_init_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must be in (0, 1]")
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.alpha = float(forgetting)
        self.cov_init = float(cov_init)
        self._rng = np.random.default_rng(seed)

        n_theta = self.n_y + self.n_u
        self.theta = self._rng.normal(
            0.0, theta_init_scale, size=(n_theta, self.n_y)
        )
        self.P = np.eye(n_theta, dtype=np.float64) * self.cov_init

        # Buffers used by the agent wrapper to form Δy, Δu.
        self.y_prev: np.ndarray | None = None
        self.y_prev2: np.ndarray | None = None
        self.u_prev: np.ndarray | None = None
        self.u_prev2: np.ndarray | None = None
        self.last_prediction_error: np.ndarray | None = None
        self.num_updates: int = 0

    @property
    def A(self) -> np.ndarray:
        """Return the identified ``A`` matrix (shape ``(n_y, n_y)``)."""
        return self.theta[: self.n_y, :].T

    @property
    def B(self) -> np.ndarray:
        """Return the identified ``B`` matrix (shape ``(n_y, n_u)``)."""
        return self.theta[self.n_y :, :].T

    def reset(self) -> None:
        """Reset the parameter buffers (keeps ``theta`` and ``P``)."""
        self.y_prev = None
        self.y_prev2 = None
        self.u_prev = None
        self.u_prev2 = None
        self.last_prediction_error = None

    def reset_covariance(self) -> None:
        """Reset ``P`` to its initial large-variance state (full re-learn)."""
        n_theta = self.n_y + self.n_u
        self.P = np.eye(n_theta, dtype=np.float64) * self.cov_init

    def update(
        self,
        y_prev: np.ndarray,
        y_curr: np.ndarray,
        y_next: np.ndarray,
        u_prev: np.ndarray,
        u_curr: np.ndarray,
    ) -> np.ndarray:
        """Perform a single RLS step from a sliding window of length 3.

        Uses the tuple ``(y_{t-1}, y_t, y_{t+1}, u_{t-1}, u_t)`` to form the
        regressor ``φ = [y_t − y_{t-1}; u_t − u_{t-1}]`` and the target
        ``δ = y_{t+1} − y_t``.

        Args:
            y_prev: ``y_{t-1}`` — observation two steps back.
            y_curr: ``y_t`` — observation one step back.
            y_next: ``y_{t+1}`` — most recent observation.
            u_prev: ``u_{t-1}`` — control applied at step ``t-1``.
            u_curr: ``u_t`` — control applied at step ``t``.

        Returns:
            The prediction error ``ε = δ − θᵀ φ`` before the update.
        """
        y_prev_v = np.asarray(y_prev, dtype=np.float64).reshape(-1)
        y_curr_v = np.asarray(y_curr, dtype=np.float64).reshape(-1)
        y_next_v = np.asarray(y_next, dtype=np.float64).reshape(-1)
        u_prev_v = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        u_curr_v = np.asarray(u_curr, dtype=np.float64).reshape(-1)

        if y_prev_v.size != self.n_y or y_curr_v.size != self.n_y:
            raise ValueError(f"expected observations of length {self.n_y}")
        if u_prev_v.size != self.n_u or u_curr_v.size != self.n_u:
            raise ValueError(f"expected controls of length {self.n_u}")

        dy = y_curr_v - y_prev_v
        du = u_curr_v - u_prev_v
        target = y_next_v - y_curr_v

        phi = np.concatenate([dy, du])
        phi_col = phi.reshape(-1, 1)

        # Prediction error before the update.
        pred = self.theta.T @ phi
        eps = target - pred

        # Gain vector and covariance recursion.
        Pphi = self.P @ phi_col
        denom = self.alpha + float((phi_col.T @ Pphi).item())
        K = Pphi / denom

        self.theta = self.theta + K @ eps.reshape(1, -1)
        self.P = (self.P - K @ (Pphi.T)) / self.alpha

        # Numerical symmetrisation — cheap insurance against drift.
        self.P = 0.5 * (self.P + self.P.T)

        self.last_prediction_error = eps
        self.num_updates += 1
        return eps

    def predict_next(
        self,
        y_curr: np.ndarray,
        y_prev: np.ndarray,
        u_curr: np.ndarray,
        u_prev: np.ndarray,
    ) -> np.ndarray:
        """Predict ``y_{t+1}`` from the current model estimates.

        Uses the incremental form ``y_{t+1} ≈ y_t + A · (y_t − y_{t-1}) +
        B · (u_t − u_{t-1})``.
        """
        y_c = np.asarray(y_curr, dtype=np.float64).reshape(-1)
        y_p = np.asarray(y_prev, dtype=np.float64).reshape(-1)
        u_c = np.asarray(u_curr, dtype=np.float64).reshape(-1)
        u_p = np.asarray(u_prev, dtype=np.float64).reshape(-1)

        return y_c + self.A @ (y_c - y_p) + self.B @ (u_c - u_p)
