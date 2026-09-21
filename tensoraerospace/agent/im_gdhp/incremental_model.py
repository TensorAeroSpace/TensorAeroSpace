"""Online incremental system identification via Recursive Least Squares.

Identifies a local linear model in *increment* form::

    Δy_{t+1} ≈ Σ_j A_{t,j} · Δy_{t-j} + Σ_j B_{t,j} · Δu_{t-j}

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

from typing import Sequence

import numpy as np


class IncrementalModelRLS:
    """Recursive least squares identifier for the incremental model.

    Maintains a parameter matrix ``theta`` of shape ``(M * (n_y + n_u), n_y)``
    such that the first ``M * n_y`` rows correspond to ``Aᵀ`` and the last
    ``M * n_u`` rows correspond to ``Bᵀ``. The identification equation is::

        Δy_{t+1}ᵀ ≈ φ_tᵀ · theta

    with regressor ``φ_t = [Δy_t; ...; Δy_{t-M+1}; Δu_t; ...; Δu_{t-M+1}]``
    of length ``M * (n_y + n_u)``.

    Args:
        n_y: Dimension of the observed/tracked output vector y.
        n_u: Dimension of the control input vector u.
        forgetting: RLS forgetting factor α ∈ (0, 1]. Values <1 give more
            weight to recent samples; ``1.0`` recovers ordinary least
            squares. Typical flight-control values: 0.995–0.9999.
        cov_init: Positive scalar for P₀ = cov_init · I, or its positive
            diagonal entries, ordered as M blocks of n_y state increments,
            then M blocks of n_u input increments. Separate entries allow
            different physical units: using normalized regressor S·phi with
            covariance c·I is equivalent to physical covariance c·S².
            Larger values reduce prior regularization; they do not guarantee
            accurate identification without informative measurements.
        theta_init_scale: Standard deviation of the zero-mean Gaussian
            used to randomly initialise ``theta``. A small non-zero value
            helps break the symmetry when later matrix operations are
            used (e.g. ``Bᵀ B``).
        seed: Optional random seed for ``theta`` initialisation.
        history_length: Measured increment window M. RLS waits for a full
            window before changing its parameters.
    """

    def __init__(
        self,
        n_y: int,
        n_u: int,
        forgetting: float = 0.999,
        cov_init: float | Sequence[float] = 1e2,
        theta_init_scale: float = 1e-3,
        seed: int | None = None,
        history_length: int = 1,
    ) -> None:
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting factor must be in (0, 1]")
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.alpha = float(forgetting)
        self._rng = np.random.default_rng(seed)

        if int(history_length) != history_length or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        self.history_length = int(history_length)
        self.dy_history: list[np.ndarray] = []
        self.du_history: list[np.ndarray] = []
        n_theta = self.history_length * (self.n_y + self.n_u)
        self.theta = self._rng.normal(0.0, theta_init_scale, size=(n_theta, self.n_y))
        covariance = np.asarray(cov_init, dtype=np.float64)
        if (
            (covariance.ndim != 0 and covariance.shape != (n_theta,))
            or not np.isfinite(covariance).all()
            or np.any(covariance <= 0)
        ):
            raise ValueError(
                "cov_init must be positive and finite: a scalar or one diagonal entry per regressor"
            )
        self.cov_init: float | np.ndarray = (
            float(covariance) if covariance.ndim == 0 else covariance.copy()
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
        """Return the identified ``A`` matrix (shape ``(n_y, M * n_y)``)."""
        return self.theta[: self.history_length * self.n_y, :].T

    @property
    def B(self) -> np.ndarray:
        """Return the identified ``B`` matrix (shape ``(n_y, M * n_u)``)."""
        return self.theta[self.history_length * self.n_y :, :].T

    def reset(self) -> None:
        """Reset the parameter buffers (keeps ``theta`` and ``P``)."""
        self.y_prev = None
        self.y_prev2 = None
        self.u_prev = None
        self.u_prev2 = None
        self.last_prediction_error = None
        self.dy_history.clear()
        self.du_history.clear()

    def reset_covariance(self) -> None:
        """Reset ``P`` to its initial large-variance state (full re-learn)."""
        n_theta = self.history_length * (self.n_y + self.n_u)
        self.P = np.eye(n_theta, dtype=np.float64) * self.cov_init

    def update(
        self,
        y_prev: np.ndarray,
        y_curr: np.ndarray,
        y_next: np.ndarray,
        u_prev: np.ndarray,
        u_curr: np.ndarray,
    ) -> np.ndarray:
        """Perform one RLS step, retaining M-1 preceding increments internally.

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

        if any(y.size != self.n_y for y in (y_prev_v, y_curr_v, y_next_v)):
            raise ValueError(f"expected observations of length {self.n_y}")
        if u_prev_v.size != self.n_u or u_curr_v.size != self.n_u:
            raise ValueError(f"expected controls of length {self.n_u}")

        samples = (y_prev_v, y_curr_v, y_next_v, u_prev_v, u_curr_v)
        if not all(np.isfinite(value).all() for value in samples):
            raise ValueError("identification samples must be finite")
        dy, du = y_curr_v - y_prev_v, u_curr_v - u_prev_v
        phi = self.regressor(y_curr_v, y_prev_v, u_curr_v, u_prev_v)
        eps = y_next_v - y_curr_v - self.theta.T @ phi
        if len(self.dy_history) >= self.history_length - 1:
            Pphi = self.P @ phi
            gain = Pphi / (self.alpha + float(phi @ Pphi))
            theta = self.theta + np.outer(gain, eps)
            # Joseph form of Eqs. (36)--(38), preserving numerical positivity.
            residual = np.eye(len(phi)) - np.outer(gain, phi)
            covariance = residual @ self.P @ residual.T / self.alpha + np.outer(
                gain, gain
            )
            if not np.isfinite(theta).all() or not np.isfinite(covariance).all():
                raise FloatingPointError("RLS update is nonfinite")
            self.theta = theta
            self.P = 0.5 * (covariance + covariance.T)
            self.num_updates += 1
        keep = self.history_length - 1
        self.dy_history = [dy.copy(), *self.dy_history][:keep]
        self.du_history = [du.copy(), *self.du_history][:keep]
        self.last_prediction_error = eps
        return np.asarray(eps)

    def regressor(self, y_curr, y_prev, u_curr, u_prev) -> np.ndarray:
        """Newest-first error/input increments from Eq. (31).

        Missing initial history is zero for prediction; identification waits
        for a complete measured window, without fitting fictitious samples.
        """
        dy = [np.asarray(y_curr).ravel() - np.asarray(y_prev).ravel(), *self.dy_history]
        du = [np.asarray(u_curr).ravel() - np.asarray(u_prev).ravel(), *self.du_history]
        dy += [np.zeros(self.n_y)] * (self.history_length - len(dy))
        du += [np.zeros(self.n_u)] * (self.history_length - len(du))
        return np.concatenate([*dy, *du])

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

        return np.asarray(y_c + self.theta.T @ self.regressor(y_c, y_p, u_c, u_p))
