"""Adaptive 3-step Kalman filter for UFTC FDD.

Lu, P. et al. (2015) "Adaptive three-step Kalman filter for air-data
sensor fault detection," AIAA JGCD — adaptive Q, R via Sage-Husa
exponentially-weighted innovation/residual covariance updates.

The filter assumes the plant is locally linear in the state with a
known control map ``G``::

    x_{t+1} ≈ x_t + F · x_t + G · u_t  (incremental form)

Both ``F`` and ``G`` may be warm-started by ``UFTCController`` once the
incremental RLS inside :class:`IADPAgent` has converged on nominal
flight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanStep:
    """One-step output of :class:`NominalKalman`."""

    x_hat: np.ndarray  # posterior state estimate (n_state,)
    nu: np.ndarray     # innovation y - H·x_hat_prior (n_state,)
    S: np.ndarray      # innovation covariance (n_state, n_state)
    K: np.ndarray      # Kalman gain (n_state, n_state)


class NominalKalman:
    """Adaptive 3-step Kalman filter on a linear nominal plant.

    Args:
        F_nominal: System matrix increment, shape ``(n_state, n_state)``.
        G_nominal: Control map, shape ``(n_state, n_control)``.
        Q: Process noise covariance, shape ``(n_state, n_state)``.
        R: Measurement noise covariance, shape ``(n_state, n_state)``.
        alpha_Q: EMA coefficient for adaptive Q (default 0.99 — slow).
        alpha_R: EMA coefficient for adaptive R (default 0.99).
        adapt_Q: Enable Sage-Husa Q adaptation (default True).
        adapt_R: Enable Sage-Husa R adaptation (default True).
    """

    def __init__(
        self,
        F_nominal: np.ndarray,
        G_nominal: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        *,
        alpha_Q: float = 0.99,
        alpha_R: float = 0.99,
        adapt_Q: bool = True,
        adapt_R: bool = True,
    ) -> None:
        self.F = np.array(F_nominal, dtype=np.float64, copy=True)
        self.G = np.array(G_nominal, dtype=np.float64, copy=True)
        self.Q = np.array(Q, dtype=np.float64, copy=True)
        self.R = np.array(R, dtype=np.float64, copy=True)
        self.alpha_Q = float(alpha_Q)
        self.alpha_R = float(alpha_R)
        self.adapt_Q = bool(adapt_Q)
        self.adapt_R = bool(adapt_R)

        self.n_state = self.F.shape[0]
        self.n_control = self.G.shape[1]
        if self.F.shape != (self.n_state, self.n_state):
            raise ValueError("F_nominal must be square")
        if self.G.shape != (self.n_state, self.n_control):
            raise ValueError("G_nominal must be (n_state, n_control)")
        if self.Q.shape != (self.n_state, self.n_state):
            raise ValueError("Q must match n_state")
        if self.R.shape != (self.n_state, self.n_state):
            raise ValueError("R must match n_state")

        self._reset_state()

    def _reset_state(self) -> None:
        self.x_hat = np.zeros(self.n_state, dtype=np.float64)
        self.P = np.eye(self.n_state, dtype=np.float64)

    def reset(self) -> None:
        """Restore filter state to zero / identity covariance."""
        self._reset_state()

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        """Replace ``F``/``G`` with refreshed estimates of nominal dynamics."""
        if F_nominal is not None:
            F_arr = np.array(F_nominal, dtype=np.float64)
            if F_arr.shape != self.F.shape:
                raise ValueError(
                    f"F_nominal shape mismatch: {F_arr.shape} vs {self.F.shape}"
                )
            self.F = F_arr
        if G_nominal is not None:
            G_arr = np.array(G_nominal, dtype=np.float64)
            if G_arr.shape != self.G.shape:
                raise ValueError(
                    f"G_nominal shape mismatch: {G_arr.shape} vs {self.G.shape}"
                )
            self.G = G_arr

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> KalmanStep:
        """Run one Kalman update.

        Args:
            x_meas: Measured state at time t, shape ``(n_state,)``.
            u_prev: Control applied at t-1, shape ``(n_control,)``.

        Returns:
            KalmanStep with posterior x_hat, innovation, S, K.
        """
        x = np.asarray(x_meas, dtype=np.float64).reshape(-1)
        u = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        if x.size != self.n_state:
            raise ValueError(f"x_meas must have length {self.n_state}, got {x.size}")
        if u.size != self.n_control:
            raise ValueError(f"u_prev must have length {self.n_control}, got {u.size}")

        # Step 1: prior prediction (incremental form).
        x_prior = self.x_hat + self.F @ self.x_hat + self.G @ u
        F_jac = np.eye(self.n_state) + self.F
        P_prior = F_jac @ self.P @ F_jac.T + self.Q

        # Step 2: innovation.
        nu = x - x_prior  # H = I (full-state measurement)
        S = P_prior + self.R
        # Solve K via S, falling back to pinv on rank deficiency.
        try:
            K = np.linalg.solve(S.T, P_prior.T).T
        except np.linalg.LinAlgError:
            K = P_prior @ np.linalg.pinv(S)

        # Step 3: posterior correction.
        x_post = x_prior + K @ nu
        I_KH = np.eye(self.n_state) - K  # H = I
        P_post = I_KH @ P_prior @ I_KH.T + K @ self.R @ K.T

        # Sage-Husa adaptive Q, R. Both updates are EMA on outer products.
        if self.adapt_R:
            residual = x - x_post  # H·x_post = x_post
            self.R = self.alpha_R * self.R + (1.0 - self.alpha_R) * (
                np.outer(residual, residual) + I_KH @ P_prior @ I_KH.T
            )
            # Keep R symmetric and positive-definite-ish.
            self.R = 0.5 * (self.R + self.R.T)

        if self.adapt_Q:
            dx = K @ nu
            self.Q = self.alpha_Q * self.Q + (1.0 - self.alpha_Q) * (
                np.outer(dx, dx) + P_post - F_jac @ self.P @ F_jac.T
            )
            self.Q = 0.5 * (self.Q + self.Q.T)

        # Persist posterior state.
        self.x_hat = x_post
        self.P = 0.5 * (P_post + P_post.T)

        return KalmanStep(
            x_hat=self.x_hat.copy(), nu=nu.copy(), S=S.copy(), K=K.copy()
        )
