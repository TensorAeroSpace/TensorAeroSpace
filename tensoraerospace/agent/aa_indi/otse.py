"""Optimal two-stage EKF algebra (Atmaca 2025, Eqs. 20--42).

The state is represented as x = x_bar + V b with independent conditional
state covariance P_bar and bias covariance P_b. This factorization is checked
against an augmented Kalman filter, including time-varying bias and correlated
process noise. Matrices passed to predict are DISCRETE, including Q and B.
"""

from __future__ import annotations

import numpy as np


def covariance(value: np.ndarray, n: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite ({n}, {n}) matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    matrix = (matrix + matrix.T) * 0.5
    if np.linalg.eigvalsh(matrix).min() < -1e-10 * max(
        1.0, float(np.linalg.norm(matrix))
    ):
        raise ValueError(f"{name} must be positive semidefinite")
    return np.asarray(matrix.copy())


class OptimalTwoStageEKF:
    """Two-stage state/input-bias estimator with caller-supplied linearization."""

    def __init__(
        self,
        state: np.ndarray,
        state_covariance: np.ndarray,
        bias_covariance: np.ndarray,
    ) -> None:
        self.x_bar = np.asarray(state, dtype=float).copy()
        if self.x_bar.ndim != 1 or not np.isfinite(self.x_bar).all():
            raise ValueError("state must be a finite vector")
        self.n = len(self.x_bar)
        self.m = len(bias_covariance)
        self.P_bar = covariance(state_covariance, self.n, "state_covariance")
        self.P_bias = covariance(bias_covariance, self.m, "bias_covariance")
        if np.linalg.eigvalsh(self.P_bias).min() <= 0:
            raise ValueError("initial bias covariance must be positive definite")
        self.bias = np.zeros(self.m)
        self.V = np.zeros((self.n, self.m))

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self.x_bar + self.V @ self.bias)

    @property
    def state_covariance(self) -> np.ndarray:
        return np.asarray(self.P_bar + self.V @ self.P_bias @ self.V.T)

    def predict(
        self,
        nominal_next: np.ndarray,
        F: np.ndarray,
        B: np.ndarray,
        Q_state: np.ndarray,
        Q_bias: np.ndarray,
        Q_cross: np.ndarray | None = None,
    ) -> None:
        """Propagate x'=f(x)+B b+w, b'=b+eta.

        ``nominal_next`` is f evaluated at the CURRENT reconstructed state;
        ``F`` is its Jacobian there. This convention makes nonlinear EKF
        linearization consistent with the independently augmented filter.
        """
        F, B = np.asarray(F, dtype=float), np.asarray(B, dtype=float)
        nominal_next = np.asarray(nominal_next, dtype=float)
        if (
            F.shape != (self.n, self.n)
            or B.shape != (self.n, self.m)
            or nominal_next.shape != (self.n,)
            or not all(np.isfinite(a).all() for a in (F, B, nominal_next))
        ):
            raise ValueError("invalid transition, bias map or predicted state")
        Q = covariance(Q_state, self.n, "Q_state")
        Qb = covariance(Q_bias, self.m, "Q_bias")
        Qxb = (
            np.zeros((self.n, self.m))
            if Q_cross is None
            else np.asarray(Q_cross, dtype=float)
        )
        covariance(
            np.block([[Q, Qxb], [Qxb.T, Qb]]),
            self.n + self.m,
            "joint process covariance",
        )
        # Eqs. (35)--(39), expressed as a conditional covariance to expose
        # the exact cross-covariance and avoid any timing ambiguity.
        U_bar = F @ self.V + B
        Pb = self.P_bias + Qb
        cross = U_bar @ self.P_bias + Qxb
        U = np.linalg.solve(Pb, cross.T).T
        Ptotal = F @ self.P_bar @ F.T + U_bar @ self.P_bias @ U_bar.T + Q
        Pbar = covariance(Ptotal - U @ Pb @ U.T, self.n, "predicted P_bar")
        xbar = nominal_next + B @ self.bias - U @ self.bias
        self.x_bar, self.V, self.P_bar, self.P_bias = xbar, U, Pbar, Pb

    def correct(
        self,
        measurement: np.ndarray,
        predicted_measurement: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        angular_indices: tuple[int, ...] = (),
    ) -> None:
        """Measurement update; h and H are evaluated at the reconstructed state."""
        y, hy, H = (
            np.asarray(a, dtype=float) for a in (measurement, predicted_measurement, H)
        )
        d = len(y)
        if (
            y.shape != (d,)
            or hy.shape != (d,)
            or H.shape != (d, self.n)
            or not all(np.isfinite(a).all() for a in (y, hy, H))
        ):
            raise ValueError("invalid measurement or measurement linearization")
        R = covariance(R, d, "R")
        innovation = y - hy
        for index in angular_indices:
            innovation[index] = (innovation[index] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.V
        Sx = H @ self.P_bar @ H.T + R
        Kx = np.linalg.solve(Sx, H @ self.P_bar).T
        Sb = Sx + S @ self.P_bias @ S.T
        Kb = np.linalg.solve(Sb, S @ self.P_bias).T
        # Eq. (21) uses the conditional-state innovation; Eq. (33) removes
        # its predicted bias contribution. All updates use the same prior.
        b = self.bias + Kb @ innovation
        xbar = self.x_bar + Kx @ (innovation + S @ self.bias)
        V = self.V - Kx @ S
        residual_x = np.eye(self.n) - Kx @ H
        Pbar = residual_x @ self.P_bar @ residual_x.T + Kx @ R @ Kx.T
        residual_b = np.eye(self.m) - Kb @ S
        Pb = residual_b @ self.P_bias @ residual_b.T + Kb @ Sx @ Kb.T
        self.x_bar, self.bias, self.V = xbar, b, V
        self.P_bar = covariance(Pbar, self.n, "updated P_bar")
        self.P_bias = covariance(Pb, self.m, "updated P_bias")
