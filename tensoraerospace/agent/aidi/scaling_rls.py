"""Per-row VFF-RLS that adapts the multiplicative scaling Θ over a known onboard
control-effectiveness matrix G_nominal — Section III.C of Ul Haq et al. 2026.

For each rate axis ``i`` we keep an independent covariance ``P_i``, an
information-content forgetting factor ``λ_i`` (Eq. 26-27 of the paper) and a
row of the scaling matrix ``Θ[i, :]``. After per-row updates a cross-axis
consistency check replaces any deviating column entry with the column mean —
this matches the practical adjustment described at the top of page 10 of the
paper.
"""

from __future__ import annotations

import numpy as np


class ScalingRLS:
    """Recursive identifier of the multiplicative scaling matrix Θ.

    Args:
        n_y: Number of rate axes (rows of Θ).
        n_u: Number of control surfaces (columns of Θ).
        lambda_min: Lower bound on the variable forgetting factor — the
            estimator falls toward this value when the residual is large
            (fast adaptation during faults).
        lambda_max: Upper bound on the variable forgetting factor — the
            estimator returns toward this value during quiescent operation.
        sigma0: Sensor-noise standard deviation σ₀ used in the
            information-content VFF (Eq. 27 of the paper, Σ₀ = σ₀²·N₀).
        memory_length: Nominal memory length N₀ in samples.
        cov_init: Initial scale of the per-row covariance matrices.
        consistency_threshold: Per-paper relative threshold for the
            cross-axis consistency check; updates that deviate by more
            than this from the column mean are replaced by the mean.
        seed: Reserved for future stochastic variants — currently unused.
    """

    def __init__(
        self,
        n_y: int,
        n_u: int,
        lambda_min: float = 0.7,
        lambda_max: float = 0.999,
        sigma0: float = 1e-3,
        memory_length: int = 100,
        cov_init: float = 1.0,
        consistency_threshold: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < lambda_min <= lambda_max <= 1.0:
            raise ValueError("require 0 < lambda_min ≤ lambda_max ≤ 1")
        if sigma0 <= 0.0 or memory_length <= 0:
            raise ValueError("sigma0 and memory_length must be positive")
        if cov_init <= 0.0:
            raise ValueError("cov_init must be positive")
        del seed  # reserved.

        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.sigma0 = float(sigma0)
        self.memory_length = int(memory_length)
        self.cov_init = float(cov_init)
        self.consistency_threshold = float(consistency_threshold)

        self.theta = np.ones((self.n_y, self.n_u), dtype=np.float64)
        self.P = np.stack(
            [np.eye(self.n_u, dtype=np.float64) * self.cov_init
             for _ in range(self.n_y)],
            axis=0,
        )  # shape (n_y, n_u, n_u)
        self.last_lambda = np.full(self.n_y, self.lambda_max, dtype=np.float64)
        self.last_residual = np.zeros(self.n_y, dtype=np.float64)
        self.num_updates: int = 0
        self._last_G_scaled: np.ndarray | None = None

    @property
    def sigma_total(self) -> float:
        """Information-content denominator Σ₀ = σ₀²·N₀."""
        return self.sigma0 ** 2 * self.memory_length

    def _info_content_lambda(self, eps_i: float, phi_K_i: float) -> float:
        """Eq. 26 of the paper: λ = 1 − (1 − φᵀK)·ε² / Σ₀, clamped."""
        lam = 1.0 - (1.0 - phi_K_i) * (eps_i ** 2) / self.sigma_total
        return float(np.clip(lam, self.lambda_min, self.lambda_max))

    def _apply_consistency_check(self, delta_theta: np.ndarray) -> np.ndarray:
        """Replace column entries that deviate from the column mean by more than
        ``consistency_threshold`` with the column mean."""
        out = delta_theta.copy()
        col_mean = out.mean(axis=0)
        deviation = np.abs(out - col_mean[np.newaxis, :])
        mask = deviation > self.consistency_threshold
        mean_broadcast = np.broadcast_to(col_mean, out.shape)
        out = np.where(mask, mean_broadcast, out)
        return out

    def update(
        self,
        du: np.ndarray,
        domega: np.ndarray,
        G_nominal: np.ndarray,
    ) -> np.ndarray:
        """Run one RLS step using ``(Δu, Δω̇, G_nominal)``.

        Args:
            du: Control increment, shape ``(n_u,)``.
            domega: Angular-rate-derivative increment, shape ``(n_y,)``.
            G_nominal: Onboard CE matrix at the linearisation point, shape
                ``(n_y, n_u)``.

        Returns:
            The pre-update residual ε of shape ``(n_y,)``.
        """
        du_v = np.asarray(du, dtype=np.float64).reshape(-1)
        dy_v = np.asarray(domega, dtype=np.float64).reshape(-1)
        G = np.asarray(G_nominal, dtype=np.float64)
        if du_v.size != self.n_u:
            raise ValueError(f"du must have length {self.n_u}, got {du_v.size}")
        if dy_v.size != self.n_y:
            raise ValueError(f"domega must have length {self.n_y}, got {dy_v.size}")
        if G.shape != (self.n_y, self.n_u):
            raise ValueError(
                f"G_nominal must have shape ({self.n_y}, {self.n_u}), got {G.shape}"
            )

        delta_theta = np.zeros((self.n_y, self.n_u), dtype=np.float64)
        residuals = np.zeros(self.n_y, dtype=np.float64)
        lambdas = np.empty(self.n_y, dtype=np.float64)

        for i in range(self.n_y):
            phi = (G[i, :] * du_v).reshape(-1, 1)            # (n_u, 1)
            theta_row = self.theta[i, :].reshape(-1, 1)      # (n_u, 1)
            P_i = self.P[i]
            eps_i = float(dy_v[i] - (theta_row.T @ phi).item())
            P_phi = P_i @ phi                                # (n_u, 1)
            denom = self.last_lambda[i] + float((phi.T @ P_phi).item())
            denom = denom if abs(denom) > 1e-12 else 1e-12
            K_i = P_phi / denom                              # (n_u, 1)

            delta_theta[i, :] = (K_i * eps_i).reshape(-1)
            phi_K_scalar = float((phi.T @ K_i).item())
            lam_new = self._info_content_lambda(eps_i, phi_K_scalar)
            lambdas[i] = lam_new

            self.P[i] = (P_i - K_i @ P_phi.T) / lam_new
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)  # symmetrise

            residuals[i] = eps_i

        delta_theta = self._apply_consistency_check(delta_theta)
        self.theta = self.theta + delta_theta
        self.last_lambda = lambdas
        self.last_residual = residuals
        self.num_updates += 1
        self._last_G_scaled = self.theta * G
        return residuals
