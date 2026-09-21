"""Variable-Forgetting-Factor Recursive Least Squares identifier.

Used by :mod:`tensoraerospace.agent.aa_indi` for its simplified incremental
regression ``delta(omega_dot) = G @ delta(u)``. The gain and forgetting update
follow Atmaca, de Visser & van Kampen (AIAA 2026-1743), Eqs. (54)--(57).
The paper fits reconstructed aerodynamic moments to surface positions; the
regressors used by this agent are a separate simplification.

With ``a = delta(u)`` and ``Sigma_0 = eps_sensitivity**2``::

    K = P @ a / (1 + a.T @ P @ a)
    lambda = clip(1 - ||epsilon||**2 / (Sigma_0 * (1 + a.T @ P @ a)),
                  forgetting_min, forgetting_max)

``forgetting_max < 1`` is a library extension; the paper permits lambda = 1.
The residual norm shares one forgetting factor across multiple output channels.

"""

from __future__ import annotations

import numpy as np


class VFFRLSEstimator:
    """VFF-RLS identifier for the control-effectiveness matrix ``G``.

    Internally stores the parameter matrix ``theta`` of shape
    ``(n_u, n_y)`` such that ``y ≈ θᵀ · φ`` where ``φ = Δu`` (length
    ``n_u``) and ``y = Δω̇`` (length ``n_y``). Under that convention
    ``G = θᵀ``, which is the usual row-action convention for INDI.

    Args:
        n_y: Dimension of the output Δω̇.
        n_u: Dimension of the control increment Δu.
        forgetting_min: Lower bound on λ — reached under strong innovations
            (fast adaptation).
        forgetting_max: Upper bound on λ — reached under quiescent
            operation (noise rejection).
        eps_sensitivity: Scale of the residual norm at which λ falls off
            significantly. Smaller values make the forgetting factor
            more reactive to transients.
        cov_init: Initial scale of the covariance matrix
            ``P₀ = cov_init · I``.
        theta_init_scale: Standard deviation of a zero-mean Gaussian used
            to randomly initialise ``theta``. Small non-zero values
            break symmetry for downstream matrix operations.
        seed: RNG seed for the initial ``theta``.
    """

    def __init__(
        self,
        n_y: int,
        n_u: int,
        forgetting_min: float = 0.7,
        forgetting_max: float = 0.999,
        eps_sensitivity: float = 1.0,
        cov_init: float = 1e2,
        theta_init_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < forgetting_min <= forgetting_max <= 1.0:
            raise ValueError("need 0 < forgetting_min ≤ forgetting_max ≤ 1")
        if not np.isfinite(eps_sensitivity) or eps_sensitivity <= 0.0:
            raise ValueError("eps_sensitivity must be finite and > 0")
        if not np.isfinite(cov_init) or cov_init <= 0.0:
            raise ValueError("cov_init must be finite and > 0")
        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.lam_min = float(forgetting_min)
        self.lam_max = float(forgetting_max)
        self.eps_sensitivity = float(eps_sensitivity)
        self.cov_init = float(cov_init)

        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0.0, theta_init_scale, size=(self.n_u, self.n_y))
        self.P = np.eye(self.n_u, dtype=np.float64) * self.cov_init

        self.last_lambda: float = self.lam_max
        self.last_prediction_error: np.ndarray | None = None
        self.num_updates: int = 0

    @property
    def G(self) -> np.ndarray:
        """Return the control-effectiveness matrix of shape ``(n_y, n_u)``."""
        return self.theta.T

    def reset_covariance(self) -> None:
        """Restore ``P`` to its initial large-variance state."""
        self.P = np.eye(self.n_u, dtype=np.float64) * self.cov_init

    def update(self, du: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Run one VFF-RLS step.

        Args:
            du: Control increment ``u_t − u_{t-1}`` (shape ``(n_u,)``).
            dy: Measured output increment ``ω̇_t − ω̇_{t-1}`` (shape
                ``(n_y,)``).

        Returns:
            The prediction residual ``ε = dy − θᵀ du`` **before** the
            update.
        """
        du_v = np.asarray(du, dtype=np.float64).reshape(-1)
        dy_v = np.asarray(dy, dtype=np.float64).reshape(-1)
        if du_v.size != self.n_u:
            raise ValueError(f"du must have length {self.n_u}, got {du_v.size}")
        if dy_v.size != self.n_y:
            raise ValueError(f"dy must have length {self.n_y}, got {dy_v.size}")

        if not np.all(np.isfinite(du_v)) or not np.all(np.isfinite(dy_v)):
            raise ValueError("RLS regressor and target must be finite")

        # Atmaca et al. Eqs. 54--57: compute K before applying forgetting.
        # 1 - a @ K = 1 / (1 + a @ P @ a) avoids cancellation in lambda.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            eps = dy_v - self.theta.T @ du_v
            Pphi = self.P @ du_v
            denom = 1.0 + float(du_v @ Pphi)
            if denom <= 0.0:
                raise FloatingPointError(
                    "RLS covariance produced nonpositive gain denominator"
                )
            K = Pphi / denom
            scaled_residual = eps / self.eps_sensitivity / np.sqrt(denom)
            lam = float(
                np.clip(
                    1.0 - scaled_residual @ scaled_residual, self.lam_min, self.lam_max
                )
            )
            theta = self.theta + np.outer(K, eps)
            residual_map = np.eye(self.n_u) - np.outer(K, du_v)
            covariance = (residual_map @ self.P @ residual_map.T + np.outer(K, K)) / lam
            covariance = 0.5 * covariance + 0.5 * covariance.T
        if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(covariance)):
            raise FloatingPointError(
                "Nonfinite RLS update; check scaling and excitation"
            )
        self.theta = theta
        self.P = covariance
        self.last_prediction_error = eps.copy()
        self.last_lambda = lam
        self.num_updates += 1
        return np.asarray(eps)

    def predict(self, du: np.ndarray) -> np.ndarray:
        """Predict Δω̇ from a candidate control increment Δu."""
        du_v = np.asarray(du, dtype=np.float64).reshape(-1)
        return (self.theta.T @ du_v.reshape(-1, 1)).reshape(-1)
