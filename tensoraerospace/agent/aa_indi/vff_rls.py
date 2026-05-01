"""Variable-Forgetting-Factor Recursive Least Squares identifier.

Used by :mod:`tensoraerospace.agent.aa_indi` to track the control-effectiveness
matrix ``G = ∂ω̇/∂u`` online while rejecting noise-driven drift and reacting
fast to abrupt changes — the key adaptive ingredient of the AA-INDI scheme in
Sun & van Kampen's TU Delft line of work on fault-tolerant INDI.

Regressor and target are formed in the incremental form::

    Δω̇ ≈ G · Δu,      φ = Δu,   y = Δω̇

Variable forgetting factor λ_k shrinks toward ``forgetting_min`` when the
prediction residual grows (fast adaptation during faults / manoeuvres) and
relaxes back toward ``forgetting_max`` during quiet operation (noise
rejection). We use the Fortescue–Kershenbaum closed-form update based on the
residual-to-noise ratio::

    λ_k = exp(−‖ε‖² / σ²_ε),   clamped to [λ_min, λ_max].
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
        if eps_sensitivity <= 0.0:
            raise ValueError("eps_sensitivity must be > 0")
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

        phi = du_v.reshape(-1, 1)

        # Prediction residual before the update.
        pred = self.theta.T @ phi  # (n_y, 1)
        eps = dy_v - pred.reshape(-1)
        self.last_prediction_error = eps

        # Variable forgetting factor: exponential fall-off in the residual
        # energy, clamped into [lam_min, lam_max].
        eps_norm_sq = float(eps @ eps)
        lam = float(
            np.clip(
                np.exp(-eps_norm_sq / (self.eps_sensitivity**2)),
                self.lam_min,
                self.lam_max,
            )
        )
        self.last_lambda = lam

        # Gain / covariance recursion.
        Pphi = self.P @ phi  # (n_u, 1)
        denom = lam + float((phi.T @ Pphi).item())
        K = Pphi / denom  # (n_u, 1)

        self.theta = self.theta + K @ eps.reshape(1, -1)
        self.P = (self.P - K @ Pphi.T) / lam
        # Numerical symmetrisation — cheap drift insurance.
        self.P = 0.5 * (self.P + self.P.T)

        self.num_updates += 1
        return np.asarray(eps)

    def predict(self, du: np.ndarray) -> np.ndarray:
        """Predict Δω̇ from a candidate control increment Δu."""
        du_v = np.asarray(du, dtype=np.float64).reshape(-1)
        return (self.theta.T @ du_v.reshape(-1, 1)).reshape(-1)
