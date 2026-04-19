"""Fixed-forgetting-factor Recursive Least Squares identifier for iADP.

Tracks the parameters of the incremental linear plant model introduced in
the iADP pseudo-code of the TU Delft flight-test paper (Konatala et al.,
AIAA SCITECH 2024, DOI 10.2514/6.2024-2402)::

    ΔX_{t+1} ≈ F̃_t · ΔX_t + G̃_t · Δδ_t.

With the stacked regressor ``W_t = [ΔX_t; Δδ_t]`` and parameter matrix
``Θ̃_t = [F̃_t; G̃_t]^T`` the model becomes ``ΔX_{t+1} ≈ Θ̃_t^T · W_t``
and the standard RLS recursion with a constant forgetting factor
``γ_RLS`` applies directly::

    ε_t  = ΔX̂_t − Θ̃_{t-1}^T · W_{t-1},
    K_t  = (Φ_{t-1} · W_{t-1}) / (γ_RLS + W_{t-1}^T · Φ_{t-1} · W_{t-1}),
    Θ̃_t = Θ̃_{t-1} + K_t · ε_t^T,
    Φ_t  = (Φ_{t-1} − K_t · W_{t-1}^T · Φ_{t-1}) / γ_RLS.

Unlike the VFF-RLS used in :mod:`tensoraerospace.agent.aa_indi`, the iADP
paper uses a *constant* forgetting factor tuned offline via MOPS: the
model-learning loop runs at 1 kHz on the real aircraft, so the data flow
itself is sufficient to keep the estimate fresh without a residual-driven
adaptive factor.
"""

from __future__ import annotations

import numpy as np


class IncrementalRLS:
    """Fixed-forgetting RLS identifier for ``Θ̃ = [F̃; G̃]^T``.

    Args:
        n_output: Size of the target vector ``ΔX̂_{t+1}``.
        n_regressor: Size of the regressor ``W_t = [ΔX_t; Δδ_t]``, i.e.
            ``n_output + n_control``.
        gamma_rls: Constant forgetting factor, in ``(0, 1]``. Closer to
            ``1`` means a longer effective memory; the paper uses values
            around ``0.95–0.999`` for the PH-LAB flight test.
        phi_init: Initial covariance scale (``Φ₀ = phi_init · I``).
            Large values advertise high initial uncertainty and let the
            first handful of updates move the parameters aggressively.
    """

    def __init__(
        self,
        n_output: int,
        n_regressor: int,
        gamma_rls: float = 0.99,
        phi_init: float = 1e2,
    ) -> None:
        if not 0.0 < gamma_rls <= 1.0:
            raise ValueError("gamma_rls must lie in (0, 1]")
        if phi_init <= 0.0:
            raise ValueError("phi_init must be > 0")
        self.n_output = int(n_output)
        self.n_regressor = int(n_regressor)
        self.gamma_rls = float(gamma_rls)
        self.phi_init_value = float(phi_init)

        self.theta = np.zeros((self.n_regressor, self.n_output), dtype=np.float64)
        self.Phi = np.eye(self.n_regressor, dtype=np.float64) * self.phi_init_value

        self.last_residual: np.ndarray = np.zeros(self.n_output, dtype=np.float64)
        self.num_updates: int = 0

    def predict(self, W: np.ndarray) -> np.ndarray:
        """One-step prediction ``ΔX̂_{t+1} = Θ̃^T · W``."""
        W_v = np.asarray(W, dtype=np.float64).reshape(-1)
        if W_v.size != self.n_regressor:
            raise ValueError(f"W must have length {self.n_regressor}, got {W_v.size}")
        return (self.theta.T @ W_v).reshape(-1)

    def update(self, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run one RLS step.

        Args:
            W: Regressor ``W_{t-1} = [ΔX_{t-1}; Δδ_{t-1}]``.
            y: Observed target ``ΔX̂_t``.

        Returns:
            Prediction residual ``ε_t = y − Θ̃_{t-1}^T · W`` **before**
            the update, for diagnostics and downstream gating.
        """
        W_v = np.asarray(W, dtype=np.float64).reshape(-1)
        y_v = np.asarray(y, dtype=np.float64).reshape(-1)
        if W_v.size != self.n_regressor:
            raise ValueError(f"W must have length {self.n_regressor}, got {W_v.size}")
        if y_v.size != self.n_output:
            raise ValueError(f"y must have length {self.n_output}, got {y_v.size}")

        eps = y_v - (self.theta.T @ W_v)
        self.last_residual = eps.copy()

        PhiW = self.Phi @ W_v  # (n_regressor,)
        denom = float(self.gamma_rls + W_v @ PhiW)
        if denom <= 0.0:
            denom = 1e-12
        K = PhiW / denom  # (n_regressor,)

        # Rank-1 parameter update.
        self.theta = self.theta + np.outer(K, eps)
        # Covariance update with constant forgetting.
        self.Phi = (self.Phi - np.outer(K, PhiW)) / self.gamma_rls
        # Cheap numerical symmetrisation.
        self.Phi = 0.5 * (self.Phi + self.Phi.T)

        self.num_updates += 1
        return eps

    def reset_covariance(self) -> None:
        """Restore ``Φ`` to its initial large-variance state."""
        self.Phi = np.eye(self.n_regressor, dtype=np.float64) * self.phi_init_value

    def extract_F(self, n_state: int) -> np.ndarray:
        """Return ``F̃`` of shape ``(n_state, n_state)`` from the first
        ``n_state`` rows of ``Θ̃``."""
        return self.theta[:n_state, :].T.copy()

    def extract_G(self, n_state: int) -> np.ndarray:
        """Return ``G̃`` of shape ``(n_state, n_control)`` from the rows
        of ``Θ̃`` after the state block."""
        return self.theta[n_state:, :].T.copy()
