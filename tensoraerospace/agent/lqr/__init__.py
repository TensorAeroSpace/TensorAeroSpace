"""Discrete linear-quadratic feedback, including caller-supplied LQI augmentation."""

import numpy as np
from scipy.linalg import solve_discrete_are


class LQRAgent:
    """Design ``u = -K @ (state-reference)`` for a discrete linear model.

    Supply A/B and Q/R in consistent state/input units. Integral states can be
    included in A/B for LQI; the caller advances those states from measurements.
    This class returns the requested input. Physical magnitude/slew limits are
    enforced by the plant or controller integration, not the Riccati equation.
    """

    def __init__(self, A, B, Q, R):
        self.A, self.B, self.Q, self.R = [
            np.asarray(value, dtype=float).copy() for value in (A, B, Q, R)
        ]
        if (
            self.A.ndim != 2
            or self.A.shape[0] != self.A.shape[1]
            or self.A.shape[0] == 0
        ):
            raise ValueError("A must be a nonempty square matrix")
        n = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != n or self.B.shape[1] == 0:
            raise ValueError("B must have n_state rows and at least one input")
        m = self.B.shape[1]
        if self.Q.shape != (n, n) or self.R.shape != (m, m):
            raise ValueError("Q/R shapes must match the state/input dimensions")
        if not all(
            np.isfinite(value).all() for value in (self.A, self.B, self.Q, self.R)
        ):
            raise ValueError("Model and weights must be finite")
        if not np.allclose(self.Q, self.Q.T) or not np.allclose(self.R, self.R.T):
            raise ValueError("Q and R must be symmetric")
        if (
            np.linalg.eigvalsh(self.Q).min() < -1e-12
            or np.linalg.eigvalsh(self.R).min() <= 0
        ):
            raise ValueError("Q must be positive semidefinite and R positive definite")
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(
            self.R + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A
        )

    def predict(self, state, reference=None):
        """Requested control; reference is a full equilibrium state vector."""
        state = np.asarray(state, dtype=float)
        if state.shape != (self.A.shape[0],) or not np.isfinite(state).all():
            raise ValueError("state must be a finite n_state vector")
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            if reference.shape != state.shape or not np.isfinite(reference).all():
                raise ValueError("reference must match state shape and be finite")
            state = state - reference
        return -self.K @ state


__all__ = ["LQRAgent"]
