"""State-feedback MRAC augmentation of a discrete LQR controller."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import LQRAgent


@dataclass
class LQRMRACConfig:
    """Adaptation settings in the units of the supplied discrete model."""

    dt: float = 0.01
    adaptation_gain: float = 1.0
    deadzone: float = 0.0
    normalization: float = 1.0
    parameter_limit: float = 100.0
    adaptive_limit: float = 10.0
    adaptation_enabled: bool = True

    def __post_init__(self):
        positive = (
            self.dt,
            self.normalization,
            self.parameter_limit,
            self.adaptive_limit,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("dt, normalization and limits must be finite and positive")
        if not np.isfinite(self.adaptation_gain) or self.adaptation_gain < 0:
            raise ValueError("adaptation_gain must be finite and nonnegative")
        if not np.isfinite(self.deadzone) or self.deadzone < 0:
            raise ValueError("deadzone must be finite and nonnegative")


class LQRMRACAgent:
    """Add bounded direct MRAC compensation to a baseline :class:`LQRAgent`.

    The reference model is the healthy closed loop ``A-BK`` driven by the same
    feedforward term as the baseline controller. Zero error on an exact model
    requires a synchronized :meth:`reset` and no external command saturation.
    Full state measurements are required. The adaptive term compensates matched
    uncertainty; its update is normalized, projected and may be disabled without
    changing the LQR feedback. These safeguards bound the adaptive command but do
    not prove stability of a nonlinear closed-loop system.
    """

    def __init__(self, baseline: LQRAgent, config: LQRMRACConfig | None = None):
        self.baseline = baseline
        self.cfg = config or LQRMRACConfig()
        self.A = baseline.A
        self.B = baseline.B
        self.K = baseline.K
        self.n_state = self.A.shape[0]
        self.n_control = self.B.shape[1]
        self.reference_dynamics: np.ndarray = self.A - self.B @ self.K
        self.P = solve_discrete_lyapunov(
            self.reference_dynamics.T, np.eye(self.n_state)
        )
        self.theta = np.zeros((self.n_state + 1, self.n_control))
        self.reference_state = np.zeros(self.n_state)
        self._last_error = np.zeros(self.n_state)
        self._last_adaptive = np.zeros(self.n_control)

    def _vector(self, value, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (self.n_state,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite n_state vector")
        return array

    def _features(self, state):
        return np.concatenate([state, [1.0]])

    def predict(self, state, reference=None):
        """Return the baseline LQR command plus bounded adaptive compensation."""
        state = self._vector(state, "state")
        baseline = self.baseline.predict(state, reference)
        adaptive = -self.theta.T @ self._features(state)
        norm = np.linalg.norm(adaptive)
        if norm > self.cfg.adaptive_limit:
            adaptive = adaptive * (self.cfg.adaptive_limit / norm)
        self._last_adaptive = adaptive
        return baseline + adaptive

    def update(self, state, next_state, reference=None):
        """Advance the reference model and update parameters from measured states."""
        state = self._vector(state, "state")
        next_state = self._vector(next_state, "next_state")
        if reference is None:
            reference = np.zeros(self.n_state)
        reference = self._vector(reference, "reference")
        reference_next = self.reference_dynamics @ self.reference_state
        reference_next += self.B @ (self.K @ reference)
        error = next_state - reference_next
        self._last_error = error
        if self.cfg.adaptation_enabled and np.linalg.norm(error) > self.cfg.deadzone:
            features = self._features(state)
            gradient = np.outer(features, self.B.T @ self.P @ error)
            self.theta += (
                self.cfg.dt
                * self.cfg.adaptation_gain
                * gradient
                / (self.cfg.normalization + features @ features)
            )
            norms = np.linalg.norm(self.theta, axis=0)
            scales = np.maximum(1.0, norms / self.cfg.parameter_limit)
            self.theta /= scales
        self.reference_state = reference_next

    def reset(self, state=None):
        """Reset adaptation and optionally initialize the reference state."""
        self.theta.fill(0.0)
        self.reference_state = (
            np.zeros(self.n_state)
            if state is None
            else self._vector(state, "state").copy()
        )
        self._last_error.fill(0.0)
        self._last_adaptive.fill(0.0)

    @property
    def diagnostics(self):
        """Return copies of the latest errors, command and parameter norm."""
        return {
            "reference_error": self._last_error.copy(),
            "adaptive_command": self._last_adaptive.copy(),
            "parameter_norm": float(np.linalg.norm(self.theta)),
        }


__all__ = ["LQRMRACAgent", "LQRMRACConfig"]
