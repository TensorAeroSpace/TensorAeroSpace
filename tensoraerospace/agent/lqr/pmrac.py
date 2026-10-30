"""Predictor-based MRAC augmentation following Lavretsky et al."""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, logm, solve_discrete_lyapunov

from . import LQRAgent


@dataclass
class LQRPMRACConfig:
    """PMRAC settings in the units of the supplied discrete model."""

    dt: float = 0.01
    adaptation_gain: float = 1.0
    deadzone: float = 0.0
    normalization: float = 1.0
    parameter_limit: float = 100.0
    adaptive_limit: float = 10.0
    adaptation_enabled: bool = True
    predictor_ratio: float = 3.0
    rbf_width: float = 2.0
    rbf_centers: np.ndarray = field(default_factory=lambda: np.arange(-10.0, 10.1, 2.0))

    def __post_init__(self):
        positive = (
            self.dt,
            self.normalization,
            self.parameter_limit,
            self.adaptive_limit,
            self.predictor_ratio,
            self.rbf_width,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("dt, normalization, ratio and limits must be positive")
        if not np.isfinite(self.adaptation_gain) or self.adaptation_gain < 0:
            raise ValueError("adaptation_gain must be finite and nonnegative")
        if not np.isfinite(self.deadzone) or self.deadzone < 0:
            raise ValueError("deadzone must be finite and nonnegative")
        self.rbf_centers = np.asarray(self.rbf_centers, dtype=float).copy()
        if (
            self.rbf_centers.ndim != 1
            or not len(self.rbf_centers)
            or not np.isfinite(self.rbf_centers).all()
        ):
            raise ValueError("rbf_centers must be a nonempty finite vector")


class LQRPMRACAgent:
    """Add predictor-based direct adaptive compensation to discrete LQR.

    The reference model is the nominal LQR closed loop. The predictor adds the
    prediction error to the adaptive laws, unlike conventional MRAC. RBFs act on
    the first state coordinate. This implementation addresses matched uncertainty.
    The predictor is discretized exactly; adaptive updates use Euler steps,
    normalization and projection. These safeguards do not prove stability of a
    nonlinear closed-loop system.
    """

    def __init__(self, baseline: LQRAgent, config: LQRPMRACConfig | None = None):
        self.baseline = baseline
        self.cfg = config or LQRPMRACConfig()
        self.A = baseline.A
        self.B = baseline.B
        self.K = baseline.K
        self.n_state = self.A.shape[0]
        self.n_control = self.B.shape[1]
        self.reference_dynamics = self.A - self.B @ self.K
        reference_continuous = logm(self.reference_dynamics) / self.cfg.dt
        if not np.allclose(reference_continuous.imag, 0.0, atol=1e-10):
            raise ValueError("PMRAC requires real logarithm of the reference dynamics")
        reference_continuous = reference_continuous.real
        predictor_continuous = self.cfg.predictor_ratio * reference_continuous
        augmented = np.block(
            [
                [predictor_continuous, reference_continuous - predictor_continuous],
                [np.zeros((self.n_state, 2 * self.n_state))],
            ]
        )
        transition = expm(self.cfg.dt * augmented)
        self.predictor_dynamics = transition[: self.n_state, : self.n_state]
        self.predictor_error_input = transition[
            : self.n_state, self.n_state : 2 * self.n_state
        ]
        self.P_ref = solve_discrete_lyapunov(
            self.reference_dynamics.T, np.eye(self.n_state)
        )
        self.P_prd = solve_discrete_lyapunov(
            self.predictor_dynamics.T, np.eye(self.n_state)
        )
        self.state_gains = np.zeros((self.n_control, self.n_state))
        self.theta = np.zeros((len(self.cfg.rbf_centers), self.n_control))
        self.reference_state = np.zeros(self.n_state)
        self.predictor_state = np.zeros(self.n_state)
        self._last_error = np.zeros(self.n_state)
        self._last_prediction_error = np.zeros(self.n_state)
        self._last_adaptive = np.zeros(self.n_control)

    def _vector(self, value, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (self.n_state,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be a finite n_state vector")
        return array

    def _features(self, state):
        return np.exp(
            -0.5 * ((state[0] - self.cfg.rbf_centers) / self.cfg.rbf_width) ** 2
        )

    def predict(self, state, reference=None):
        """Return baseline LQR plus bounded state and RBF compensation."""
        state = self._vector(state, "state")
        baseline = self.baseline.predict(state, reference)
        adaptive = -self.state_gains @ state - self.theta.T @ self._features(state)
        norm = np.linalg.norm(adaptive)
        if norm > self.cfg.adaptive_limit:
            adaptive = adaptive * (self.cfg.adaptive_limit / norm)
        self._last_adaptive = adaptive
        return baseline + adaptive

    def update(self, state, next_state, reference=None):
        """Update adaptive parameters, then advance reference and predictor."""
        state = self._vector(state, "state")
        next_state = self._vector(next_state, "next_state")
        if reference is None:
            reference = np.zeros(self.n_state)
        reference = self._vector(reference, "reference")
        command_input = self.B @ (self.K @ reference)
        predictor_offset = self.predictor_state - self.reference_state
        tracking_offset = state - self.reference_state
        offset_next = (
            self.predictor_dynamics @ predictor_offset
            + self.predictor_error_input @ tracking_offset
        )
        reference_next = self.reference_dynamics @ self.reference_state + command_input
        predictor_next = reference_next + offset_next
        error = next_state - reference_next
        prediction_error = predictor_next - next_state
        self._last_error = error
        self._last_prediction_error = prediction_error
        training = self.B.T @ (self.P_ref @ error - self.P_prd @ prediction_error)
        if self.cfg.adaptation_enabled and np.linalg.norm(error) > self.cfg.deadzone:
            features = self._features(state)
            scale = self.cfg.normalization + state @ state + features @ features
            self.state_gains += (
                self.cfg.dt
                * self.cfg.adaptation_gain
                * np.outer(training, state)
                / scale
            )
            self.theta += (
                self.cfg.dt
                * self.cfg.adaptation_gain
                * np.outer(features, training)
                / scale
            )
            for parameters in (self.state_gains, self.theta):
                norms = np.linalg.norm(parameters, axis=0)
                parameters /= np.maximum(1.0, norms / self.cfg.parameter_limit)
        self.reference_state = reference_next
        self.predictor_state = predictor_next

    def reset(self, state=None):
        """Reset adaptation and optionally synchronize reference and predictor."""
        initial = (
            np.zeros(self.n_state)
            if state is None
            else self._vector(state, "state").copy()
        )
        self.state_gains.fill(0.0)
        self.theta.fill(0.0)
        self.reference_state = initial.copy()
        self.predictor_state = initial
        self._last_error.fill(0.0)
        self._last_prediction_error.fill(0.0)
        self._last_adaptive.fill(0.0)

    @property
    def diagnostics(self):
        return {
            "reference_error": self._last_error.copy(),
            "prediction_error": self._last_prediction_error.copy(),
            "adaptive_command": self._last_adaptive.copy(),
            "parameter_norm": float(
                np.linalg.norm(self.state_gains) + np.linalg.norm(self.theta)
            ),
        }


__all__ = ["LQRPMRACAgent", "LQRPMRACConfig"]
