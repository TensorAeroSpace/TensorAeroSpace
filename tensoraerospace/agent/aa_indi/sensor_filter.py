"""Lightweight sensor filter approximating the OTSEKF-HOSM stack.

The reference AA-INDI paper combines an Optimal Two-Stage Extended Kalman
Filter with a Higher-Order Sliding-Mode (HOSM) differentiator to estimate
angular-rate derivatives and detect IMU biases. A fully fledged OTSEKF-HOSM
is a sizeable project and hardware-specific to boot; for the initial library
release we provide a minimal two-block stand-in that already makes INDI
practical:

1. :class:`LowPassDerivative` — second-order Butterworth-style low-pass
   differentiator that converts noisy ω readings into ω̇ with bounded
   high-frequency noise. Acts as the HOSM surrogate.
2. :class:`BiasEstimator` — a scalar exponential-forgetting mean of the
   residual between the raw and differentiated-then-reintegrated signals.
   Provides a coarse IMU bias estimate that the AA-INDI agent can subtract
   from the measured state, emulating the OTSEKF fault-state branch.

The real HOSM / two-stage EKF can be dropped in later as a child class —
these primitives keep the agent's I/O contract small and unit-testable.
"""

from __future__ import annotations

import numpy as np


class LowPassDerivative:
    """Causal finite-difference differentiator with a low-pass filter.

    Computes ω̇_t from a sequence of ω_t readings using the first-order
    backward difference ``(ω_t − ω_{t-1}) / dt`` followed by an
    exponential filter with cut-off set by ``cutoff_hz``. The filter is
    a discrete first-order IIR with α = dt · 2π · cutoff.

    Args:
        n: Dimension of the input signal.
        dt: Sampling period [s].
        cutoff_hz: Low-pass cut-off frequency [Hz]. Values in 5–20 Hz
            are typical for sub-sonic flight envelopes.
    """

    def __init__(self, n: int, dt: float, cutoff_hz: float = 10.0) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive")
        self.n = int(n)
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)

        alpha = dt * 2.0 * np.pi * cutoff_hz
        # Clamp to [0, 1] so the IIR is always stable even for silly inputs.
        self._alpha = float(np.clip(alpha, 0.0, 1.0))

        self._prev_x: np.ndarray | None = None
        self._y: np.ndarray = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        """Clear the internal filter state."""
        self._prev_x = None
        self._y = np.zeros(self.n, dtype=np.float64)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Ingest a new sample and return the filtered derivative estimate."""
        x_v = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_v.size != self.n:
            raise ValueError(f"x must have length {self.n}, got {x_v.size}")
        if self._prev_x is None:
            self._prev_x = x_v.copy()
            return self._y.copy()
        raw_deriv = (x_v - self._prev_x) / self.dt
        self._y = self._y + self._alpha * (raw_deriv - self._y)
        self._prev_x = x_v.copy()
        return self._y.copy()

    @property
    def last_output(self) -> np.ndarray:
        return self._y.copy()


class BiasEstimator:
    """Exponential-forgetting mean of an innovation signal.

    Used by :mod:`aa_indi` to produce a scalar IMU bias estimate ``b̂``
    from the residual between the raw measurement and the
    reintegrated-from-derivative one. When an actual bias appears, the
    residual has a non-zero mean and ``b̂`` tracks it with a time
    constant of roughly ``dt / (1 − lambda)``.

    Args:
        n: Dimension of the innovation.
        forgetting: Exponential-moving-average retention (``0 < λ < 1``).
            Values near 1 average over long windows (slow bias tracking);
            smaller values react faster at the cost of noisier estimates.
    """

    def __init__(self, n: int, forgetting: float = 0.99) -> None:
        if not 0.0 < forgetting < 1.0:
            raise ValueError("forgetting must lie in (0, 1)")
        self.n = int(n)
        self.lam = float(forgetting)
        self._bias = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        self._bias = np.zeros(self.n, dtype=np.float64)

    def update(self, innovation: np.ndarray) -> np.ndarray:
        """Update the bias estimate with a new innovation sample."""
        inn = np.asarray(innovation, dtype=np.float64).reshape(-1)
        if inn.size != self.n:
            raise ValueError(f"innovation must have length {self.n}, got {inn.size}")
        self._bias = self.lam * self._bias + (1.0 - self.lam) * inn
        return self._bias.copy()

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()
