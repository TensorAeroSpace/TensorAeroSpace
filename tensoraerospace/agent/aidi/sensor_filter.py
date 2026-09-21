"""Low-pass measurement differentiator used by AIDI."""

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
        """Return a copy of the latest filtered derivative without advancing the filter."""
        return self._y.copy()
