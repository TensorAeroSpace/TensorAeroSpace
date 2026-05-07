"""Inner-loop (L2) extensions for UFTC: SM observer, mode switch,
trust-region wrapper around aa_indi.AAINDIAgent.

Phase 1 MVP — see docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


class SuperTwistingObserver:
    """Higher-order sliding-mode observer (super-twisting algorithm).

    Estimates the unmodeled high-frequency disturbance δ̂ on each angular
    axis from the residual ``s = ω̇_meas − ν_des − δ̂``::

        ṡ = −k₁·|s|^{1/2}·sign(s) + z
        ż = −k₂·sign(s)

    Args:
        n_axes: Number of angular axes (typically 3 for an aircraft).
        k1: Outer super-twisting gain.
        k2: Inner super-twisting gain.
        dt: Sampling period [s].
    """

    def __init__(
        self,
        n_axes: int,
        *,
        k1: float = 3.0,
        k2: float = 1.5,
        dt: float = 0.01,
    ) -> None:
        self.n_axes = int(n_axes)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        """Clear observer state."""
        self._s = np.zeros(self.n_axes, dtype=np.float64)
        self._z = np.zeros(self.n_axes, dtype=np.float64)

    def update(
        self,
        omega_dot_meas: np.ndarray,
        nu_des: np.ndarray,
    ) -> np.ndarray:
        """Run one observer step. Returns δ̂ ≈ (ω̇_meas − ν_des)."""
        wd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        nd = np.asarray(nu_des, dtype=np.float64).reshape(-1)
        if wd.size != self.n_axes:
            raise ValueError(
                f"omega_dot_meas must have length {self.n_axes}, got {wd.size}"
            )
        if nd.size != self.n_axes:
            raise ValueError(
                f"nu_des must have length {self.n_axes}, got {nd.size}"
            )

        # Discrete-time Euler integration of the super-twisting law.
        # The sliding variable σ = s − e drives s toward e = ω̇_meas − ν_des.
        # At convergence σ → 0, so s → e and s IS the disturbance estimate.
        e = wd - nd
        sigma = self._s - e  # sliding variable
        sgn = np.sign(sigma)
        abs_term = np.sqrt(np.abs(sigma))
        ds = -self.k1 * abs_term * sgn + self._z
        dz = -self.k2 * sgn

        self._s = self._s + self.dt * ds
        self._z = self._z + self.dt * dz

        # δ̂ = s; the observer state converges to e in finite time.
        return self._s.copy()


class ModeSwitcher:
    """Hysteretic rate-INDI ↔ angle-INDI mode selector.

    Args:
        alpha_threshold_deg: AoA above which the mode switches to angle-INDI.
        hysteresis_deg: AoA must drop below ``alpha_threshold_deg − hysteresis_deg``
            to switch back to rate-INDI.
    """

    def __init__(
        self,
        alpha_threshold_deg: float = 25.0,
        hysteresis_deg: float = 5.0,
    ) -> None:
        if hysteresis_deg < 0.0:
            raise ValueError("hysteresis_deg must be non-negative")
        self.alpha_threshold = float(np.deg2rad(alpha_threshold_deg))
        self.alpha_clear = float(np.deg2rad(alpha_threshold_deg - hysteresis_deg))
        self.reset()

    def reset(self) -> None:
        """Return to the default rate-INDI mode."""
        self._mode: Literal["rate", "angle"] = "rate"

    def select(self, alpha_rad: float) -> Literal["rate", "angle"]:
        """Update mode given current α and return new mode label."""
        a = float(alpha_rad)
        if self._mode == "rate" and a > self.alpha_threshold:
            self._mode = "angle"
        elif self._mode == "angle" and a < self.alpha_clear:
            self._mode = "rate"
        return self._mode

    @property
    def mode(self) -> Literal["rate", "angle"]:
        return self._mode
