"""Discrete HOSM from Atmaca et al. (2025), Eqs. (47)--(50).

DOI: 10.2514/1.G009147. All four updates use the previous state and the
same zeroth-order error (the article's non-recursive differentiation form).
Gains depend on signal units and sampling; no universal gains are published.
"""

from __future__ import annotations

import numpy as np


class HOSMDifferentiator:
    """Four-state differentiator; return the estimated first derivative.

    ``gains`` is a positive length-four vector or a (4, n) matrix. The
    exponents below follow the cited article literally, including 2/3 in
    Eq. (48); this is not a substitution of a different Levant recursion.
    """

    def __init__(self, n: int, dt: float, gains: np.ndarray) -> None:
        self.n = int(n)
        self.dt = float(dt)
        gains = np.asarray(gains, dtype=float)
        if gains.shape == (4,):
            gains = np.repeat(gains[:, None], self.n, axis=1)
        if self.n <= 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("n and dt must be positive")
        if (
            gains.shape != (4, self.n)
            or not np.isfinite(gains).all()
            or np.any(gains <= 0)
        ):
            raise ValueError(
                "gains must be positive finite values of shape (4,) or (4, n)"
            )
        self.gains = gains.copy()
        self.z = np.zeros((4, self.n))
        self.initialized = False

    def reset(self) -> None:
        self.z.fill(0)
        self.initialized = False

    def step(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=float)
        if sample.shape != (self.n,) or not np.isfinite(sample).all():
            raise ValueError("sample must be a finite vector of length n")
        if not self.initialized:
            self.z[0] = sample
            self.initialized = True
            return np.asarray(self.z[1].copy())
        error = self.z[0] - sample
        sign = np.sign(error)
        powers = np.array([0.75, 2 / 3, 0.5, 0.0])[:, None]
        correction = self.gains * np.abs(error)[None, :] ** powers * sign
        derivative = -correction
        derivative[:3] += self.z[1:]
        following = self.z + self.dt * derivative
        if not np.isfinite(following).all():
            raise FloatingPointError(
                "HOSM diverged; check sample scaling, gains and dt"
            )
        self.z = following
        return np.asarray(self.z[1].copy())
