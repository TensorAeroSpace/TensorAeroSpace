"""Pseudo-Control Hedging block.

PCH stores the previous-tick virtual control demand ``ν_des_prev`` and
compares it to the current measured acceleration ``ω̇_meas`` to compute the
hedge signal ``ν_h = ν_des_prev − ω̇_meas`` — the gap the inner loop failed
to close, attributed to actuator dynamics or saturation. The reference
models subtract this hedge from their derivatives before integrating, which
prevents reference-rate wind-up during saturation. After ``freeze_after``
ticks of persistent gap on a given axis, that axis's reference rate is
hard-frozen until the gap closes.
"""

from __future__ import annotations

import numpy as np


class PseudoControlHedge:
    """PCH state machine, one entry per rate axis.

    Args:
        n_y: Number of rate axes.
        freeze_after: Number of consecutive saturated ticks before the
            corresponding reference rate is hard-frozen.
        gap_tol: Magnitude of ``|ν_h|`` below which the axis is considered
            tracked (resets the saturation counter).
    """

    def __init__(
        self,
        n_y: int,
        freeze_after: int = 20,
        gap_tol: float = 1e-6,
    ) -> None:
        if n_y <= 0:
            raise ValueError("n_y must be positive")
        if freeze_after <= 0:
            raise ValueError("freeze_after must be positive")
        if gap_tol < 0.0:
            raise ValueError("gap_tol must be ≥ 0")
        self.n_y = int(n_y)
        self.freeze_after = int(freeze_after)
        self.gap_tol = float(gap_tol)

        self.last_hedge = np.zeros(self.n_y, dtype=np.float64)
        self.saturation_counter = np.zeros(self.n_y, dtype=np.int32)
        self.is_frozen = np.zeros(self.n_y, dtype=bool)

    def reset(self) -> None:
        self.last_hedge = np.zeros(self.n_y, dtype=np.float64)
        self.saturation_counter = np.zeros(self.n_y, dtype=np.int32)
        self.is_frozen = np.zeros(self.n_y, dtype=bool)

    def update(
        self,
        nu_des_prev: np.ndarray,
        omega_dot_meas: np.ndarray,
    ) -> np.ndarray:
        """Compute hedge and update the freeze counters.

        Args:
            nu_des_prev: Virtual control demanded on the previous tick.
            omega_dot_meas: Measured angular acceleration this tick.

        Returns:
            Hedge vector ``ν_h`` of shape ``(n_y,)``.
        """
        nu = np.asarray(nu_des_prev, dtype=np.float64).reshape(-1)
        omd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        if nu.size != self.n_y or omd.size != self.n_y:
            raise ValueError(f"both inputs must have length {self.n_y}")
        hedge = nu - omd
        gap_active = np.abs(hedge) > self.gap_tol
        self.saturation_counter = np.where(
            gap_active,
            self.saturation_counter + 1,
            0,
        ).astype(np.int32)
        self.is_frozen = self.saturation_counter >= self.freeze_after
        self.last_hedge = hedge
        return hedge.copy()
