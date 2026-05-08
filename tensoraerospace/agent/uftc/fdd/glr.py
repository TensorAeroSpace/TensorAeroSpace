"""Generalised likelihood ratio (GLR) test on Kalman innovations.

Two-sided GLR over a sliding window for slow drifts in the innovation
mean. Under the nominal hypothesis ``ν_t ~ N(0, S_t)``. For an unknown
drift ``μ ≠ 0`` starting at unknown change-time ``τ``,

    T_t = max_{t-W ≤ τ ≤ t-1}  ‖ Σ_{i=τ}^t S_i^{-1} ν_i ‖²_{(Σ_{i=τ}^t S_i^{-1})^{-1}}

This implementation keeps an O(W) window of ``S^{-1} ν`` and ``S^{-1}``
per step; the sup is computed by sweeping τ at update-time. Hysteresis
between ``h_alarm`` and ``h_clear`` plus a ``cooldown_steps`` window
prevent chattering.

References:
    Basseville & Nikiforov (1993) Detection of Abrupt Changes, ch. 7.
    Willsky (1976) Survey of failure detection methods.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class GLRConfig:
    window: int = 200
    h_alarm: float = 30.0
    h_clear: float = 8.0
    cooldown_steps: int = 200
    mu_min_norm: float = 0.05  # discard drift estimate below this magnitude


@dataclass
class GLRState:
    statistic: float
    alarm: bool
    severity: float            # statistic / h_alarm, clipped to [0, 10]
    drift_estimate: np.ndarray
    time_since_alarm: int


class GLRDetector:
    """Sliding-window GLR detector on Kalman innovations."""

    def __init__(self, n_dim: int, cfg: GLRConfig) -> None:
        if cfg.h_clear >= cfg.h_alarm:
            raise ValueError("h_clear must be strictly below h_alarm")
        if cfg.window < 2:
            raise ValueError("window must be ≥ 2")
        self.n_dim = int(n_dim)
        self.cfg = cfg
        self._buf_Sinv_nu: deque[np.ndarray] = deque(maxlen=cfg.window)
        self._buf_Sinv: deque[np.ndarray] = deque(maxlen=cfg.window)
        self._in_alarm = False
        self._steps_in_alarm = 0
        self._steps_since_alarm = 10**9

    def update(self, nu: np.ndarray, S: np.ndarray) -> GLRState:
        nu = np.asarray(nu, dtype=np.float64).reshape(-1)
        S = np.asarray(S, dtype=np.float64)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        Sinv_nu = Sinv @ nu

        self._buf_Sinv_nu.append(Sinv_nu)
        self._buf_Sinv.append(Sinv)

        # Sweep tau in [t-W, t-1] to find max statistic.
        cum_Sinv_nu = np.zeros(self.n_dim)
        cum_Sinv = np.zeros((self.n_dim, self.n_dim))
        T_max = 0.0
        mu_hat = np.zeros(self.n_dim)
        for k in range(len(self._buf_Sinv) - 1, -1, -1):
            cum_Sinv_nu += self._buf_Sinv_nu[k]
            cum_Sinv += self._buf_Sinv[k]
            try:
                solve = np.linalg.solve(cum_Sinv, cum_Sinv_nu)
            except np.linalg.LinAlgError:
                solve = np.linalg.pinv(cum_Sinv) @ cum_Sinv_nu
            T = float(cum_Sinv_nu @ solve)
            if T > T_max:
                T_max = T
                mu_hat = solve

        # Hysteresis & cooldown.
        if (not self._in_alarm) and T_max > self.cfg.h_alarm:
            self._in_alarm = True
            self._steps_since_alarm = 0
            self._steps_in_alarm = 1
        elif self._in_alarm:
            self._steps_in_alarm += 1
            self._steps_since_alarm += 1
            if (T_max < self.cfg.h_clear
                    and self._steps_since_alarm > self.cfg.cooldown_steps):
                self._in_alarm = False
        else:
            self._steps_since_alarm = min(self._steps_since_alarm + 1, 10**9)

        if float(np.linalg.norm(mu_hat)) < self.cfg.mu_min_norm:
            mu_hat = np.zeros(self.n_dim)

        severity = float(min(T_max / self.cfg.h_alarm, 10.0))
        return GLRState(
            statistic=float(T_max),
            alarm=bool(self._in_alarm),
            severity=severity,
            drift_estimate=mu_hat,
            time_since_alarm=int(self._steps_since_alarm),
        )

    def reset(self) -> None:
        self._buf_Sinv_nu.clear()
        self._buf_Sinv.clear()
        self._in_alarm = False
        self._steps_in_alarm = 0
        self._steps_since_alarm = 10**9
