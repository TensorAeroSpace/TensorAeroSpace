"""Composite FDD detector: NominalKalman + CUSUM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .change_point import ChangePointDetector
from .kalman_3step import NominalKalman


@dataclass
class FDDConfig:
    """Hyper-parameters for :class:`FDDDetector`."""

    # Kalman.
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    alpha_Q: float = 0.99
    alpha_R: float = 0.99
    adapt_Q: bool = True
    adapt_R: bool = True
    # CUSUM.
    drift: float | None = None
    h_alarm: float = 20.0
    h_clear: float = 5.0
    cooldown_steps: int = 200
    # Innovation gating: skip Kalman update when innovation is too far out.
    innovation_sigma_gate: float = 5.0


@dataclass
class FDDOutput:
    """One-step output of :class:`FDDDetector`."""

    fault_present: bool
    severity: float
    confidence: float          # 1 − exp(−severity); ∈ [0, 1)
    innovation_norm: float
    time_since_event: float    # seconds since last rising edge


class FDDDetector:
    """One nominal Kalman + one CUSUM detector → FDDOutput."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        kalman: NominalKalman,
        cpd: ChangePointDetector,
        *,
        dt: float,
        innovation_sigma_gate: float = 5.0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.kalman = kalman
        self.cpd = cpd
        self.dt = float(dt)
        self.innovation_sigma_gate = float(innovation_sigma_gate)

    @classmethod
    def from_config(
        cls,
        n_state: int,
        n_control: int,
        *,
        dt: float,
        config: FDDConfig,
        F_nominal: np.ndarray,
        G_nominal: np.ndarray,
    ) -> "FDDDetector":
        """Build detector with default Kalman / CPD wired from FDDConfig."""
        Q = np.eye(n_state) * config.process_noise
        R = np.eye(n_state) * config.measurement_noise
        kf = NominalKalman(
            F_nominal=F_nominal, G_nominal=G_nominal, Q=Q, R=R,
            alpha_Q=config.alpha_Q, alpha_R=config.alpha_R,
            adapt_Q=config.adapt_Q, adapt_R=config.adapt_R,
        )
        cpd = ChangePointDetector(
            n_dim=n_state, drift=config.drift,
            h_alarm=config.h_alarm, h_clear=config.h_clear,
            cooldown_steps=config.cooldown_steps,
        )
        return cls(n_state=n_state, n_control=n_control,
                   kalman=kf, cpd=cpd, dt=dt,
                   innovation_sigma_gate=config.innovation_sigma_gate)

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        """Update Kalman F/G from a refreshed nominal estimate."""
        self.kalman.warm_start(F_nominal=F_nominal, G_nominal=G_nominal)

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> FDDOutput:
        """Run Kalman + CUSUM; return FDDOutput.

        The fault score is the Mahalanobis distance of the Kalman
        innovation under the predicted innovation covariance ``S``::

            d_t = νᵀ S⁻¹ ν.

        Under nominal dynamics this is approximately χ²-distributed with
        ``n_state`` degrees of freedom; under most faults the
        innovation either grows in magnitude or shifts in direction
        relative to ``S``, increasing ``d_t``.
        """
        kal = self.kalman.step(x_meas, u_prev)
        try:
            d_t = float(kal.nu @ np.linalg.solve(kal.S, kal.nu))
        except np.linalg.LinAlgError:
            d_t = float(kal.nu @ (np.linalg.pinv(kal.S) @ kal.nu))
        d_t = max(d_t, 0.0)

        cp = self.cpd.update(d_t)

        confidence = float(1.0 - np.exp(-cp.severity))
        time_since_event = float(cp.time_since_alarm) * self.dt
        return FDDOutput(
            fault_present=cp.alarm,
            severity=cp.severity,
            confidence=confidence,
            innovation_norm=float(np.linalg.norm(kal.nu)),
            time_since_event=time_since_event,
        )

    def reset(self) -> None:
        """Reset Kalman + CUSUM state. Configs unchanged."""
        self.kalman.reset()
        self.cpd.reset()
