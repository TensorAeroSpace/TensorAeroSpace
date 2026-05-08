"""Composite FDD detector: NominalKalman + CUSUM + optional GLR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .change_point import ChangePointDetector
from .glr import GLRDetector
from .kalman_3step import NominalKalman


@dataclass
class FDDConfig:
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    alpha_Q: float = 0.99
    alpha_R: float = 0.99
    adapt_Q: bool = True
    adapt_R: bool = True
    drift: float | None = None
    h_alarm: float = 20.0
    h_clear: float = 5.0
    cooldown_steps: int = 200
    innovation_sigma_gate: float = 5.0


FaultKind = Literal["none", "abrupt", "gradual", "compound"]


@dataclass
class FDDOutput:
    """One-step output of :class:`FDDDetector`.

    Phase 1 fields: ``fault_present``, ``severity``, ``confidence``,
    ``innovation_norm``, ``time_since_event``. ``severity`` always equals
    ``max(severity_abrupt, severity_gradual)`` for compatibility with
    Phase 1 consumers.

    Phase 2 additions: ``fault_kind`` ∈ {"none","abrupt","gradual","compound"},
    ``severity_abrupt``, ``severity_gradual``, ``glr_drift_estimate``.
    """

    fault_present: bool
    severity: float
    confidence: float
    innovation_norm: float
    time_since_event: float
    fault_kind: FaultKind = "none"
    severity_abrupt: float = 0.0
    severity_gradual: float = 0.0
    glr_drift_estimate: Optional[np.ndarray] = None


class FDDDetector:
    """One nominal Kalman + CUSUM + optional GLR → FDDOutput."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        kalman: NominalKalman,
        cpd: ChangePointDetector,
        *,
        dt: float,
        glr: GLRDetector | None = None,
        innovation_sigma_gate: float = 5.0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.kalman = kalman
        self.cpd = cpd
        self.glr = glr
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
        glr: GLRDetector | None = None,
    ) -> "FDDDetector":
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
        return cls(
            n_state=n_state, n_control=n_control,
            kalman=kf, cpd=cpd, dt=dt, glr=glr,
            innovation_sigma_gate=config.innovation_sigma_gate,
        )

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        self.kalman.warm_start(F_nominal=F_nominal, G_nominal=G_nominal)

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> FDDOutput:
        kal = self.kalman.step(x_meas, u_prev)
        try:
            d_t = float(kal.nu @ np.linalg.solve(kal.S, kal.nu))
        except np.linalg.LinAlgError:
            d_t = float(kal.nu @ (np.linalg.pinv(kal.S) @ kal.nu))
        d_t = max(d_t, 0.0)

        cp = self.cpd.update(d_t)
        gl = self.glr.update(kal.nu, kal.S) if self.glr is not None else None

        abrupt = bool(cp.alarm)
        gradual = bool(gl.alarm) if gl is not None else False
        kind: FaultKind = (
            "compound" if abrupt and gradual
            else "abrupt" if abrupt
            else "gradual" if gradual
            else "none"
        )
        sev_a = float(cp.severity)
        sev_g = float(gl.severity) if gl is not None else 0.0
        severity = max(sev_a, sev_g)
        confidence = float(1.0 - np.exp(-(sev_a + sev_g)))
        return FDDOutput(
            fault_present=(abrupt or gradual),
            severity=severity,
            confidence=confidence,
            innovation_norm=float(np.linalg.norm(kal.nu)),
            time_since_event=float(cp.time_since_alarm) * self.dt,
            fault_kind=kind,
            severity_abrupt=sev_a,
            severity_gradual=sev_g,
            glr_drift_estimate=(gl.drift_estimate if gl is not None and gl.alarm
                                else None),
        )

    def reset(self) -> None:
        self.kalman.reset()
        self.cpd.reset()
        if self.glr is not None:
            self.glr.reset()
