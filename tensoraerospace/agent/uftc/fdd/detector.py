"""Composite FDD detector: NominalKalman + CUSUM.

Score formulation
-----------------
A pure Mahalanobis score  ``d_t = nu^T S^{-1} nu``  has expected value
``~0.94`` under H₀ (well below the CUSUM drift of 3.0) and only ``~1.77``
under a partial-control-authority fault (G → 0.05 G).  Both values are
below the drift, so the CUSUM cannot distinguish the two reliably.

The reason is that reducing G *lowers* the nominal control contribution to
the innovation:  ``nu ≈ (G_fault – G) u + noise``.  Because u is zero-mean,
the expected innovation stays zero, but the per-step variance *decreases*
under the fault — a Mahalanobis-only CUSUM cannot detect a variance drop.

To recover sensitivity the detector augments the Mahalanobis score with a
**control-projection term**

    ctrl_proj = –(nu^T u_prev) / ‖u_prev‖

Under H₀ this term has zero mean (innovations are uncorrelated with
control).  Under a partial-authority fault (G_fault < G_nominal) the
innovation satisfies  ``E[nu | u] = (G_fault – G) u``, so

    E[ctrl_proj] = ‖(G – G_fault) u‖ · E[‖u‖] / ‖u‖ > 0

Multiplying by ``ctrl_projection_alpha`` shifts the fault distribution
above the CUSUM drift while keeping the nominal distribution well below it.
The default ``ctrl_projection_alpha = 20.0`` was chosen so that
``E[score_fault] ≈ drift``, derived from typical fault magnitudes:

    alpha ≈ (drift − E[maha_fault]) / E[ctrl_proj_fault]
          ≈ (3.0 − 1.77) / 0.060 ≈ 20
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    # Control-projection augmentation scale (see module docstring).
    ctrl_projection_alpha: float = 20.0


@dataclass
class FDDOutput:
    """One-step output of :class:`FDDDetector`."""

    fault_present: bool
    severity: float
    confidence: float          # 1 − exp(−severity); ∈ [0, 1)
    innovation_norm: float
    time_since_event: float    # seconds since last rising edge


class FDDDetector:
    """One nominal Kalman + one CUSUM detector → FDDOutput.

    The CUSUM is fed a hybrid score that combines the standard Mahalanobis
    distance with a control-projection term; see module docstring for the
    rationale.
    """

    def __init__(
        self,
        n_state: int,
        n_control: int,
        kalman: NominalKalman,
        cpd: ChangePointDetector,
        *,
        dt: float,
        innovation_sigma_gate: float = 5.0,
        ctrl_projection_alpha: float = 20.0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.kalman = kalman
        self.cpd = cpd
        self.dt = float(dt)
        self.innovation_sigma_gate = float(innovation_sigma_gate)
        self.ctrl_projection_alpha = float(ctrl_projection_alpha)

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
                   innovation_sigma_gate=config.innovation_sigma_gate,
                   ctrl_projection_alpha=config.ctrl_projection_alpha)

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        """Update Kalman F/G from a refreshed nominal estimate."""
        self.kalman.warm_start(F_nominal=F_nominal, G_nominal=G_nominal)

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> FDDOutput:
        """Run Kalman + CUSUM; return FDDOutput.

        Score = max(maha + alpha * ctrl_proj, 0), where

        * ``maha``      = ``nu^T S^{-1} nu``  (Mahalanobis innovation distance),
        * ``ctrl_proj`` = ``–(nu^T u_prev) / ‖u_prev‖``  (control-projection
          term; positive when innovation is anti-correlated with control,
          which is the signature of partial control-authority loss).
        """
        kal = self.kalman.step(x_meas, u_prev)

        # Mahalanobis distance (regularised solve).
        try:
            maha = float(kal.nu @ np.linalg.solve(kal.S, kal.nu))
        except np.linalg.LinAlgError:
            maha = float(kal.nu @ (np.linalg.pinv(kal.S) @ kal.nu))
        maha = max(maha, 0.0)

        # Control-projection augmentation.
        u = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-8:
            ctrl_proj = float(-kal.nu @ u) / u_norm
        else:
            ctrl_proj = 0.0

        d_t = max(maha + self.ctrl_projection_alpha * ctrl_proj, 0.0)

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
