"""UFTCController orchestrator: composes L2 inner + L3 middle + FDD detector.

Phase 1 MVP — see docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tensoraerospace.agent.aa_indi.model import AAINDIAgent, AAINDIConfig
from tensoraerospace.agent.base import BaseRLModel
from tensoraerospace.agent.iadp.model import IADPAgent, IADPConfig

from .fdd.detector import FDDConfig, FDDDetector, FDDOutput
from .inner import ModeSwitcher, SuperTwistingObserver, WrappedAAINDI
from .middle import IADPMiddle, RLSResetPolicy


@dataclass
class UFTCConfig:
    """Hyper-parameters for :class:`UFTCController`."""

    dt: float = 0.01
    fdd_update_every: int = 1
    fdd_warmup_steps: int = 200

    inner_cfg: AAINDIConfig = field(default_factory=AAINDIConfig)
    middle_cfg: IADPConfig = field(default_factory=IADPConfig)
    fdd_cfg: FDDConfig = field(default_factory=FDDConfig)
    rls_reset_policy: RLSResetPolicy = field(default_factory=RLSResetPolicy)

    sm_obs_k1: float = 3.0
    sm_obs_k2: float = 1.5
    trust_radius_nominal: float = 0.1
    trust_radius_fault: float = 0.5
    alpha_threshold_deg: float = 25.0
    alpha_hysteresis_deg: float = 5.0

    alpha_index: Optional[int] = None
    omega_indices: Optional[list] = None
    middle_lookahead_dt: float = 0.05

    enable_outer: bool = False  # placeholder for L4 D-SAC (Phase 3)


class UFTCController(BaseRLModel):
    """Top-level UFTC orchestrator (Phase 1 MVP)."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        *,
        nominal_F: Optional[np.ndarray] = None,
        nominal_G: Optional[np.ndarray] = None,
        config: Optional[UFTCConfig] = None,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else UFTCConfig()

        # Inner / middle dt sync to outer dt.
        inner_dict = dict(self.cfg.inner_cfg.__dict__)
        inner_dict["dt"] = self.cfg.dt
        self.cfg.inner_cfg = AAINDIConfig(**inner_dict)
        middle_dict = {k: v for k, v in self.cfg.middle_cfg.__dict__.items()
                       if k != "history"}
        middle_dict["dt"] = self.cfg.dt
        self.cfg.middle_cfg = IADPConfig(**middle_dict)

        # L2 inner.
        inner_base = AAINDIAgent(
            n_state=n_state, n_control=n_control, config=self.cfg.inner_cfg,
        )
        sm_obs = SuperTwistingObserver(
            n_axes=n_state, k1=self.cfg.sm_obs_k1,
            k2=self.cfg.sm_obs_k2, dt=self.cfg.dt,
        )
        mode_sw = ModeSwitcher(
            alpha_threshold_deg=self.cfg.alpha_threshold_deg,
            hysteresis_deg=self.cfg.alpha_hysteresis_deg,
        )
        self.inner = WrappedAAINDI(
            base=inner_base, sm_obs=sm_obs, mode_switch=mode_sw,
            trust_radius_nominal=self.cfg.trust_radius_nominal,
            trust_radius_fault=self.cfg.trust_radius_fault,
            dt=self.cfg.dt,
        )

        # L3 middle.
        middle_base = IADPAgent(
            n_state=n_state, n_control=n_control, config=self.cfg.middle_cfg,
        )
        self.middle = IADPMiddle(
            base=middle_base, reset_policy=self.cfg.rls_reset_policy,
            omega_indices=self.cfg.omega_indices,
            lookahead_dt=self.cfg.middle_lookahead_dt,
        )

        # FDD.
        F_init = (np.array(nominal_F, dtype=np.float64)
                  if nominal_F is not None
                  else np.zeros((n_state, n_state), dtype=np.float64))
        G_init = (np.array(nominal_G, dtype=np.float64)
                  if nominal_G is not None
                  else np.zeros((n_state, n_control), dtype=np.float64))
        self.fdd = FDDDetector.from_config(
            n_state=n_state, n_control=n_control, dt=self.cfg.dt,
            config=self.cfg.fdd_cfg, F_nominal=F_init, G_nominal=G_init,
        )
        # _fdd_ready: nominal matrices are loaded (can run fdd.step).
        # _fdd_active: warmup has elapsed AND matrices are ready.
        self._fdd_ready = nominal_F is not None and nominal_G is not None
        self._fdd_active = self._fdd_ready and self.cfg.fdd_warmup_steps == 0

        # Rolling state.
        self._step = 0
        self._last_fdd: FDDOutput = FDDOutput(
            fault_present=False, severity=0.0, confidence=0.0,
            innovation_norm=0.0, time_since_event=0.0,
        )
        self._last_u_indi = np.zeros(n_control, dtype=np.float64)
        self._last_omega_ref = np.zeros(n_state, dtype=np.float64)

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> np.ndarray:
        u_iadp, omega_ref = self.middle.predict(x_obs, reference, time_step)
        omega_meas = self._extract_omega(x_obs)
        alpha = self._extract_alpha(x_obs)
        u_indi = self.inner.predict(
            omega_ref=omega_ref,
            omega_meas=omega_meas,
            alpha=alpha,
            u_blend_target=u_iadp,
            fault_severity=self._last_fdd.severity,
            time_step=time_step,
        )
        self._last_u_indi = u_indi.copy()
        self._last_omega_ref = omega_ref.copy()
        return u_indi

    def learn(
        self,
        next_x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> dict:
        next_omega = self._extract_omega(next_x_obs)
        inner_diag = self.inner.learn(next_omega, self._last_omega_ref,
                                      time_step=time_step)

        # FDD warm-up trigger: once warmup_steps have elapsed, activate FDD.
        if not self._fdd_active and self._step + 1 >= self.cfg.fdd_warmup_steps:
            if not self._fdd_ready:
                # Nominal matrices not supplied at init — derive from IADP.
                F_warm = self.middle.base.F[:self.n_state, :self.n_state].copy()
                G_warm = self.middle.base.G[:self.n_state, :self.n_control].copy()
                self.fdd.warm_start(F_nominal=F_warm, G_nominal=G_warm)
                self._fdd_ready = True
            self._fdd_active = True

        if self._fdd_active and self._step % self.cfg.fdd_update_every == 0:
            self._last_fdd = self.fdd.step(next_x_obs, self._last_u_indi)

        middle_diag = self.middle.learn(
            next_x_obs, reference, time_step=time_step, fdd=self._last_fdd,
        )

        self._step += 1
        return {
            **{f"inner_{k}": v for k, v in inner_diag.items()},
            **{f"middle_{k}": v for k, v in middle_diag.items()},
            "fault_present": self._last_fdd.fault_present,
            "fdd_severity": self._last_fdd.severity,
            "fdd_confidence": self._last_fdd.confidence,
            "fdd_innovation_norm": self._last_fdd.innovation_norm,
        }

    def diagnostics(self) -> dict:
        """Snapshot of all sub-components for logging / plotting."""
        return {
            "step": int(self._step),
            "fault_present": bool(self._last_fdd.fault_present),
            "severity": float(self._last_fdd.severity),
            "confidence": float(self._last_fdd.confidence),
            "innovation_norm": float(self._last_fdd.innovation_norm),
            "rls_gamma": float(self.middle.base.rls.gamma_rls),
            "mode": str(self.inner.mode),
            "fdd_ready": bool(self._fdd_ready),
            "fdd_active": bool(self._fdd_active),
        }

    def reset(self) -> None:
        self.inner.reset()
        self.middle.reset()
        self.fdd.reset()
        self._step = 0
        self._fdd_active = self._fdd_ready and self.cfg.fdd_warmup_steps == 0
        self._last_fdd = FDDOutput(False, 0.0, 0.0, 0.0, 0.0)
        self._last_u_indi.fill(0.0)
        self._last_omega_ref.fill(0.0)

    def _extract_omega(self, x_obs: np.ndarray) -> np.ndarray:
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        if self.cfg.omega_indices is None:
            return x[: self.n_state]
        idx = np.asarray(self.cfg.omega_indices, dtype=np.int64)
        return x[idx]

    def _extract_alpha(self, x_obs: np.ndarray) -> float:
        if self.cfg.alpha_index is None:
            return 0.0
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        return float(x[int(self.cfg.alpha_index)])

    def get_param_env(self) -> dict[str, Any]:
        return {
            "policy": {
                "name": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "params": {
                    "n_state": self.n_state,
                    "n_control": self.n_control,
                },
            },
        }
