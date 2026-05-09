"""L3 middle for UFTC: IADPAgent + innovation-driven RLS reset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tensoraerospace.agent.iadp.model import IADPAgent

from .fdd.detector import FDDOutput


@dataclass
class RLSResetPolicy:
    """How L3 reacts to an FDD rising edge."""

    cov_inflation: float = 100.0  # Φ ← Φ + κ·I
    forgetting_drop: float = 0.9  # γ_RLS ← drop on rising edge
    forgetting_recover_steps: int = 500  # linear ramp back to nominal


class IADPMiddle:
    """Wraps IADPAgent with FDD-triggered RLS reset and rate-target derivation."""

    def __init__(
        self,
        base: IADPAgent,
        reset_policy: RLSResetPolicy,
        *,
        omega_indices: Optional[list[int]] = None,
        lookahead_dt: float = 0.05,
    ) -> None:
        self.base = base
        self.reset_policy = reset_policy
        self.omega_indices = list(omega_indices) if omega_indices else None
        self.lookahead_dt = float(lookahead_dt)
        self._gamma_nominal = float(base.rls.gamma_rls)
        self._gamma_active = self._gamma_nominal
        self._recover_countdown = 0
        self._prev_fault = False
        # Phase 4 — monitor V_iADP extractor reads this.
        self._last_state_error: Optional[np.ndarray] = None

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_iadp = self.base.predict(x_obs, reference, time_step=time_step)
        omega_ref = self._derive_omega_ref(x_obs, reference, time_step)
        # Phase 4 — cache plant-state error e = ref - x for monitor V_iADP.
        x_v = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        n = self.base.n_state if hasattr(self.base, "n_state") else x_v.size
        x_n = x_v[:n]
        ref_v = np.asarray(reference, dtype=np.float64)
        if ref_v.ndim == 2:
            idx = int(np.clip(time_step, 0, ref_v.shape[1] - 1))
            r_vec = ref_v[:, idx]
        elif ref_v.ndim == 1 and ref_v.size != x_n.size:
            idx = int(np.clip(time_step, 0, ref_v.size - 1))
            r_vec = np.full(x_n.size, ref_v[idx], dtype=np.float64)
        else:
            r_vec = ref_v.astype(np.float64).reshape(-1)
            if r_vec.size != x_n.size:
                r_vec = np.broadcast_to(r_vec, x_n.shape).copy()
        self._last_state_error = (r_vec - x_n).copy()
        return u_iadp, omega_ref

    def learn(
        self,
        next_x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
        *,
        fdd: FDDOutput,
    ) -> dict:
        # Rising-edge detection.
        rising = bool(fdd.fault_present and not self._prev_fault)
        if rising:
            self._trigger_reset()
        self._prev_fault = bool(fdd.fault_present)

        # Linear γ_RLS recovery (skip the step where we just triggered reset).
        if self._recover_countdown > 0 and not rising:
            self._recover_countdown -= 1
            frac = 1.0 - (
                self._recover_countdown
                / max(1, self.reset_policy.forgetting_recover_steps)
            )
            self._gamma_active = (
                self.reset_policy.forgetting_drop
                + (self._gamma_nominal - self.reset_policy.forgetting_drop) * frac
            )
            self.base.rls.gamma_rls = float(self._gamma_active)

        return self.base.learn(next_x_obs, reference, time_step=time_step)

    def reset(self) -> None:
        """Restore base agent + recovery state."""
        self.base.reset()
        self.base.rls.gamma_rls = self._gamma_nominal
        self._gamma_active = self._gamma_nominal
        self._recover_countdown = 0
        self._prev_fault = False
        self._last_state_error = None

    def _trigger_reset(self) -> None:
        n = self.base.rls.n_regressor
        self.base.rls.Phi = (
            self.base.rls.Phi
            + self.reset_policy.cov_inflation * np.eye(n, dtype=np.float64)
        )
        self.base.rls.gamma_rls = float(self.reset_policy.forgetting_drop)
        self._gamma_active = float(self.reset_policy.forgetting_drop)
        self._recover_countdown = int(self.reset_policy.forgetting_recover_steps)

    def force_reset(self, severity_hint: float = 1.0) -> None:
        """Inflate RLS covariance and drop forgetting factor independent of FDD.

        Used as a Phase 4 macro-action sink. Severity scales the inflation
        multiplier; ``1.0`` matches the standard FDD-triggered reset.
        """
        sev = float(max(0.1, min(severity_hint, 5.0)))
        inflate = float(self.reset_policy.cov_inflation) * sev
        n = int(self.base.rls.Phi.shape[0])
        self.base.rls.Phi = self.base.rls.Phi + inflate * np.eye(n, dtype=np.float64)
        self.base.rls.gamma_rls = float(self.reset_policy.forgetting_drop)
        self._gamma_active = float(self.reset_policy.forgetting_drop)
        self._recover_countdown = int(self.reset_policy.forgetting_recover_steps)

    def _derive_omega_ref(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
    ) -> np.ndarray:
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 2:
            idx = int(np.clip(time_step, 0, ref.shape[1] - 1))
            ref_vec = ref[:, idx]
        elif ref.ndim == 1 and ref.size != x.size:
            idx = int(np.clip(time_step, 0, ref.size - 1))
            ref_vec = np.full(x.size, ref[idx], dtype=np.float64)
        else:
            ref_vec = ref.astype(np.float64).reshape(-1)
            if ref_vec.size != x.size:
                ref_vec = np.broadcast_to(ref_vec, x.shape).copy()

        if self.omega_indices is None:
            return ref_vec.copy()

        idx_arr = np.asarray(self.omega_indices, dtype=np.int64)
        omega_now = x[idx_arr]
        omega_target = ref_vec[idx_arr]
        return (omega_target - omega_now) / max(self.lookahead_dt, 1e-9)
