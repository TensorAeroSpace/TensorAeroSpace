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


def _zero_fdd_output(n_state: int) -> FDDOutput:
    """Return a benign all-zero :class:`FDDOutput` (Phase-2 schema)."""
    return FDDOutput(
        fault_present=False,
        severity=0.0,
        confidence=0.0,
        innovation_norm=0.0,
        time_since_event=0.0,
        fault_kind="none",
        severity_abrupt=0.0,
        severity_gradual=0.0,
    )


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

    # Phase 2 — L1 HJ-reachability shield
    enable_l1_shield: bool = False
    l1_h_clear: float = 0.20
    l1_cbf_lambda: float = 1.0
    l1_u_min: Optional[list] = None
    l1_u_max: Optional[list] = None
    l1_value_fn_path: Optional[str] = None
    l1_conformal_eps_0: float = 0.05

    # Phase 2 — GLR slow-drift detector (composed into FDDDetector)
    enable_glr: bool = False
    glr_window: int = 200
    glr_h_alarm: float = 30.0
    glr_h_clear: float = 8.0
    glr_cooldown_steps: int = 200

    # Phase 3 — L4 D-SAC outer
    enable_l4_outer: bool = False
    l4_n_ref_dim: int = 0          # if 0 and enable_l4_outer, defaults to n_state
    l4_action_scale: float = 0.1
    l4_actor_hidden: tuple = (64, 64)
    l4_critic_hidden: tuple = (64, 64)
    l4_n_quantiles: int = 16
    l4_cvar_alpha: float = 0.2
    l4_glr_reset_threshold: float = 0.10
    l4_eval_mode: bool = True
    l4_replay_capacity: int = 10_000
    l4_batch_size: int = 64
    l4_seed: int = 0
    l4_trim_free: Optional[dict] = None    # {V_idx, gamma_idx, alpha_idx, q_idx}

    # Phase 4 — composite Lyapunov monitor
    enable_monitor: bool = False
    monitor_c_weights: tuple = (0.2, 0.2, 0.2, 0.2, 0.2)
    monitor_a_diag: tuple = (0.5, 0.5, 0.5, 0.5, 0.5)
    monitor_eps_matrix: tuple = (
        (0.0, 0.05, 0.05, 0.05, 0.05),
        (0.05, 0.0, 0.05, 0.05, 0.05),
        (0.05, 0.05, 0.0, 0.05, 0.05),
        (0.05, 0.05, 0.05, 0.0, 0.05),
        (0.05, 0.05, 0.05, 0.05, 0.0),
    )
    monitor_d_disturbance: tuple = (0.05, 0.05, 0.05, 0.05, 0.05)
    monitor_alarm_warn_frac: float = 0.7
    monitor_alarm_critical_frac: float = 0.95
    monitor_cooldown_steps: int = 200


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
        self._omega_indices = (
            np.asarray(self.cfg.omega_indices, dtype=np.int64)
            if self.cfg.omega_indices is not None
            else None
        )
        if self._omega_indices is not None:
            if self._omega_indices.ndim != 1 or self._omega_indices.size == 0:
                raise ValueError("omega_indices must be a non-empty 1-D list")
            if np.any(self._omega_indices < 0) or np.any(
                self._omega_indices >= self.n_state
            ):
                raise ValueError(
                    f"omega_indices={self.cfg.omega_indices!r} are out of "
                    f"bounds for n_state={self.n_state}"
                )
        self.n_inner_state = (
            int(self._omega_indices.size)
            if self._omega_indices is not None
            else self.n_state
        )

        # Inner / middle dt sync to outer dt.
        inner_dict = dict(self.cfg.inner_cfg.__dict__)
        inner_dict["dt"] = self.cfg.dt
        inner_g0 = inner_dict.get("G_init")
        if inner_g0 is not None and self._omega_indices is not None:
            inner_g0_arr = np.asarray(inner_g0, dtype=np.float64)
            if inner_g0_arr.shape == (self.n_state, self.n_control):
                inner_dict["G_init"] = inner_g0_arr[self._omega_indices].copy()
        self.cfg.inner_cfg = AAINDIConfig(**inner_dict)
        middle_dict = {k: v for k, v in self.cfg.middle_cfg.__dict__.items()
                       if k != "history"}
        middle_dict["dt"] = self.cfg.dt
        self.cfg.middle_cfg = IADPConfig(**middle_dict)

        # L2 inner.
        inner_base = AAINDIAgent(
            n_state=self.n_inner_state,
            n_control=n_control,
            config=self.cfg.inner_cfg,
        )
        sm_obs = SuperTwistingObserver(
            n_axes=self.n_inner_state, k1=self.cfg.sm_obs_k1,
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

        # Phase 2 — GLR slow-drift detector wired into the existing FDD.
        if self.cfg.enable_glr:
            from .fdd.glr import GLRConfig, GLRDetector
            self.fdd.glr = GLRDetector(
                n_dim=self.n_state,
                cfg=GLRConfig(
                    window=self.cfg.glr_window,
                    h_alarm=self.cfg.glr_h_alarm,
                    h_clear=self.cfg.glr_h_clear,
                    cooldown_steps=self.cfg.glr_cooldown_steps,
                ),
            )

        # Phase 2 — HJ-reachability safety shield.
        self.l1: Optional[Any] = None
        if self.cfg.enable_l1_shield:
            from .l1 import (
                ConformalMargin,
                ConformalMarginConfig,
                DeepReachValueFn,
                HJReachabilityShield,
                HJShieldConfig,
            )
            from .l1.shield import _Identity
            if self.cfg.l1_value_fn_path is None:
                # No saved network — use the placeholder constant V_θ that
                # always reports "deep inside safe set". The shield then
                # short-circuits to nominal control on every call.
                value_fn = _Identity()
            else:
                value_fn = DeepReachValueFn.load(self.cfg.l1_value_fn_path)
            cm = ConformalMargin(
                ConformalMarginConfig(eps_0=self.cfg.l1_conformal_eps_0),
                lipschitz_const=value_fn.lipschitz_const(),
            )
            self.l1 = HJReachabilityShield(
                n_state=self.n_state, n_control=self.n_control,
                value_fn=value_fn,
                dynamics_fn=None,                # uses RLS-borrow path
                cfg=HJShieldConfig(
                    h_clear=self.cfg.l1_h_clear,
                    cbf_lambda=self.cfg.l1_cbf_lambda,
                    u_min=(np.asarray(self.cfg.l1_u_min, dtype=np.float64)
                           if self.cfg.l1_u_min is not None else None),
                    u_max=(np.asarray(self.cfg.l1_u_max, dtype=np.float64)
                           if self.cfg.l1_u_max is not None else None),
                ),
                conformal_margin=cm,
            )

        # Phase 3 — L4 D-SAC outer-loop reference planner.
        self.l4: Optional[Any] = None
        self.l4_trim_free: Optional[Any] = None
        self._last_r_eff: Optional[np.ndarray] = None
        self._last_beta: float = 0.0
        self._last_reset_hint: bool = False
        if self.cfg.enable_l4_outer:
            from tensoraerospace.agent.uftc.l4 import (
                DSACConfig,
                DSACOuter,
                LongitudinalTrimFreeConfig,
                LongitudinalTrimFreeWrapper,
            )
            n_ref = self.cfg.l4_n_ref_dim or self.n_state
            # ``n_action`` matches the actor's output (= reference-perturbation
            # space). The replay's ``a_actual`` is ``u_safe`` re-projected into
            # this space (right-pad with zeros / left-truncate to ``n_action``)
            # so the critic input ``(s, a_actual)`` has consistent shape.
            dsac_cfg = DSACConfig(
                n_state=self.n_state, n_ref_dim=n_ref, n_action=n_ref,
                cvar_alpha=self.cfg.l4_cvar_alpha,
                n_quantiles=self.cfg.l4_n_quantiles,
                actor_hidden=tuple(self.cfg.l4_actor_hidden),
                critic_hidden=tuple(self.cfg.l4_critic_hidden),
                glr_reset_threshold=self.cfg.l4_glr_reset_threshold,
                eval_mode=self.cfg.l4_eval_mode,
                action_scale=self.cfg.l4_action_scale,
                replay_capacity=self.cfg.l4_replay_capacity,
                batch_size=self.cfg.l4_batch_size,
                seed=self.cfg.l4_seed,
            )
            self.l4 = DSACOuter(dsac_cfg)
            if self.cfg.l4_trim_free:
                tf_cfg = LongitudinalTrimFreeConfig(
                    enabled=True, **self.cfg.l4_trim_free)
                self.l4_trim_free = LongitudinalTrimFreeWrapper(tf_cfg)

        # Rolling state.
        self._step = 0
        self._last_fdd: FDDOutput = _zero_fdd_output(self.n_state)
        self._last_u_indi = np.zeros(n_control, dtype=np.float64)
        self._last_u_safe: Optional[np.ndarray] = None
        self._last_omega_ref = np.zeros(self.n_inner_state, dtype=np.float64)

        # Phase 4 — composite Lyapunov monitor.
        self.monitor = None
        self.dispatcher = None
        self._monitor_out = None
        self._monitor_alarm = "OK"
        self._last_dispatch_diag: dict = {}
        if self.cfg.enable_monitor:
            from tensoraerospace.agent.uftc.monitor import (
                CompositeLyapunovMonitor,
                MacroActionDispatcher,
                MonitorConfig,
                MonitorOutput,
            )
            mcfg = MonitorConfig(
                c_weights=tuple(self.cfg.monitor_c_weights),
                a_diag=tuple(self.cfg.monitor_a_diag),
                eps_matrix=tuple(tuple(row) for row in self.cfg.monitor_eps_matrix),
                d_disturbance=tuple(self.cfg.monitor_d_disturbance),
                alarm_warn_frac=float(self.cfg.monitor_alarm_warn_frac),
                alarm_critical_frac=float(self.cfg.monitor_alarm_critical_frac),
                cooldown_steps=int(self.cfg.monitor_cooldown_steps),
            )
            self.monitor = CompositeLyapunovMonitor(mcfg)
            self.dispatcher = MacroActionDispatcher(
                l3=self.middle, l4=self.l4, l1=self.l1,
            )
            self._monitor_out = MonitorOutput.zero()

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> np.ndarray:
        # Phase 3 — L4 outer planner perturbs/replaces the reference fed to L3.
        if self.l4 is not None:
            fdd_for_l4 = (self._last_fdd if self._last_fdd is not None
                          else _zero_fdd_output(self.n_state))
            r_eff, beta_t, reset_hint = self.l4.predict(
                x_obs, reference, fdd_for_l4, monitor_alarm="OK")
            if self.l4_trim_free is not None:
                r_eff = self.l4_trim_free.apply(
                    r_eff[: self.l4.cfg.n_ref_dim],
                    x_obs=x_obs, base_reference=reference)
            self._last_r_eff = np.asarray(r_eff, dtype=np.float64).copy()
            self._last_beta = float(beta_t)
            self._last_reset_hint = bool(reset_hint)
            reference_for_l3 = r_eff
        else:
            self._last_r_eff = None
            self._last_beta = 0.0
            self._last_reset_hint = False
            reference_for_l3 = reference

        u_iadp, omega_ref = self.middle.predict(x_obs, reference_for_l3, time_step)
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

        if self.l1 is not None:
            # IADP RLS estimates ``F̃, G̃`` over the augmented state
            # (n_aug = 2*n_state); slice to the first ``n_state`` rows/cols
            # so the shield sees a square (n_state, n_state) Jacobian.
            F_full = self.middle.base.F[: self.n_state, : self.n_state]
            G_full = self.middle.base.G[: self.n_state, : self.n_control]
            self.l1.set_dynamics_jacobian(F_full, G_full)
            fdd = self._last_fdd if self._last_fdd is not None \
                else _zero_fdd_output(self.n_state)
            x_obs_arr = np.asarray(x_obs, dtype=np.float64).reshape(-1)[: self.n_state]
            out = self.l1.filter(x_obs_arr, u_indi, fdd, monitor_alarm="OK")
            u_out = np.asarray(out.u_safe, dtype=np.float64).reshape(-1)
            self._last_l1_out = out
        else:
            u_out = u_indi
            self._last_l1_out = None

        self._last_u_safe = u_out.copy()
        return u_out

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
            u_for_fdd = (self._last_u_safe
                         if self._last_u_safe is not None
                         else self._last_u_indi)
            self._last_fdd = self.fdd.step(next_x_obs, u_for_fdd)

        middle_diag = self.middle.learn(
            next_x_obs, reference, time_step=time_step, fdd=self._last_fdd,
        )

        if (self.l4 is not None
                and self._last_u_safe is not None
                and self._last_r_eff is not None):
            from tensoraerospace.agent.uftc.l4 import Transition
            r_eff_vec = np.asarray(self._last_r_eff, dtype=np.float64).copy()
            n_ref = int(self.l4.cfg.n_ref_dim)
            n_action = int(self.l4.cfg.n_action)
            x_for_err = np.asarray(next_x_obs, dtype=np.float64).reshape(-1)[:n_ref]
            r_for_err = r_eff_vec.reshape(-1)[:n_ref]
            # Project ``u_safe`` (n_control,) into action-space (n_action,) by
            # right-pad / truncate. This is the off-policy correction:
            # ``a_actual`` is the env-applied action embedded in actor space.
            u_raw = np.asarray(self._last_u_safe, dtype=np.float64).reshape(-1)
            a_actual = np.zeros(n_action, dtype=np.float64)
            a_actual[: min(n_action, u_raw.size)] = u_raw[: min(n_action, u_raw.size)]
            self.l4.learn(Transition(
                s=np.asarray(next_x_obs, dtype=np.float64).reshape(-1)[:self.n_state].copy(),
                a_actual=a_actual,
                r_used=r_eff_vec,
                reward=float(-(np.linalg.norm(x_for_err - r_for_err) ** 2)),
                s_next=np.asarray(next_x_obs, dtype=np.float64).reshape(-1)[:self.n_state].copy(),
                done=False,
                fdd=self._last_fdd,
                alarm="OK",
            ))

        # Phase 4 — composite Lyapunov monitor (passive: collect VState, step,
        # dispatch macro-actions). Runs after middle.learn so RLS state is
        # current; failures are caught at the controller boundary.
        monitor_block: Optional[dict] = None
        dispatch_diag: dict = {}
        if self.monitor is not None:
            try:
                from tensoraerospace.agent.uftc.monitor import collect_vstate
                vstate = collect_vstate(self)
                self._monitor_out = self.monitor.step(vstate)
                self._monitor_alarm = self._monitor_out.alarm
                if self.dispatcher is not None:
                    dispatch_diag = self.dispatcher.dispatch(
                        self._monitor_out.interventions, self._step,
                    )
                self._last_dispatch_diag = dispatch_diag
                monitor_block = {
                    "alarm": str(self._monitor_out.alarm),
                    "V_total": float(self._monitor_out.V_total),
                    "mu_uub_pred": float(self._monitor_out.mu_uub_pred),
                    "margin": float(self._monitor_out.margin),
                }
            except Exception:                          # pragma: no cover
                monitor_block = None

        self._step += 1
        out = {
            **{f"inner_{k}": v for k, v in inner_diag.items()},
            **{f"middle_{k}": v for k, v in middle_diag.items()},
            "fault_present": self._last_fdd.fault_present,
            "fdd_severity": self._last_fdd.severity,
            "fdd_confidence": self._last_fdd.confidence,
            "fdd_innovation_norm": self._last_fdd.innovation_norm,
            "fdd": {
                "fault_present": bool(self._last_fdd.fault_present),
                "severity": float(self._last_fdd.severity),
                "severity_abrupt": float(self._last_fdd.severity_abrupt),
                "severity_gradual": float(self._last_fdd.severity_gradual),
                "fault_kind": str(self._last_fdd.fault_kind),
                "confidence": float(self._last_fdd.confidence),
                "innovation_norm": float(self._last_fdd.innovation_norm),
                "time_since_event": float(self._last_fdd.time_since_event),
            },
        }
        if monitor_block is not None:
            out["monitor"] = monitor_block
            out.update(dispatch_diag)
        return out

    def diagnostics(self) -> dict:
        """Snapshot of all sub-components for logging / plotting."""
        diag = {
            "step": int(self._step),
            "fault_present": bool(self._last_fdd.fault_present),
            "severity": float(self._last_fdd.severity),
            "confidence": float(self._last_fdd.confidence),
            "innovation_norm": float(self._last_fdd.innovation_norm),
            "rls_gamma": float(self.middle.base.rls.gamma_rls),
            "mode": str(self.inner.mode),
            "fdd_ready": bool(self._fdd_ready),
            "fdd_active": bool(self._fdd_active),
            "fdd": {
                "fault_present": bool(self._last_fdd.fault_present),
                "severity": float(self._last_fdd.severity),
                "severity_abrupt": float(self._last_fdd.severity_abrupt),
                "severity_gradual": float(self._last_fdd.severity_gradual),
                "fault_kind": str(self._last_fdd.fault_kind),
                "confidence": float(self._last_fdd.confidence),
                "innovation_norm": float(self._last_fdd.innovation_norm),
                "time_since_event": float(self._last_fdd.time_since_event),
                "glr_active": bool(self.fdd.glr is not None),
            },
        }
        if self.l1 is not None:
            last = getattr(self, "_last_l1_out", None)
            diag["l1"] = {
                "enabled": True,
                "severity": float(self._last_fdd.severity),
                "hjb_value": (float(last.hjb_value)
                              if last is not None else 0.0),
                "intervention_norm": (float(last.intervention_norm)
                                      if last is not None else 0.0),
                "active": (bool(last.active) if last is not None else False),
            }
        if self.l4 is not None:
            diag["l4"] = {
                "enabled": True,
                "beta_t": float(self._last_beta),
                "reset_hint": bool(self._last_reset_hint),
                "frozen_until": self.l4._frozen_until,
                "hold_mode": bool(self.l4._hold_mode),
                "replay_size": int(len(self.l4.replay)),
                "eval_mode": bool(self.l4.cfg.eval_mode),
            }
        if self.monitor is not None and self._monitor_out is not None:
            diag["monitor"] = {
                "enabled": True,
                "alarm": str(self._monitor_out.alarm),
                "V_total": float(self._monitor_out.V_total),
                "mu_uub_pred": float(self._monitor_out.mu_uub_pred),
                "margin": float(self._monitor_out.margin),
            }
        return diag

    def reset(self) -> None:
        self.inner.reset()
        self.middle.reset()
        self.fdd.reset()
        if self.l1 is not None:
            self.l1.reset()
        if self.l4 is not None:
            self.l4.reset()
        self._step = 0
        self._fdd_active = self._fdd_ready and self.cfg.fdd_warmup_steps == 0
        self._last_fdd = _zero_fdd_output(self.n_state)
        self._last_u_indi.fill(0.0)
        self._last_u_safe = None
        self._last_l1_out = None
        self._last_r_eff = None
        self._last_beta = 0.0
        self._last_reset_hint = False
        self._last_omega_ref.fill(0.0)
        if self.monitor is not None:
            self.monitor.reset()
            from tensoraerospace.agent.uftc.monitor import MonitorOutput
            self._monitor_out = MonitorOutput.zero()
            self._monitor_alarm = "OK"
            self._last_dispatch_diag = {}

    def _extract_omega(self, x_obs: np.ndarray) -> np.ndarray:
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        if self._omega_indices is None:
            return x[: self.n_state]
        return x[self._omega_indices]

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


# ---------------------------------------------------------------------------
# Save / from_pretrained — multi-component layout.
#
# A saved UFTCController is a directory with the following children:
#
#     <save_dir>/
#     ├── config.json           # UFTCConfig + nested configs (numpy → list)
#     ├── manifest.json         # subdir names produced by inner/middle saves
#     ├── controller_state.npz  # _step, _last_u, _last_omega_ref, sm state
#     ├── inner/<sub>/          # AAINDIAgent.save() output
#     ├── middle/<sub>/         # IADPAgent.save() output
#     └── fdd/
#         ├── kalman.npz        # F, G, Q, R, x_hat, P
#         └── cpd.npz           # cusum, alarm, cooldown, time_since_alarm
# ---------------------------------------------------------------------------

import dataclasses as _dataclasses
import datetime as _datetime
import json as _json
from pathlib import Path as _Path
from typing import Union as _Union


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if _dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v)
                for k, v in _dataclasses.asdict(obj).items()
                if k != "history"}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable_for_iadpconfig(payload: dict) -> dict:
    arr_keys = ("Q", "R", "F_init", "G_init", "P_init", "excitation_signal")
    out = dict(payload)
    for k in arr_keys:
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], dtype=np.float64)
    return out


def _uftc_save(self, path: _Union[str, _Path, None] = None) -> str:
    base = _Path.cwd() if path is None else _Path(path)
    run_dir = base / (
        _datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        + f"_{self.__class__.__name__}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_payload = {
        "policy": {
            "name": (f"{self.__class__.__module__}."
                     f"{self.__class__.__name__}"),
            "params": {
                "n_state": self.n_state,
                "n_control": self.n_control,
            },
            "config": _to_jsonable(self.cfg),
        },
    }
    (run_dir / "config.json").write_text(_json.dumps(cfg_payload, indent=2))

    inner_dir = self.inner.base.save(run_dir / "inner")
    middle_dir = self.middle.base.save(run_dir / "middle")

    fdd_dir = run_dir / "fdd"
    fdd_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        fdd_dir / "kalman.npz",
        F=self.fdd.kalman.F, G=self.fdd.kalman.G,
        Q=self.fdd.kalman.Q, R=self.fdd.kalman.R,
        x_hat=self.fdd.kalman.x_hat, P=self.fdd.kalman.P,
    )
    np.savez(
        fdd_dir / "cpd.npz",
        cusum=np.asarray(self.fdd.cpd._cusum),
        alarm=np.asarray(self.fdd.cpd._alarm),
        cooldown=np.asarray(self.fdd.cpd._cooldown),
        time_since_alarm=np.asarray(self.fdd.cpd._time_since_alarm),
    )

    prev_omega = self.inner._prev_omega
    np.savez(
        run_dir / "controller_state.npz",
        step=np.asarray(self._step),
        last_u=self._last_u_indi,
        last_omega_ref=self._last_omega_ref,
        fdd_active=np.asarray(self._fdd_active),
        fdd_ready=np.asarray(getattr(self, "_fdd_ready", self._fdd_active)),
        sm_s=self.inner.sm_obs._s,
        sm_z=self.inner.sm_obs._z,
        mode=np.asarray(self.inner.mode),
        prev_omega=(prev_omega if prev_omega is not None
                    else np.zeros(self.n_inner_state, dtype=np.float64)),
        has_prev_omega=np.asarray(prev_omega is not None),
        fdd_fault_present=np.asarray(self._last_fdd.fault_present),
        fdd_severity=np.asarray(self._last_fdd.severity),
        fdd_confidence=np.asarray(self._last_fdd.confidence),
        fdd_innovation_norm=np.asarray(self._last_fdd.innovation_norm),
        fdd_time_since_event=np.asarray(self._last_fdd.time_since_event),
    )

    (run_dir / "manifest.json").write_text(_json.dumps({
        "inner_subdir": _Path(inner_dir).name,
        "middle_subdir": _Path(middle_dir).name,
    }, indent=2))
    return str(run_dir)


@classmethod
def _uftc_from_pretrained(cls, repo_name: str,
                          access_token=None, version=None):
    p = _Path(str(repo_name)).expanduser()
    if p.is_dir():
        return cls._load_from_dir(p)
    from huggingface_hub import snapshot_download
    folder = snapshot_download(repo_id=str(repo_name),
                               token=access_token, revision=version)
    return cls._load_from_dir(_Path(folder))


@classmethod
def _uftc_load_from_dir(cls, folder: _Path):
    cfg_payload = _json.loads((folder / "config.json").read_text())
    params = cfg_payload["policy"]["params"]
    cfg_dict = cfg_payload["policy"]["config"]

    inner_cfg = AAINDIConfig(**cfg_dict["inner_cfg"])
    middle_cfg = IADPConfig(**_from_jsonable_for_iadpconfig(
        cfg_dict["middle_cfg"]))
    fdd_cfg = FDDConfig(**cfg_dict["fdd_cfg"])
    rls_pol = RLSResetPolicy(**cfg_dict["rls_reset_policy"])

    cfg = UFTCConfig(
        **{k: v for k, v in cfg_dict.items() if k not in (
            "inner_cfg", "middle_cfg", "fdd_cfg", "rls_reset_policy")},
        inner_cfg=inner_cfg, middle_cfg=middle_cfg,
        fdd_cfg=fdd_cfg, rls_reset_policy=rls_pol,
    )

    ctl = cls(
        n_state=int(params["n_state"]),
        n_control=int(params["n_control"]),
        nominal_F=np.zeros((params["n_state"], params["n_state"])),
        nominal_G=np.zeros((params["n_state"], params["n_control"])),
        config=cfg,
    )

    manifest = _json.loads((folder / "manifest.json").read_text())
    ctl.inner.base = AAINDIAgent._load_from_dir(
        folder / "inner" / manifest["inner_subdir"])
    ctl.middle.base = IADPAgent._load_from_dir(
        folder / "middle" / manifest["middle_subdir"])
    ctl.middle._gamma_nominal = ctl.middle.base.rls.gamma_rls

    with np.load(folder / "fdd" / "kalman.npz") as npz:
        ctl.fdd.kalman.F = npz["F"]
        ctl.fdd.kalman.G = npz["G"]
        ctl.fdd.kalman.Q = npz["Q"]
        ctl.fdd.kalman.R = npz["R"]
        ctl.fdd.kalman.x_hat = npz["x_hat"]
        ctl.fdd.kalman.P = npz["P"]
    with np.load(folder / "fdd" / "cpd.npz") as npz:
        ctl.fdd.cpd._cusum = float(npz["cusum"])
        ctl.fdd.cpd._alarm = bool(npz["alarm"])
        ctl.fdd.cpd._cooldown = int(npz["cooldown"])
        ctl.fdd.cpd._time_since_alarm = int(npz["time_since_alarm"])

    with np.load(folder / "controller_state.npz") as npz:
        ctl._step = int(npz["step"])
        ctl._last_u_indi = npz["last_u"]
        ctl._last_omega_ref = npz["last_omega_ref"]
        ctl._fdd_active = bool(npz["fdd_active"])
        if "fdd_ready" in npz.files:
            ctl._fdd_ready = bool(npz["fdd_ready"])
        ctl.inner.sm_obs._s = npz["sm_s"]
        ctl.inner.sm_obs._z = npz["sm_z"]
        ctl.inner.mode_switch._mode = str(npz["mode"])
        if "has_prev_omega" in npz.files and bool(npz["has_prev_omega"]):
            ctl.inner._prev_omega = npz["prev_omega"].copy()
        else:
            ctl.inner._prev_omega = None
        if "fdd_severity" in npz.files:
            ctl._last_fdd = FDDOutput(
                fault_present=bool(npz["fdd_fault_present"]),
                severity=float(npz["fdd_severity"]),
                confidence=float(npz["fdd_confidence"]),
                innovation_norm=float(npz["fdd_innovation_norm"]),
                time_since_event=float(npz["fdd_time_since_event"]),
            )
    return ctl


UFTCController.save = _uftc_save
UFTCController.from_pretrained = _uftc_from_pretrained
UFTCController._load_from_dir = _uftc_load_from_dir
