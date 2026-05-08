# UFTC Cascade Extension — Master Spec for Phases 2/3/4

**Date:** 2026-05-08
**Status:** Draft, pending implementation plans
**Predecessor:** `2026-05-07-uftc-phase1-mvp-design.md` (L2+L3+FDD/CUSUM MVP)
**Source:** `~/Documents/tensoraerospace/tensoraeropsace/wiki/synthesis/uftc-extended-walkthrough.md`

## 1. Background

Phase 1 MVP shipped:

- L2 inner — `WrappedAAINDI` + `SuperTwistingObserver` + `ModeSwitcher`;
- L3 middle — `IADPMiddle` with FDD-triggered RLS reset;
- FDD — `NominalKalman` + `ChangePointDetector` (CUSUM on Mahalanobis innovation);
- `UFTCController` orchestrator with multi-component save/load.

What is left from the v2 walkthrough (§3, §15):

| Layer / Component | Role | Phase |
|---|---|---|
| L1 HJ-Reachability shield + conformal margin + bank | Safety post-filter on `u_indi`. | **2** |
| GLR detector for slow-drift (gradual) faults | Parallel channel inside `FDDDetector`. | **2** |
| L4 distributional SAC outer + CVaR risk gate | Risk-aware reference planner + `β_t`. | **3** |
| Trim-free longitudinal reference under degraded plants | L4 wrapper that replaces static trim with adaptive. | **3** |
| Composite Lyapunov runtime monitor + UUB lemma | 5-component `V_total`, ALARM, macro-actions. | **4** |

This master spec documents the cascade-level integration: how L1, L4, and the monitor plug into the existing `UFTCController` without breaking Phase 1 contracts; what cross-layer signals flow; and what regression invariants must hold across phases.

Each phase has its own sub-spec with detailed component design, tests, and pre-training tooling. This document is the integration contract those sub-specs all conform to.

## 2. Cascade after extension

```
                    ┌────────────── reference r_t (user) ───────────────┐
                    │                                                    ▼
            [L4 D-SAC outer] ── r̃_t, β_t, reset_hint ─────────────►
                    │                                                    │
                    │   ◄──── env feedback ◄─────                       │
                    ▼                                                    ▼
            [L3 IADPMiddle (Phase 1)] ── u_iadp, ω_ref ─►
                                                          [L2 WrappedAAINDI (Phase 1)] ── u_indi ─►
                                                                                                    [L1 HJ-shield] ── u_safe ─► env
                                                                                       ▲                          │
                                                                                       │                          │
            ┌────────────────────── x_meas ◄───────────────────────────────────────────┘                          │
            ▼                                                                                                      │
     [FDDDetector (Phase 1) ⊕ GLRDetector] ─ FDDOutput ──► L3, L4, L1, monitor                                    │
            │                                                                                                      │
            ▼                                                                                                      │
     [CompositeLyapunovMonitor]                                                                                    │
            │   reads V_HJ, V_INDI, V_iADP, V_DSAC, V_FDD                                                         │
            │   emits V_total, ALARM, μ̂_uub, list[MacroAction]                                                    │
            ▼                                                                                                      │
     macro-actions:  L3.force_reset, L4.freeze_learning, L4.degrade_reference, L1.request_actuator_hold ◄─────────┘
```

## 3. Integration contracts

### 3.1 L1 (Phase 2)

```python
class HJReachabilityShield:
    def __init__(self, n_state, n_control, *,
                 value_fn: HJValueFunction,
                 conformal_margin_fn: Callable[[FDDOutput], float],
                 dt: float, ...): ...

    def filter(self, x: np.ndarray, u_nominal: np.ndarray,
               fdd: FDDOutput) -> ShieldOutput: ...
    def request_actuator_hold(self) -> None: ...   # macro-action sink (Phase 4)
    def reset(self): ...

@dataclass
class ShieldOutput:
    u_safe: np.ndarray
    intervention_norm: float
    hjb_value: float
    active: bool
```

The shield is a strict post-filter on `u_indi`. The QP it solves is detailed in the L1 sub-spec.

### 3.2 L4 (Phase 3)

```python
class DSACOuter(BaseRLModel):
    def __init__(self, n_state, n_ref_dim, n_control, *,
                 critic: QRDistCritic, actor: GaussianActor,
                 cvar_alpha: float, ...): ...

    def predict(self, x_obs, base_reference, fdd: FDDOutput,
                monitor_alarm: Literal["OK","WARN","CRITICAL"]) \
        -> tuple[np.ndarray, float, bool]:
        # returns (r̃_t, β_t, reset_hint)
        ...

    def learn(self, transition: Transition) -> dict: ...
    def freeze_learning(self, until_step: int) -> None: ...    # macro-action sink
    def degrade_reference_to_hold(self) -> None: ...           # macro-action sink
    def reset(self): ...
```

Replay buffer stores `u_actual = u_safe` (the action that actually entered the env), so off-policy bias from L1 clipping is corrected.

### 3.3 FDD extension (Phase 2)

```python
@dataclass
class FDDOutput:
    fault_present: bool
    fault_kind: Literal["none", "abrupt", "gradual", "compound"]
    severity_abrupt: float
    severity_gradual: float
    severity: float                # = max(abrupt, gradual)  — Phase 1 compatibility
    confidence: float
    innovation_norm: float
    time_since_event: float
    glr_drift_estimate: np.ndarray | None
```

Phase 1 consumers of `severity` keep working unchanged.

### 3.4 Monitor (Phase 4)

```python
@dataclass
class VState:
    V_hj: float
    V_indi: float
    V_iadp: float
    V_dsac: float
    V_fdd: float
    timestamp: float

@dataclass
class MonitorOutput:
    V_total: float
    components: VState
    alarm: Literal["OK", "WARN", "CRITICAL"]
    mu_uub_pred: float
    margin: float
    interventions: list[MacroAction]    # macro-action queue (Variant B)

class CompositeLyapunovMonitor:
    def step(self, vstate: VState) -> MonitorOutput: ...
    def reset(self): ...
```

`MacroAction` is a discriminated union — see the Phase 4 sub-spec.

### 3.5 UFTCController extensions (controller.py)

```python
@dataclass
class UFTCConfig:
    # ... all Phase 1 fields ...

    # NEW
    enable_l1_shield: bool = False
    enable_l4_outer: bool = False
    enable_monitor: bool = False
    enable_glr: bool = False

    l1_cfg: HJShieldConfig | None = None
    l4_cfg: DSACConfig | None = None
    monitor_cfg: MonitorConfig | None = None
    glr_cfg: GLRConfig | None = None
```

With all four flags `False`, behaviour is bit-identical to Phase 1 — protected by `tests/agents/uftc/test_uftc_phase1_invariance.py`.

### 3.6 `predict()` flow (extended)

```
1.  alarm = self._monitor_out.alarm if cfg.enable_monitor else "OK"
2.  fdd  = self._last_fdd
3.  if cfg.enable_l4_outer:
        r_eff, beta_t, reset_hint = self.l4.predict(x, reference, fdd, alarm)
    else:
        r_eff, beta_t, reset_hint = reference, 0.0, False
4.  u_iadp, omega_ref = self.middle.predict(x, r_eff, time_step, beta=beta_t)
5.  u_indi = self.inner.predict(omega_ref, omega_meas, alpha=alpha,
                                u_blend_target=u_iadp,
                                fault_severity=fdd.severity,
                                time_step=time_step)
6.  if cfg.enable_l1_shield:
        u_safe = self.l1.filter(x, u_indi, fdd).u_safe
    else:
        u_safe = u_indi
7.  return u_safe
```

### 3.7 `learn()` flow (extended)

```
1.  inner.learn(next_x, ...)                 # AAINDI VFF-RLS
2.  if step >= fdd_warmup_steps and step % fdd_update_every == 0:
        fdd_out = fdd.step(next_x, last_u_safe)   # NB: last_u_safe, not last_u_indi
        self._last_fdd = fdd_out
3.  middle.learn(next_x, r_eff, time_step, fdd=self._last_fdd, reset_hint=reset_hint)
4.  if cfg.enable_l4_outer:
        l4.learn(Transition(s, u_safe, r_eff, r, s_next, done, fdd, alarm))
5.  if cfg.enable_monitor:
        vstate = collect_vstate(self)
        self._monitor_out = self.monitor.step(vstate)
        diag.update(self.dispatcher.dispatch(self._monitor_out.interventions, self._step))
        self._monitor_alarm = self._monitor_out.alarm
6.  return diagnostics
```

`self.dispatcher: MacroActionDispatcher` is built by `__init__` when `enable_monitor=True` and is wired to `self.middle` (and optionally `self.l4`, `self.l1`).

The FDD step uses `last_u_safe` (what the plant actually saw), not `last_u_indi`. This keeps the residual `ν = x_meas − F·x − G·u_safe` consistent with the dynamics under L1 clipping.

### 3.8 Cross-layer signals

| From | To | Channel | Purpose |
|---|---|---|---|
| FDD | L1 | `conformal_margin_fn(fdd)` | tighten ε under fault |
| FDD | L3 | RLS reset rising-edge (Phase 1) | re-adapt under fault |
| FDD | L4 | `predict(... fdd ...)` | risk gate driver |
| FDD | monitor | `V_FDD` component | aggregate severity |
| L4 | L3 | `r̃_t`, `β_t`, `reset_hint` | risk-modulated lookahead/trust |
| L1 | L4 (via replay) | `u_actual = u_safe` | off-policy correction |
| Monitor | L3 | `force_rls_reset` macro-action | force adaptation |
| Monitor | L4 | `freeze_learning`, `degrade_reference` | conservative under WARN/CRITICAL |
| Monitor | L1 | `request_actuator_hold` | one-tick safety hold |

## 4. Init lifecycle

1. `UFTCController.__init__` builds Phase 1 components as before; conditionally constructs `l1`, `l4`, `monitor` if flags are set; `MacroActionDispatcher` wires monitor → other layers.
2. Warm-up `fdd_warmup_steps` (Phase 1) — FDD inactive, L4 in eval-mode without learning, L1 disabled-pass-through, monitor inactive (`MonitorOutput.zero()`).
3. After warm-up: FDD activates → all dependent layers go online.

If user passes pre-trained `nominal_F/nominal_G`, value-network artifacts (`l1_cfg.value_fn_path`), and DSAC weights (`l4_cfg.weights_path`), warm-up can be set to 0.

## 5. Save / load layout (extended)

```
<save_dir>/
├── config.json
├── inner/                  # Phase 1
├── middle/                 # Phase 1
├── fdd/
│   ├── kalman.npz          # Phase 1
│   ├── cpd.npz             # Phase 1
│   └── glr.npz             # NEW (Phase 2)
├── l1/                     # NEW (Phase 2)
│   ├── value_fn.pt
│   ├── value_fn.json
│   ├── bank/               # optional bank artifacts
│   │   ├── nominal/
│   │   ├── elevator_jam/
│   │   └── ...
│   └── conformal.json
├── l4/                     # NEW (Phase 3)
│   ├── actor.pt
│   ├── critic.pt
│   ├── critic_target.pt
│   ├── log_alpha.pt
│   ├── replay.npz          # optional
│   └── dsac_config.json
├── monitor/                # NEW (Phase 4)
│   ├── monitor_config.json
│   ├── certificate.json
│   └── alarm_state.npz
└── controller_state.npz    # extended: + last r_eff, last beta, last alarm
```

`from_pretrained` reconstructs each subdirectory with the corresponding sub-component's loader, then re-instantiates `UFTCController`.

## 6. Regression invariance

`tests/agents/uftc/test_uftc_phase1_invariance.py`:

- Configures `UFTCController` with `enable_l1_shield=enable_l4_outer=enable_monitor=enable_glr=False`.
- Runs 1000-step seed-fixed roll-out on the F-16 nonlinear angular plant (no damage).
- Asserts that `predict()` and `learn()` outputs are identical (atol=0, rtol=0) to a Phase-1-pinned reference roll-out.

This test is created in Phase 2 and must remain green throughout Phases 3 and 4. Any merge that breaks it requires explicit migration plan.

## 7. Phase ordering and dependencies

- Phase 2 (L1 + GLR) — independent of Phases 3/4. Can ship first.
- Phase 3 (L4) — uses Phase 2's `u_actual = u_safe` for replay. If Phase 2 is delayed, Phase 3 falls back to `u_actual = u_indi` — slightly biased, fixed when Phase 2 lands.
- Phase 4 (monitor) — requires both Phase 2 (`V_HJ`) and Phase 3 (`V_DSAC`) for full 5-component `V_total`. With Phases 2/3 absent, monitor still runs with `V_HJ = V_DSAC = 0` — informative but degenerate.

Phases 2 and 3 can be implemented in parallel sessions (`l1/` and `l4/` share no files); Phase 4 follows.

## 8. Testing strategy

Each sub-spec lists its own tests. The master spec adds these cross-cutting cases:

| File | What it covers |
|---|---|
| `tests/agents/uftc/test_uftc_phase1_invariance.py` | bit-identical Phase 1 behaviour with all flags off (created in Phase 2). |
| `tests/agents/uftc/test_uftc_full_cascade_smoke.py` | 1000-step smoke run with all flags on, mock-plant — no NaN, `diagnostics()` populated, save/load round-trip identical (created in Phase 4). |
| `tests/agents/uftc/test_uftc_full_cascade_f16_engine_flameout.py` | F-16 + ENGINE_FLAMEOUT damage, all flags on, end-to-end success metric: aircraft stays in UUB ball, GLR + monitor agree, L4 trim-free converges (created in Phase 4). |

## 9. Out of scope for this spec

- Online D-SAC learning inside `UFTCController.learn()` while flying (deferred to Phase 3.1 — eval-only at Phase 3).
- α-β-CROWN formal Lipschitz certificate of `V_θ` (deferred to Phase 2.1 — power-iteration upper-bound at Phase 2).
- SOS/SDP-based UUB certificate (Phase 4 uses analytical lemma + numerical Metzler check).
- MMAE bank classifier between known fault modes (open-world detection via GLR + CUSUM is sufficient for MVP).

## 10. References

- `2026-05-07-uftc-phase1-mvp-design.md` — predecessor (L2+L3+FDD-CUSUM MVP).
- `~/Documents/tensoraerospace/tensoraeropsace/wiki/synthesis/uftc-extended-walkthrough.md` §3, §15 — primary architectural source.
- Bansal, S. et al. (2017) "Hamilton-Jacobi reachability: A brief overview and recent advances," CDC — HJ-reachability foundation.
- Bansal & Tomlin (2021) "DeepReach: A Deep Learning Approach to High-Dimensional Reachability," ICRA — V_θ training.
- Dabney, W. et al. (2018) "Distributional Reinforcement Learning with Quantile Regression," AAAI — QR critic.
- Khalil, H. (2002), Nonlinear Systems (3rd ed.), §9.5 — vector-comparison principle for UUB.
- Basseville, M. & Nikiforov, I. (1993) Detection of Abrupt Changes — GLR baseline.
- Sieberling, S. et al. (2010) — INDI baseline.
- Konatala, R. et al. (2024) — iADP middle reference.
