# UFTC Phase 4 — Composite Lyapunov runtime monitor + UUB lemma

**Date:** 2026-05-08
**Status:** Draft, pending implementation plan
**Master spec:** `2026-05-08-uftc-cascade-extension-design.md`
**Predecessors:** Phase 1 MVP, Phase 2 (L1+GLR), Phase 3 (L4 D-SAC).

## 1. Scope

Independent runtime safety monitor for the UFTC cascade. Reads state from L1, L2, L3, L4, FDD; computes a composite Lyapunov scalar `V_total`; raises a 3-level ALARM with hysteresis; and dispatches a bounded set of macro-actions to the other layers (Variant B — advisory + macro-actions). Includes:

1. **5-component composite Lyapunov function** `V_total = c^T · v(t)` with `v = (V_HJ, V_INDI, V_iADP, V_DSAC, V_FDD)`.
2. **Numerical UUB certificate** — script that verifies the Metzler-comparison lemma's hypotheses (Hurwitz-Metzler matrix, conformal margins, Lipschitz bounds) on saved artifacts and reports `μ̂_uub` together with empirical pass-rate over 7 damage-preset rollouts.
3. **Macro-action dispatcher** — explicit calls to `IADPMiddle.force_reset()`, `DSACOuter.freeze_learning()`, `DSACOuter.degrade_reference_to_hold()`, `HJReachabilityShield.request_actuator_hold()`.
4. **Lemma 4.1** — vector-comparison UUB statement, formalised in `article/uftc-architecture-mai/main.tex` § 7.

Gated behind `UFTCConfig.enable_monitor`. Default `False` keeps prior behaviour bit-identical.

## 2. Package layout

```
tensoraerospace/agent/uftc/monitor/
├── __init__.py
├── components.py       # 5 component extractors V_i (NaN-guarded)
├── composite.py        # CompositeLyapunovMonitor + VState/MonitorOutput dataclasses
├── intervention.py     # MacroAction discriminated union + MacroActionDispatcher
├── alarm.py            # 3-level alarm with hysteresis and cooldown
├── certificate.py      # numerical UUB certificate script
└── README.md
```

## 3. Components V_i

Walkthrough §15.3 names five summands. Each is normalised by a per-component cap `V_max_i` collected during warm-up so that `V̂_i = V_i / V_max_i ∈ [0, 1+]`.

| i | V_i | Source | Normalisation `V_max_i` |
|---|---|---|---|
| 1 | `V_HJ = max(0, ε_t − V_θ(x))` | L1 shield (`shield._last_value`, `conformal._last_eps`) | `2·ε_t_warmup_p99` |
| 2 | `V_INDI = ½ ‖ω − ω_ref‖²` | L2 (`WrappedAAINDI._last_omega`, `_last_omega_ref`) | `‖ω_max − ω_min‖² / 4` |
| 3 | `V_iADP = ½ x_errᵀ P̃ x_err` | L3 (`IADPMiddle.base.P_critic`, `_last_state_error`) | `½ x_err_max² · λ_max(P̃)` |
| 4 | `V_DSAC = max(0, var(Z) − var_target)` | L4 (`DSACOuter.critic._last_z`) | `var_max_warmup_p95` |
| 5 | `V_FDD = severity_abrupt + severity_gradual` | FDD (`FDDOutput`) | `2.0` (severity normalised to ~1.0 at threshold) |

`components.py` exposes one extractor per index plus a NaN-guarded aggregator:

```python
@dataclass
class VState:
    V_hj: float
    V_indi: float
    V_iadp: float
    V_dsac: float
    V_fdd: float
    timestamp: float

def collect_vstate(controller: "UFTCController") -> VState:
    return VState(
        V_hj   = _safe(extract_v_hj(controller)) if controller.cfg.enable_l1_shield else 0.0,
        V_indi = _safe(extract_v_indi(controller)),
        V_iadp = _safe(extract_v_iadp(controller)),
        V_dsac = _safe(extract_v_dsac(controller)) if controller.cfg.enable_l4_outer else 0.0,
        V_fdd  = _safe(extract_v_fdd(controller)),
        timestamp = controller._step * controller.cfg.dt,
    )

def _safe(x: float) -> float:
    return 0.0 if (x is None or math.isnan(x) or math.isinf(x)) else float(x)
```

A NaN in any component does not crash — it degrades to 0 with a warning.

## 4. Composite monitor

```python
@dataclass
class MonitorConfig:
    c_weights: np.ndarray          # shape (5,), nonneg, sum to 1
    eps_matrix: np.ndarray         # shape (5,5), Metzler (off-diag ≥ 0)
    a_diag: np.ndarray             # shape (5,), positive — exponential rates a_i
    d_disturbance: np.ndarray      # shape (5,), nonneg — exogenous bound d_i
    alarm_warn_frac: float = 0.7
    alarm_critical_frac: float = 0.95
    cooldown_steps: int = 200

@dataclass
class MonitorOutput:
    V_total: float
    components: VState
    alarm: Literal["OK", "WARN", "CRITICAL"]
    mu_uub_pred: float
    margin: float
    interventions: list[MacroAction]

class CompositeLyapunovMonitor:
    def __init__(self, cfg: MonitorConfig): ...
    def step(self, vstate: VState) -> MonitorOutput: ...
    def reset(self): ...
```

`mu_uub_pred` is computed once at construction time from `M = diag(a) − ε`, `‖M⁻¹ d‖_c` (assuming Hurwitz-Metzler — verified by `certificate.py`).

`alarm` uses hysteresis:

```python
warn_thresh = cfg.alarm_warn_frac * mu_uub_pred
crit_thresh = cfg.alarm_critical_frac * mu_uub_pred
clear_warn = 0.5 * warn_thresh
clear_crit = 0.5 * crit_thresh

# rising-edge: V_total > thresh; falling-edge: V_total < clear AND time_in_state > cooldown
```

## 5. Macro-actions (Variant B)

```python
@dataclass
class MacroAction:
    kind: Literal[
        "force_rls_reset",
        "freeze_l4_learning",
        "degrade_reference_to_hold",
        "request_actuator_hold",
    ]
    payload: dict = field(default_factory=dict)

class MacroActionDispatcher:
    def __init__(self, *, l3: IADPMiddle, l4: DSACOuter | None,
                 l1: HJReachabilityShield | None): ...
    def dispatch(self, actions: list[MacroAction], current_step: int) -> dict:
        diag = {}
        for a in actions:
            try:
                if a.kind == "force_rls_reset":
                    self.l3.force_reset(severity_hint=a.payload.get("severity", 1.0))
                    diag["force_rls_reset"] = current_step
                elif a.kind == "freeze_l4_learning" and self.l4 is not None:
                    self.l4.freeze_learning(until_step=current_step + a.payload["duration"])
                    diag["freeze_l4_learning_until"] = current_step + a.payload["duration"]
                elif a.kind == "degrade_reference_to_hold" and self.l4 is not None:
                    self.l4.degrade_reference_to_hold()
                    diag["degrade_reference_to_hold"] = current_step
                elif a.kind == "request_actuator_hold" and self.l1 is not None:
                    self.l1.request_actuator_hold()
                    diag["request_actuator_hold"] = current_step
            except Exception as e:                  # monitor never crashes the loop
                LOG.warning("macro-action %s failed: %s", a.kind, e)
        return diag
```

Triggers (rule-based, deterministic):

| Condition | Actions enqueued |
|---|---|
| `alarm == "WARN"` | `freeze_l4_learning` (duration=cooldown_steps) |
| `alarm == "CRITICAL"` | `force_rls_reset` + `freeze_l4_learning(2·cooldown)` + `degrade_reference_to_hold` |
| `V_total > μ̂_uub` (one-tick burst) | + `request_actuator_hold` (one tick) |

Emitted from `step()` in `MonitorOutput.interventions`. UFTCController's `_dispatch_interventions` calls `MacroActionDispatcher.dispatch`.

## 6. Lemma 4.1 (vector-comparison UUB)

Formal statement, to appear in `article/uftc-architecture-mai/main.tex` § 7.

> **Lemma 4.1** Suppose each component `V_i(t) ≥ 0`, `i ∈ {1, …, 5}`, satisfies the differential inequality
>
> ```
> V̇_i(t) ≤ −aᵢ Vᵢ(t) + Σ_{j ≠ i} εᵢⱼ Vⱼ(t) + dᵢ ,        aᵢ > 0, εᵢⱼ ≥ 0, dᵢ ≥ 0,
> ```
>
> almost everywhere on the system trajectory. Let `M ≡ diag(a) − ε` (where `ε` is the off-diagonal-only Metzler matrix). If `M` is Hurwitz, then for every nonnegative weight vector `c ∈ ℝ⁵_{≥0}` with `Σ cᵢ = 1`,
>
> ```
> V_total(t) = c^⊤ v(t) ≤ ‖c‖₁ · ‖v(0)‖∞ · exp(−λ_min(M)·t) + ‖M⁻¹ d‖_c ,
> ```
>
> where `‖·‖_c` is the c-weighted norm. In particular,
>
> ```
> lim sup_{t → ∞} V_total(t) ≤ μ_uub ≡ ‖M⁻¹ d‖_c.
> ```

**Proof sketch.** Apply the vector-comparison principle (Khalil 2002 §9.5) to `v̇ ≤ −M v + d`. Since `M` is Hurwitz and Metzler, by Perron-Frobenius-type results for Metzler matrices, `M⁻¹` exists and is non-negative. The bound `v(t) ≤ exp(−Mt) v(0) + M⁻¹ d` follows component-wise; taking the c-weighted sum yields the stated inequality. ∎

**Lemma 4.1' (monitor-augmented).** With Variant-B macro-actions, replacing `aᵢ` by `aᵢ' = aᵢ + κᵢ` (where `κᵢ ≥ 0` reflects the contraction added by macro-actions on component `i`) leaves the conclusion unchanged with `μ_uub' ≡ ‖(M − diag(κ))⁻¹ d‖_c ≤ μ_uub`. Macro-actions only tighten the UUB ball.

The work's contribution: assembling concrete `(aᵢ, εᵢⱼ, dᵢ, κᵢ)` for each layer of the UFTC cascade, expressed in known parameters of L1–L4 and FDD:

| Term | Expression |
|---|---|
| `a₁` (V_HJ) | `λ_HJ-CBF / L(V_θ)` (HJ-CBF gain over Lipschitz upper-bound; see Phase 2 §3, §5) |
| `a₂` (V_INDI) | `min(k₁, k₂)` (super-twisting gains; Phase 1 §3.4) |
| `a₃` (V_iADP) | `(1 − γ) · γ_RLS` (Bellman contraction × RLS forgetting) |
| `a₄` (V_DSAC) | `(1 − γ) · η_critic` (distributional Bellman contraction; Phase 3 §3) |
| `a₅` (V_FDD) | `cooldown_steps⁻¹` (CUSUM/GLR exponential decay rate) |
| `εᵢⱼ` | tabulated from cross-layer signals (Phase 0 master spec §3.8) |
| `dᵢ` | conformal margin `ε_0` + sensor noise budget `σ_R` + `‖V_θ − V*‖_∞` from validation |
| `κᵢ` | nonzero on rows touched by triggered macro-actions; tabulated in §6 of this spec |

The numerical certificate (Section 7) plugs concrete values from the deployed config into this table.

## 7. Numerical certificate

`certificate.py` is an offline script:

```bash
python -m tensoraerospace.agent.uftc.monitor.certificate \
    --config artifacts/uftc/cfg.yaml \
    --rollouts artifacts/uftc/cert_rollouts.npz \
    --report artifacts/uftc/uub_certificate.json
```

Verifies, in order:

1. `eps_matrix` is Metzler (off-diagonal entries ≥ 0). Failure → exit 1, no further checks.
2. `M = diag(a_diag) − eps_matrix` is Hurwitz: all eigenvalues have positive real part (computed via `numpy.linalg.eig`). Failure → exit 1.
3. `mu_uub_pred = ‖M⁻¹ d‖_c` matches the closed-form computation (sanity-check vs `solve(M, d)`).
4. **Empirical pass-rate.** For each saved roll-out (one per damage preset, ≥ 100 seeds), check `V_total < mu_uub_pred` after the transient (configurable `transient_steps`). Pass criterion: ≥ 99 % of trajectories.

JSON report:

```json
{
  "metzler_check": "pass",
  "hurwitz_check": "pass",
  "lambda_min": 0.412,
  "mu_uub_pred": 1.234,
  "rollouts": {
    "nominal": {"n": 100, "transient_steps": 200, "pass_rate": 1.0, "max_v_total": 0.78},
    "wing_strike_left_tip": {"n": 100, "pass_rate": 0.99, "max_v_total": 1.21},
    "elevator_jam_neutral": {"n": 100, "pass_rate": 0.97, "max_v_total": 1.36},
    "engine_flameout":      {"n": 100, "pass_rate": 1.00, "max_v_total": 0.92}
  },
  "verdict": "pass"
}
```

JSON report is referenced from `main.tex` § 7 as the "verification certificate". CI runs the certificate on every Phase 4+ change and fails the build on `verdict != "pass"`.

## 8. UFTCController integration

`UFTCController.__init__` (when `enable_monitor=True`) constructs:

- `self.monitor: CompositeLyapunovMonitor` — built from `cfg.monitor_cfg`;
- `self.dispatcher: MacroActionDispatcher` — wired with `self.middle`, `self.l4` (or `None`), `self.l1` (or `None`);
- `self._monitor_out: MonitorOutput` — initialised to `MonitorOutput.zero()`;
- `self._monitor_alarm: str` — initialised to `"OK"`.

In `learn()`, after `fdd.step` and `middle.learn`, before returning diagnostics:

```python
if self.cfg.enable_monitor:
    vstate = collect_vstate(self)
    self._monitor_out = self.monitor.step(vstate)
    diag.update(self.dispatcher.dispatch(self._monitor_out.interventions, self._step))
    self._monitor_alarm = self._monitor_out.alarm
else:
    self._monitor_out = MonitorOutput.zero()
    self._monitor_alarm = "OK"
```

Monitor failure (any exception in `monitor.step` or `dispatcher.dispatch`) is caught at `UFTCController.learn()` boundary and logged; the loop continues with `_monitor_alarm = "OK"` for that tick. This guarantees that a monitor bug cannot disable flight control.

## 9. Tests

| File | Coverage |
|---|---|
| `tests/agents/uftc/monitor/test_components.py` | Each `V_i` extractor returns expected value on synthetic state; NaN/inf in any input → 0 with warning logged. |
| `tests/agents/uftc/monitor/test_composite.py` | `V_total` strictly monotone in increasing inputs; ALARM hysteresis between `c_warn`, `c_crit`, and clear thresholds; cooldown enforced. |
| `tests/agents/uftc/monitor/test_certificate_unit.py` | toy 5×5 Metzler with known analytical `M⁻¹ d` — script reports identical `mu_uub_pred`; Metzler-violation detected; Hurwitz-violation detected. |
| `tests/agents/uftc/monitor/test_macro_actions.py` | each macro-action calls correct method on correct layer; missing layer is a no-op (not crash); dispatcher swallows exceptions. |
| `tests/agents/uftc/test_uftc_monitor_alarm_propagation.py` | F-16 + WING_STRIKE: monitor reaches CRITICAL; β_t→1 next tick; conformal margin widens; `force_rls_reset` recorded in diagnostics; aircraft does not diverge. |
| `tests/agents/uftc/test_uftc_monitor_intervention_chain.py` | injected synthetic FDD-stream escalates from OK→WARN→CRITICAL→burst; expected macro-action sequence recorded in diagnostics with correct timestamps. |
| `tests/agents/uftc/test_uftc_monitor_phase123_invariance.py` | `enable_monitor=False` → `predict()/learn()` bit-identical to Phase 1+2+3 (composed invariance). |
| `tests/agents/uftc/test_uftc_full_cascade_smoke.py` | all flags on, mock-plant 1000 steps — no NaN, no exceptions, save/load round-trip identical. |
| `tests/agents/uftc/test_uftc_full_cascade_f16_engine_flameout.py` | F-16 + ENGINE_FLAMEOUT, all flags on: aircraft stays in UUB ball; GLR + monitor agree on detection; L4 trim-free converges within 5 % of degraded trim within 10s. |
| `tests/agents/uftc/test_uftc_monitor_uub_emp.py` | 7 damage-presets × 100 seeds: `≥ 99 %` of trajectories satisfy `V_total < μ̂_uub` after transient — empirical verification of Lemma 4.1. (Long; runs in CI nightly.) |

Coverage target: ≥ 80 % line coverage on `tensoraerospace/agent/uftc/monitor/`.

## 10. Latex deliverables

In `article/uftc-architecture-mai/main.tex`:

- **§ 7 (new).** Vector-comparison UUB-bound for the UFTC cascade. Subsections:
  - 7.1 Component definitions `V_i`.
  - 7.2 Hypotheses on `(aᵢ, εᵢⱼ, dᵢ, κᵢ)` — table mapping each entry to UFTC parameters.
  - 7.3 Lemma 4.1 statement.
  - 7.4 Lemma 4.1' (monitor-augmented).
  - 7.5 Proof (full, ~1.5 pages).
  - 7.6 Numerical certificate — concrete values, JSON-report-derived; trade-off discussion (conformal tightness vs μ_uub, RL learning rate vs `a₃`).
- **Bibliography update**: Khalil 2002 §9.5; Bansal & Tomlin 2021; Dabney 2018; Basseville & Nikiforov 1993; Wabersich & Zeilinger 2018; Haarnoja 2018.

## 11. Out of scope for Phase 4

- SOS / SDP-based formal certificate (Phase 4 uses analytical lemma + numerical Metzler/Hurwitz check, not symbolic SOS).
- α-β-CROWN tightened Lipschitz (Phase 2.1; if delivered, the certificate script auto-uses tighter `‖V_θ − V*‖_∞`).
- Probabilistic UUB through conformal hold-out (Phase 4.1; would replace the Metzler argument with statistical guarantee).

## 12. References

- Khalil, H. (2002), *Nonlinear Systems*, 3rd ed., §9.5 — Vector comparison principle.
- Berman, A. & Plemmons, R. (1994), *Nonnegative Matrices in the Mathematical Sciences* — Metzler / M-matrices.
- Bansal & Tomlin (2021), DeepReach — V_θ artefact this monitor consumes.
- Dabney et al. (2018), QR — variance-of-Z monitor signal.
- Haarnoja et al. (2018), SAC — entropy temperature feeding `α₄`.
- Master spec: `2026-05-08-uftc-cascade-extension-design.md`.
- Phase 2 sub-spec: `2026-05-08-uftc-l1-hjshield-and-glr-design.md`.
- Phase 3 sub-spec: `2026-05-08-uftc-l4-dsac-cvar-design.md`.
