# UFTC Phase 1 MVP — Unified Fault-Tolerant Control orchestrator (L2 + L3 + FDD)

**Date:** 2026-05-07
**Status:** Draft, pending implementation plan
**Source:** `~/Documents/tensoraerospace/tensoraeropsace/wiki/synthesis/uftc-extended-walkthrough.md`

## 1. Background

UFTC (Unified Fault-Tolerant Control) is a 6-layer online active fault-tolerant
flight-control architecture from Mazaev's research wiki (v2 walkthrough,
2026-05-02). The full v2 design composes:

| Layer | Component | Status in this spec |
|---|---|---|
| L4 | Distributional SAC outer + risk gate β_t | Out of scope (Phase 3) |
| L3 | iADP middle (GDHP critic + TD(λ) + Retrace + bank-switching) | **In scope, simplified** |
| L2 | A-INDI inner + super-twisting SM observer + rate↔angle mode switch | **In scope** |
| L1 | HJ-Reachability safety shield | Out of scope (Phase 2) |
| — | Parallel FDD-bank (K+1 Kalman + MMAE + dwell guard) | **In scope, redesigned** |
| — | Vector Lyapunov runtime monitor | Out of scope (Phase 4) |

This spec covers **Phase 1 MVP**: L2 inner extensions + L3 middle with online
adaptation + general fault detector + UFTCController orchestrator. Subsequent
phases bolt on L1, L4, monitor, and bank pre-training tooling.

### 1.1 Why this scope

- The walkthrough §16.2.1 itself recommends Phase 1 = L2 + L3 + FDD as the MVP.
- The most risky integration concern (cross-layer signal flow with online
  adaptation) is fully exercised at this scope.
- Existing `aa_indi.AAINDIAgent` and `iadp.IADPAgent` cover L2 and L3 building
  blocks; the gap is the FDD detector and the orchestrator.

### 1.2 Plant-agnostic design

The algorithm is plant-agnostic — `UFTCController` accepts `n_state`,
`n_control`, and an optional warm-start `(F_nominal, G_nominal)`. F-16
nonlinear angular plus damage presets is the **reference example** for
integration tests and the example notebook, not a hardcoded dependency.

### 1.3 Open-world fault assumption

Unlike the walkthrough's K+1 MMAE-bank design, **MVP makes no closed-world
assumption about the fault catalog**. The detector flags *any* anomaly in
plant dynamics (innovation-driven CUSUM change-point detection) and the L3
middle reacts via covariance-inflation RLS reset. The user explicitly required
this orientation: «алгоритм должен находить любые отказы».

The walkthrough's bank-based mode classification is deferred to Phase 2 as an
*optional* augmentation for known fault classes — it never replaces the
generic detector.

## 2. Architecture

### 2.1 Package layout

```
tensoraerospace/agent/uftc/
├── __init__.py            # exports UFTCController, UFTCConfig, public submodules
├── controller.py          # UFTCController orchestrator + UFTCConfig
├── inner.py               # WrappedAAINDI, SuperTwistingObserver, ModeSwitcher
├── middle.py              # IADPMiddle, RLSResetPolicy
├── fdd/
│   ├── __init__.py
│   ├── kalman_3step.py    # NominalKalman (one adaptive 3-step Kalman)
│   ├── change_point.py    # ChangePointDetector (CUSUM with hysteresis)
│   └── detector.py        # FDDDetector composite, FDDOutput dataclass
└── utils.py               # innovation gating, smooth blending helpers
```

### 2.2 Component dependencies

```
UFTCController
   ├─ uses → uftc.inner.WrappedAAINDI
   │           └─ composes aa_indi.AAINDIAgent (unchanged)
   ├─ uses → uftc.middle.IADPMiddle
   │           └─ composes iadp.IADPAgent (unchanged)
   └─ uses → uftc.fdd.FDDDetector
               ├─ composes fdd.kalman_3step.NominalKalman
               └─ composes fdd.change_point.ChangePointDetector
```

Existing `aa_indi.AAINDIAgent` and `iadp.IADPAgent` are **not modified**. UFTC
wraps them. This keeps existing tests and downstream users intact.

### 2.3 Public API entry point

```python
from tensoraerospace.agent import UFTCController, UFTCConfig

ctl = UFTCController(
    n_state=3, n_control=4,
    nominal_F=None,            # optional warm-start; None → learn during warmup
    nominal_G=None,
    config=UFTCConfig(...),
)
u = ctl.predict(x_obs, ref, time_step=k)
info = ctl.learn(x_next, ref, time_step=k)
diag = ctl.diagnostics()       # FDD severity, RLS state, mode-switch state, ...
```

Subclasses `BaseRLModel`; `train`, `save`, `from_pretrained`, `publish_to_hub`
follow library conventions.

## 3. Components in detail

### 3.1 `NominalKalman` (`uftc/fdd/kalman_3step.py`)

Adaptive 3-step Kalman filter (Lu 2015 style) tuned to nominal dynamics. Inputs
`(x_meas, u_prev)`, outputs `KalmanStep(x_hat, nu, S, K)`.

```python
class NominalKalman:
    def __init__(self, F_nominal, G_nominal, Q, R, *,
                 alpha_Q=0.99, alpha_R=0.99): ...
    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> KalmanStep:
        # Step 1: state prediction x̂⁻ = F·x̂ + G·u_prev
        # Step 2: innovation ν = x_meas − x̂⁻; S = H·P·Hᵀ + R
        # Step 3: state correction x̂⁺ = x̂⁻ + K·ν, P⁺ = (I − K·H)·P⁻
        # Adaptive Q, R: Sage-Husa-style EMA on outer-products of ν and residual.
        return KalmanStep(...)
    def reset(self): ...
```

Used by `FDDDetector` only — not exposed publicly except via `uftc.fdd`.

### 3.2 `ChangePointDetector` (`uftc/fdd/change_point.py`)

CUSUM on Mahalanobis innovation distance. Under the nominal hypothesis,
`d_t = νᵀ S⁻¹ ν ~ χ²_n`, so `E[d_t] = n_dim` and `drift = n_dim` is the
canonical default.

```python
class ChangePointDetector:
    def __init__(self, *,
                 n_dim: int,
                 drift: float | None = None,    # default = n_dim
                 h_alarm: float = 20.0,
                 h_clear: float = 5.0,
                 cooldown_steps: int = 200): ...
    def update(self, d_t: float) -> ChangePointState:
        # CUSUM_t = max(0, CUSUM_{t-1} + d_t − drift)
        # Hysteresis: alarm rising-edge when CUSUM > h_alarm and not in cooldown
        #             clear when CUSUM < h_clear and time_since_alarm > cooldown
        return ChangePointState(cusum, alarm, severity, time_since_alarm)
    def reset(self): ...
```

`severity` is `min(cusum / h_alarm, 10.0)` — normalised so that 1.0 means
"just crossed", 10.0 means "way past threshold". Hysteresis between
`h_alarm` and `h_clear` plus a `cooldown_steps` window prevent chattering.

Default thresholds target false-alarm rate ≤ ~10⁻⁴ per hour at dt=0.01s with
default Kalman covariances. Verified empirically in `test_change_point.py`.

### 3.3 `FDDDetector` (`uftc/fdd/detector.py`)

Composite: orchestrates `NominalKalman` and `ChangePointDetector`, exposes
`FDDOutput` to the controller.

```python
@dataclass
class FDDOutput:
    fault_present: bool        # alarm currently active
    severity: float            # normalised CUSUM (≥0; 1.0 = at threshold)
    confidence: float          # 1 − exp(−severity); ∈ [0, 1)
    innovation_norm: float     # ‖ν_t‖ for diagnostics
    time_since_event: float    # seconds since last rising edge

class FDDDetector:
    def __init__(self, n_state, n_control, kalman: NominalKalman,
                 cpd: ChangePointDetector, *, dt: float): ...
    def step(self, x_meas, u_prev) -> FDDOutput: ...
    def warm_start(self, F_nominal, G_nominal): ...   # update Kalman F, G
    def reset(self): ...
```

`confidence` is the soft variant of `fault_present` — `IADPMiddle` uses it to
scale RLS-reset linearly rather than as a binary toggle, so partial-confidence
events still trigger gentle adaptation.

### 3.4 `SuperTwistingObserver` (`uftc/inner.py`)

Higher-order sliding-mode observer (walkthrough §5.2):

```
ṡ = −k₁·|s|^{1/2}·sign(s) + z
ż = −k₂·sign(s)
```

Where `s = ω̇_meas − ν_des − δ̂`. Output `δ̂` is an estimate of the
high-frequency unmodeled disturbance, fed back into INDI as
`u = u_{t-1} + Ĝ⁺(ν_des − ω̇_meas − δ̂)`.

```python
class SuperTwistingObserver:
    def __init__(self, n_axes, *, k1=3.0, k2=1.5, dt=0.01): ...
    def update(self, omega_dot_meas, nu_des) -> np.ndarray: ...
    def reset(self): ...
```

### 3.5 `ModeSwitcher` (`uftc/inner.py`)

Rate-INDI ↔ angle-INDI hysteretic switch (walkthrough §5.3). MVP doesn't yet
ship a separate angle-INDI law — the switch outputs a flag that
`WrappedAAINDI` reads to swap `Ĝ` allocation. In the F-16 reference, both
modes use the same `aa_indi` core but with different `Ĝ` warm-starts and
different effective control allocations.

```python
class ModeSwitcher:
    def __init__(self, alpha_threshold_deg=25.0, hysteresis_deg=5.0): ...
    def select(self, alpha_rad: float) -> Literal["rate", "angle"]: ...
    def reset(self): ...
```

### 3.6 `WrappedAAINDI` (`uftc/inner.py`)

Composes `aa_indi.AAINDIAgent` with `SuperTwistingObserver`, `ModeSwitcher`,
and bounded trust-region clipping (walkthrough §5.4).

**Interface alignment with AAINDIAgent.** `AAINDIAgent.predict(omega,
reference, ...)` takes `reference = ω_cmd` (commanded angular rate). It
computes the angular-acceleration command ν_des internally via its
second-order reference model. Therefore the L3 → L2 signal in UFTC is **ω_ref
(rate command)**, not the ν_des that walkthrough §2.3 names abstractly. The
walkthrough's "rate command" maps onto our ω_ref; ν_des stays internal to
AAINDIAgent. Sliding-mode disturbance estimate δ̂ is injected into the
INDI residual via a small additive correction on ω_meas before AAINDI runs:
`omega_corrected = omega_meas − δ̂ · dt`.

```python
class WrappedAAINDI:
    def __init__(self, base: AAINDIAgent, *,
                 sm_obs: SuperTwistingObserver,
                 mode_switch: ModeSwitcher,
                 trust_radius_nominal: float = 0.1,
                 trust_radius_fault: float = 0.5): ...

    def predict(self, omega_ref, omega_meas, *,
                alpha: float,
                u_blend_target: np.ndarray,
                fault_severity: float,
                time_step: int) -> np.ndarray:
        # 1. ω̇_meas comes from base.sensor_filter (LowPassDerivative inside
        #    AAINDIAgent). We borrow the cached ω̇ from base after first call,
        #    or compute one ourselves from successive omega_meas.
        # 2. δ̂ = sm_obs.update(omega_dot_meas, ν_des_estimated)
        #    (ν_des_estimated = ω̇ that AAINDI's ref-model would produce for ω_ref)
        # 3. mode = mode_switch.select(alpha)
        # 4. omega_corrected = omega_meas − δ̂ · dt   # SM disturbance compensation
        # 5. u_indi_raw = base.predict(omega_corrected, omega_ref, time_step)
        # 6. trust radius δ = lerp(trust_radius_nominal, trust_radius_fault,
        #                          clip(fault_severity, 0, 1))
        # 7. Δ = u_indi_raw − u_blend_target
        #    if ‖Δ‖ > δ:
        #        u_indi = u_blend_target + δ · Δ / ‖Δ‖
        #    else:
        #        u_indi = u_indi_raw
        return u_indi

    def learn(self, next_omega, omega_ref, time_step) -> dict: ...
    def reset(self): ...
```

The trust-region radius **expands** under fault (linear interpolation by
severity) — the L3 middle needs more freedom to re-adapt after a fault, and
clipping aggressively while the system is recovering hurts more than it
helps.

### 3.7 `IADPMiddle` (`uftc/middle.py`)

Composes `iadp.IADPAgent` with innovation-driven RLS reset.

```python
@dataclass
class RLSResetPolicy:
    cov_inflation: float = 100.0
    forgetting_drop: float = 0.9
    forgetting_recover_steps: int = 500

class IADPMiddle:
    def __init__(self, base: IADPAgent, reset_policy: RLSResetPolicy,
                 omega_indices: list[int] | None = None,
                 lookahead_dt: float = 0.05): ...

    def predict(self, x_obs, reference, time_step) -> tuple[np.ndarray, np.ndarray]:
        # 1. u_iadp = base.predict(x_obs, reference, time_step) — standard iADP step
        # 2. omega_ref = derive_omega_ref(x_obs, reference, time_step)
        #    — desired angular rate that drives state error to zero over
        #      lookahead_dt. For omega-states (when omega_indices is given):
        #         omega_ref = (reference[omega_indices] − x_obs[omega_indices])
        #                     / lookahead_dt
        #      bounded by configurable max-rate.
        #    — for non-aero plants without omega_indices: omega_ref = reference
        #      (passthrough, AAINDI tracks state directly as a rate target).
        return u_iadp, omega_ref

    def learn(self, next_x_obs, reference, time_step, *,
              fdd: FDDOutput) -> dict:
        # Detect rising edge on fdd.fault_present
        # On rising edge:
        #   self.base.rls.Phi += cov_inflation · I
        #   self.base.cfg.gamma_rls = forgetting_drop  # mutate config used by RLS
        #   self.base.rls.gamma_rls = forgetting_drop
        #   self._recover_countdown = forgetting_recover_steps
        # Linear recovery: γ_RLS interpolated back to 0.99
        # over recover_steps.
        return self.base.learn(next_x_obs, reference, time_step)

    def reset(self): ...
```

`derive_omega_ref` translates L3's state-tracking objective into a rate
command for L2. Two strategies:

1. **Aero plant (`omega_indices` set)** — angular states are a strict subset
   of `x_obs`. `omega_ref = (ref_for_omega − current_omega) / lookahead_dt`,
   clipped to a configurable max-rate envelope. `lookahead_dt` is a tuning
   knob analogous to AAINDI's `ref_wn` — sets how aggressively L2 chases the
   state error.

2. **Generic plant (`omega_indices=None`)** — `omega_ref = reference`
   passthrough. AAINDI then treats the full state-reference as a rate target.
   Coarser, but works for any plant including non-aero MVP smoke tests.

This design keeps IADPAgent's internals untouched. The mapping from
`u_iadp` to ν_des-equivalent (used by SM observer) is computed inside
`WrappedAAINDI` from `Ĝ_t` borrowed off `base.G`.

### 3.8 `UFTCController` (`uftc/controller.py`)

Top-level orchestrator subclassing `BaseRLModel`.

```python
@dataclass
class UFTCConfig:
    dt: float = 0.01
    fdd_update_every: int = 1     # FDD subsamples controller rate
    fdd_warmup_steps: int = 200   # collect nominal F̂, Ĝ before activating

    # Sub-component configs
    inner_cfg: AAINDIConfig = field(default_factory=AAINDIConfig)
    middle_cfg: IADPConfig = field(default_factory=IADPConfig)
    fdd_cfg: FDDConfig = field(default_factory=FDDConfig)
    rls_reset_policy: RLSResetPolicy = field(default_factory=RLSResetPolicy)

    # L2 trust-region and SM observer
    sm_obs_k1: float = 3.0
    sm_obs_k2: float = 1.5
    trust_radius_nominal: float = 0.1
    trust_radius_fault: float = 0.5
    alpha_threshold_deg: float = 25.0
    alpha_hysteresis_deg: float = 5.0

    # Hooks for state extraction (plant-agnostic)
    alpha_index: int | None = None       # x_obs[alpha_index] → α
    omega_indices: list[int] | None = None
    # When None, controller assumes generic state — α=0 (rate-INDI always)

    enable_outer: bool = False    # placeholder for L4 D-SAC (Phase 3)


class UFTCController(BaseRLModel):
    def __init__(self, n_state, n_control, *,
                 nominal_F=None, nominal_G=None,
                 config: UFTCConfig | None = None): ...

    def predict(self, x_obs, reference, time_step=0) -> np.ndarray: ...
    def learn(self, next_x_obs, reference, time_step=0) -> dict: ...
    def diagnostics(self) -> dict: ...
    def reset(self): ...
    def save(self, path=None) -> str: ...
    @classmethod
    def from_pretrained(cls, repo_name, ...) -> "UFTCController": ...
```

#### `predict()` flow (one tick)

1. `u_iadp, omega_ref = middle.predict(x_obs, reference, time_step)`
2. `omega_meas = x_obs[omega_indices]` if configured, else `x_obs`
3. `alpha = x_obs[alpha_index]` if configured else `0.0`
4. `u_indi = inner.predict(omega_ref, omega_meas, alpha=alpha,
   u_blend_target=u_iadp, fault_severity=last_fdd.severity,
   time_step=time_step)`
5. Cache `(u_iadp, omega_ref, u_indi)` for `learn()`.
6. Return `u_indi`.

#### `learn()` flow

1. `inner.learn(next_x_obs, ...)` — VFF-RLS update inside AAINDI.
2. If `step >= fdd_warmup_steps` and `step % fdd_update_every == 0`:
     `fdd_out = fdd.step(next_x_obs, last_u_indi)`
     `self._last_fdd = fdd_out`
   Otherwise during warm-up: accumulate nominal `F̂, Ĝ` from
   IADPAgent's RLS for FDD warm-start. At step `fdd_warmup_steps` exactly:
     `fdd.warm_start(middle.base.F[:n_state, :n_state], middle.base.G[:n_state, :])`
3. `middle.learn(next_x_obs, reference, time_step, fdd=self._last_fdd)`
4. Merge diagnostics from all three sub-components and return.

### 3.9 Save / load

Multi-component layout under one directory (mirrors `IADPAgent.save`):

```
<save_dir>/
├── config.json          # UFTCConfig + nested configs (numpy → list)
├── inner/               # AAINDIAgent.save() output
├── middle/              # IADPAgent.save() output
├── fdd/
│   ├── kalman.npz       # F_nominal, G_nominal, Q, R, P, x_hat
│   └── cpd.npz          # cusum, time_since_alarm, in_cooldown
└── controller_state.npz # _last_fdd snapshot, step counter, last_u, last_nu_des
```

`from_pretrained` reconstructs nested components in turn, then re-instantiates
`UFTCController` and rehydrates `controller_state.npz`.

## 4. Data flow (one control tick)

```
                        [reference r_t]
                              │
                              ▼
         ┌───────────────────────────────────────────────┐
         │  IADPMiddle (L3)                              │
         │   in:  x_obs, r_t, fdd_prev                    │
         │   out: u_iadp, ω_ref                           │
         └───────────────────┬───────────────────────────┘
                              │ u_iadp (as u_blend_target)
                              │ ω_ref  (rate command for L2)
                              ▼
         ┌───────────────────────────────────────────────┐
         │  WrappedAAINDI (L2)                           │
         │   1. δ̂ = SM-observer(ω̇_meas, ν_des_internal)  │
         │   2. mode = ModeSwitcher(α)                   │
         │   3. ω_corr = ω_meas − δ̂·dt                   │
         │   4. u_indi = AAINDI.predict(ω_corr, ω_ref)   │
         │   5. trust-region clip к u_iadp с радиусом    │
         │      lerp(r_nom, r_fault, severity)           │
         └───────────────────┬───────────────────────────┘
                              │ u_indi
                              ▼
                          ACTUATORS / env.step
                              │
                              ▼
                          x_{t+1} (sensor)
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
         ┌──────────┐                    ┌─────────────┐
         │ AAINDI   │ learn()            │ FDDDetector │ step(x_{t+1}, u_indi)
         │ VFF-RLS  │                    │ Kalman+CPD  │
         └──────────┘                    └──────┬──────┘
                                                │ FDDOutput
                                                ▼
         ┌───────────────────────────────────────────────┐
         │  IADPMiddle.learn()                            │
         │   if rising_edge(fault_present):               │
         │      Φ += κ·I; γ_RLS ← 0.9                     │
         │   IADPAgent.learn() — стандартный шаг          │
         └───────────────────────────────────────────────┘
```

### 4.1 Sample rates

For MVP, all layers tick at `config.dt`. `fdd_update_every` allows FDD to run
at a sub-multiple (e.g. dt=0.001 inner with fdd_update_every=20 → FDD at
50 Hz, matching walkthrough §2.2). Inner-loop sub-stepping inside `predict()`
is **not** done in MVP — env step rate dictates effective controller rate;
this matches the existing AAINDIAgent / IADPAgent pattern.

### 4.2 Init lifecycle

```
1. UFTCController(n_state, n_control, nominal_F=None, ...)
   creates WrappedAAINDI, IADPMiddle, FDDDetector with stubs.
2. First fdd_warmup_steps:
   - inner and middle work as usual
   - FDD inactive (fault_present=False always)
   - at the end of warmup: nominal_F/G copied from IADPAgent.{F, G},
     FDD activated.
3. Steady state: all three layers active.
```

If the user passes `nominal_F`/`nominal_G` at construction, `fdd_warmup_steps`
can be set to 0.

## 5. Error handling and edge cases

| Situation | Behaviour |
|---|---|
| FDD not warmed up (`step < fdd_warmup_steps`) | `FDDOutput.fault_present=False, severity=0`. RLS-reset never fires. Inner uses `nominal` trust-radius. |
| RLS divergence in L3 (`‖Φ‖ > threshold` or NaN in `F̃`/`G̃`) | Existing `IADPAgent.rls` regularises `Φ ← Φ + ε·I`. UFTC additionally: detects NaN, performs hard reset `Φ ← phi_init·I`, logs a warning, recover via warm-up. |
| `Ĝ_t` singular when computing ν_des | Fallback ν_des = 0 for that tick, log warning. Inner gets ν_des=0 and holds previous δ. |
| Innovation explosion in Kalman (`‖ν‖ > 5σ`) | Kalman state-update is gated (P, x_hat held); CPD still receives raw `d_t`. Often this is itself the fault signal — CPD will fire, RLS-reset will follow. |
| FDD rising-edge during cooldown | Ignored. Re-inflating Φ a second time within cooldown rarely helps. |
| Mode-switch chattering near α=25° | `alpha_hysteresis_deg=5°` built into `ModeSwitcher`. |
| Trust-region breached | Hard clip to `u_iadp ± δ`. Falling back to `u_iadp` directly would be safer at first glance but throws away aerodynamic damping that AAINDI provides. |
| `predict()` / `learn()` called out of order | Each sub-component caches its own `last_*` state — no hard asserts on step ordering, matching IADPAgent pattern. |
| `reset()` mid-episode | Roll-state cleared in each sub-component (matches IADPAgent.reset). Trained weights `Θ̃, P̃, Φ, F_nominal, G_nominal` are **preserved** — next episode starts with accumulated knowledge. |
| User-supplied `alpha_index=None` on a non-aero plant | `ModeSwitcher` always returns `"rate"`. Trust-region behaviour unchanged. |

## 6. Testing

`tests/agents/uftc/` — parallel to `tests/agent/aidi/`.

### 6.1 Unit tests

| File | What it covers |
|---|---|
| `test_kalman_3step.py` | Kalman tracks linear plant within tolerance; adaptive Q/R converges on Gaussian noise. |
| `test_change_point.py` | CUSUM stays below threshold on χ²_n nominal; rises after step shift in mean; false-alarm rate under target; hysteresis works. |
| `test_fdd_detector.py` | End-to-end: nominal flight → fault_present=False; synthetic step-fault inject → fault_present=True within expected detection latency. |
| `test_inner_super_twisting.py` | SM observer estimates step disturbance; mode-switch hysteresis; trust-region clip honors radius bounds. |
| `test_middle_rls_reset.py` | Rising-edge `fault_present` → Φ inflated, γ_RLS dropped; recovery countdown returns γ to 0.99. |
| `test_controller_init.py` | Warm-up activates FDD; passing `nominal_F/G` skips warm-up; reset preserves trained weights. |

### 6.2 Integration tests (F-16 reference plant)

| File | Scenario |
|---|---|
| `test_uftc_nominal_f16.py` | F-16 nonlinear angular, no damage. UFTC tracking-RMS within ~baseline IADPAgent (regression guard). |
| `test_uftc_elevator_jam.py` | F-16 + `ELEVATOR_JAM_NEUTRAL` at t=5s. Expect: CPD fires within < 0.5s; tracking-RMS recovers within < 5s. |
| `test_uftc_wing_strike.py` | F-16 + `WING_STRIKE_LEFT_TIP` at t=5s. Expect: CPD fires; aircraft does not diverge (`‖x‖` stays bounded). |
| `test_uftc_engine_flameout.py` | F-16 + `ENGINE_FLAMEOUT`. Boundary: gradual degradation — RLS catches without hard CPD trigger. |
| `test_uftc_save_load_round_trip.py` | save/load round-trip: identical `predict()` output on identical seed. |

### 6.3 Smoke test

| `test_uftc_smoke.py` | 1000 steps on mock-plant (n_state=3, n_control=2), random reference, no damage. No NaN/exception, `diagnostics()` returns populated fields. |

### 6.4 Coverage target

≥ 80 % line coverage on `tensoraerospace/agent/uftc/` per project convention.

## 7. Documentation deliverables

- Module-level docstring on `tensoraerospace/agent/uftc/__init__.py` linking
  the wiki walkthrough and naming this design as Phase 1 MVP.
- Class docstrings on every public symbol following the existing IADP/AIDI
  style (Args / Returns / Notes blocks).
- One example notebook: `example/reinforcement_learning/uftc/uftc_f16_damage_demo.ipynb`
  — F-16 + WING_STRIKE_LEFT_TIP at t=5s, plots tracking error, FDD severity,
  RLS state pre/post detection.

## 8. Out of scope (deferred phases)

| Phase | Scope |
|---|---|
| 2 | L1 HJ-Reachability shield: V_HJ pre-training pipeline + bank + Lipschitz cert (α-β-CROWN) + conformal margin + filter logic. Optional MMAE bank for known fault classes (anchor policies). |
| 3 | L4 D-SAC outer + risk gate β_t + FDD-conditioning. |
| 4 | Vector Lyapunov runtime monitor (5-component V_total, Hurwitz Metzler check, ε-monitor, 3-level ALARM). Bank pre-training tooling (Eureka-style auto-bank). |

## 9. Open decisions

None at draft time — all six brainstorming questions resolved in chat:
plant-agnostic, hardcoded class refs, AAINDIAgent as L2 base, no fault catalog
(open-world detection), single uftc/ package, bank-wrapper-style L3.

## 10. References

- `~/Documents/tensoraerospace/tensoraeropsace/wiki/synthesis/uftc-extended-walkthrough.md`
  — primary source (UFTC v2 walkthrough, 2026-05-02).
- `tensoraerospace/agent/aa_indi/model.py` — L2 inner core (AAINDIAgent).
- `tensoraerospace/agent/iadp/model.py` — L3 middle core (IADPAgent).
- `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py` — fault
  presets used in integration tests.
- Lu, P. et al. (2015), "Adaptive three-step Kalman filter for air-data
  sensor fault detection," AIAA JGCD — Kalman design reference.
- Sieberling, S. et al. (2010), "Robust flight control using INDI," AIAA
  JGCD — INDI baseline.
- Konatala, R. et al. (2024), "Flight Testing Reinforcement Learning based
  Online Adaptive Flight Control Laws on CS-25 Class Aircraft," AIAA
  SCITECH — iADP middle reference.
