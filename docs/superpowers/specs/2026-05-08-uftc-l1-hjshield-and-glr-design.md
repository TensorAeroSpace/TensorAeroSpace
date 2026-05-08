# UFTC Phase 2 — L1 HJ-Reachability shield + GLR-extended FDD

**Date:** 2026-05-08
**Status:** Draft, pending implementation plan
**Master spec:** `2026-05-08-uftc-cascade-extension-design.md`
**Predecessor:** `2026-05-07-uftc-phase1-mvp-design.md`

## 1. Scope

Two additions to the Phase 1 UFTC stack:

1. **L1 HJ-Reachability shield** — post-filter on `u_indi` that solves a small QP using a learned Hamilton-Jacobi value function `V_θ(x)` and a conformal margin `ε_t = ε_t(FDDOutput)`. The shield enforces forward-invariance of a configurable safe set under the current FDD state.
2. **GLR-extended FDD** — generalised likelihood-ratio test for slow-drift (gradual) faults, running in parallel with the existing CUSUM channel on the same Kalman innovations. The composite `FDDOutput` gains `fault_kind ∈ {none, abrupt, gradual, compound}` and `severity_gradual`.

Both are gated behind `UFTCConfig.enable_l1_shield` and `UFTCConfig.enable_glr` — default `False`, regression invariance to Phase 1 preserved.

## 2. Package layout

```
tensoraerospace/agent/uftc/
├── fdd/
│   ├── glr.py                # NEW — GLRDetector
│   └── detector.py           # extended FDDOutput
└── l1/                       # NEW
    ├── __init__.py
    ├── value_fn.py           # HJValueFunction protocol + DeepReachValueFn
    ├── deepreach_train.py    # off-line PINN training of V_θ
    ├── shield.py             # HJReachabilityShield (post-filter, QP)
    ├── conformal.py          # ConformalMargin
    ├── lipschitz.py          # power-iteration upper bound on Lipschitz of V_θ
    ├── bank.py               # ValueBank{nominal, fault classes}
    └── README.md             # pre-training and bank-construction guide
```

## 3. HJ value function

### 3.1 Interface

```python
class HJValueFunction(Protocol):
    """V(x) ≥ 0 outside safe set; V(x) ≤ 0 inside; V = 0 on boundary."""
    def value(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def lipschitz_const(self) -> float: ...

@dataclass
class DeepReachConfig:
    n_state: int
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: Literal["tanh", "sine"] = "tanh"
    state_bounds: np.ndarray | None = None        # shape (n_state, 2)
    time_horizon: float = 5.0
    safe_set_fn_name: str = "alpha_envelope"      # name of registered ℓ(x)
    dt: float = 0.01

class DeepReachValueFn(HJValueFunction):
    def __init__(self, model: nn.Module, cfg: DeepReachConfig): ...
    @classmethod
    def load(cls, path: str | Path) -> "DeepReachValueFn": ...
    def save(self, path: str | Path) -> None: ...
```

`safe_set_fn_name` registers callable `ℓ(x)` (signed-distance to boundary of safe set). For F-16 longitudinal: `ℓ_envelope(x) = min(α_max - α, α - α_min, q_max - |q|, V_max - V, V - V_min)`. For full 6-DoF: `ℓ_full = min(ℓ_envelope, ℓ_omega_bounds, ℓ_load_factor)`.

### 3.2 DeepReach training

`deepreach_train.py` follows Bansal & Tomlin (2021):

```
L_total = L_HJI + λ_bdy · L_bdy + λ_smooth · L_smooth

L_HJI(x, t) = ( ∂V/∂t  +  min_u  max_d  ⟨∇V, f̂(x, u, d)⟩ )²        # PINN-residual
L_bdy(x)    = ( V(x, T) − ℓ(x) )²                                    # terminal condition
L_smooth(x) = α · ‖Hess V(x)‖²_F                                     # smoothness regulariser
```

`f̂` is the dynamics callable. For Phase 2 we register two: `f_f16_nominal` (full F-16 angular) and `f_linearised_warm_start` (F̂, Ĝ from `IADPMiddle.base.rls` after warm-up). Bank entries are tagged by which `f̂` they were trained against.

Curriculum: 200 epochs total — first 50 epochs `λ_smooth = 0`, last 150 with `λ_smooth = 0.01` to enforce gentle gradients before final fine-tune.

Sampling:
- 50 % uniform in `state_bounds`;
- 30 % rejection-sample near `ℓ(x) = 0` boundary (where the value function changes most);
- 20 % from a pre-recorded F-16 nominal-tracking roll-out (in-distribution).

Training script CLI:

```bash
python -m tensoraerospace.agent.uftc.l1.deepreach_train \
    --plant f16-nonlinear-angular \
    --mode nominal \
    --epochs 200 --batch 4096 \
    --out artifacts/v_hj/nominal/
```

Output: `value_fn.pt` (state-dict), `value_fn.json` (cfg + training metadata + on-grid validation metrics).

### 3.3 Lipschitz constant

`lipschitz.py` provides power-iteration upper bound on `‖∇²V‖∞ → L(∇V)`:

```python
def power_iteration_lipschitz(model: nn.Module, sample_fn: Callable[[], np.ndarray],
                              n_iter: int = 200) -> float:
    # Iterate v_{k+1} ∝ J(x_k) · v_k where J is the Jacobian matrix
    # Use vmap + jacrev to batch.
```

Returned `L` is stored in `value_fn.json`. If `α-β-CROWN` is installed, `lipschitz.alpha_beta_crown_certified(...)` provides a tighter cert, but it's not required for Phase 2.

## 4. Conformal margin

```python
@dataclass
class ConformalMarginConfig:
    eps_0: float = 0.05                # baseline margin from holdout calibration
    k_grad: float = 0.10
    k_abrupt: float = 0.20
    k_innov: float = 0.05
    k_alarm: float = 0.30

class ConformalMargin:
    def __init__(self, cfg: ConformalMarginConfig, lipschitz_const: float): ...
    def compute(self, fdd: FDDOutput, monitor_alarm: str = "OK") -> float:
        eps = (
            self.cfg.eps_0
            + self.cfg.k_grad * fdd.severity_gradual
            + self.cfg.k_abrupt * fdd.severity_abrupt
            + self.cfg.k_innov * fdd.innovation_norm
            + self.cfg.k_alarm * (0.0 if monitor_alarm == "OK"
                                  else 0.5 if monitor_alarm == "WARN"
                                  else 1.0)
        )
        return float(eps * self.lipschitz_const)   # rescale by Lipschitz
```

`eps_0` calibrated empirically: collect 5000 nominal states, target false-block rate ≤ 5 % when `u_nominal = u_indi` is forced through the shield. Calibration script `example/uftc/calibrate_conformal.py`.

## 5. Shield post-filter

```python
@dataclass
class HJShieldConfig:
    h_clear: float = 0.20           # passthrough if V > h_clear (deep inside safe)
    qp_solver: Literal["mosek", "osqp"] = "osqp"
    cbf_lambda: float = 1.0         # forward-invariance gain
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None
    bank: ValueBankConfig | None = None
    conformal: ConformalMarginConfig = field(default_factory=ConformalMarginConfig)

@dataclass
class ShieldOutput:
    u_safe: np.ndarray
    intervention_norm: float
    hjb_value: float
    active: bool

class HJReachabilityShield:
    def __init__(self, n_state, n_control, *,
                 value_fn: HJValueFunction | ValueBank,
                 dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
                 cfg: HJShieldConfig): ...
    # If dynamics_fn is None at filter() time, the shield borrows F̃, G̃ from
    # the controller's IADPMiddle.base.rls and constructs an affine-in-u
    # surrogate `f̂(x, u) = F̃ x + G̃ u`. Both modes share the QP path.

    def filter(self, x, u_nominal, fdd, monitor_alarm="OK") -> ShieldOutput:
        V_x = self.value_fn.value(x, fdd)
        eps_t = self.conformal.compute(fdd, monitor_alarm)
        h_safe = V_x - eps_t

        if h_safe > self.cfg.h_clear:
            return ShieldOutput(u_nominal, 0.0, V_x, active=False)

        grad_V = self.value_fn.gradient(x, fdd)

        # QP:  min ‖u − u_nominal‖² s.t. ⟨∇V, f̂(x,u)⟩ + λ V ≥ ε,  u_min ≤ u ≤ u_max
        u_safe = self._solve_qp(x, u_nominal, grad_V, V_x, eps_t)
        return ShieldOutput(
            u_safe=u_safe,
            intervention_norm=float(np.linalg.norm(u_safe - u_nominal)),
            hjb_value=float(V_x),
            active=True,
        )

    def request_actuator_hold(self) -> None:
        # macro-action sink: next filter() returns u_safe = self._last_u_safe
        self._hold_one_tick = True

    def reset(self): ...
```

QP details:

- Affine-in-control assumption: `f̂(x, u) = f₀(x) + g(x)·u` (true for INDI-style models). With `f₀ = F̃·x`, `g = G̃` from `IADPMiddle.base.rls`, the constraint becomes linear in `u`.
- `cvxpy` with `OSQP` default; `Mosek` if installed (faster, license required).
- Decision time: ~0.5 ms on CPU for 6-DoF / 4 inputs at default solver tolerance.
- Solver failure → return `u_nominal` and log warning (`active=False`, `intervention_norm=0`); does not crash the controller.

## 6. Value bank

```python
@dataclass
class ValueBankConfig:
    nominal_path: Path
    fault_paths: dict[str, Path]            # {"elevator_jam_neutral": ..., ...}
    fallback: Literal["nominal", "min", "interp"] = "min"
    abrupt_lookup_threshold: float = 0.7    # MMAE prob threshold

class ValueBank(HJValueFunction):
    """Picks per-mode V_θ^(h) based on FDDOutput."""
    def value(self, x, fdd: FDDOutput) -> float:
        if fdd.fault_kind == "none":
            return self._vs["nominal"].value(x)
        if (fdd.fault_kind == "abrupt"
                and fdd.mmae_probs is not None
                and fdd.mmae_probs.max() > self.cfg.abrupt_lookup_threshold):
            h = self._modes[fdd.mmae_probs.argmax()]
            return self._vs[h].value(x)
        # open-world fallback — worst-case shielding
        if self.cfg.fallback == "min":
            return min(v.value(x) for v in self._vs.values())
        ...
```

For Phase 2 MVP, `fdd.mmae_probs` defaults to `None` and the fallback path runs unconditionally. MMAE classification is a Phase 5 augmentation.

## 7. GLR detector

### 7.1 Mathematical core

For Kalman innovations `ν_t ~ N(0, S_t)` under nominal hypothesis, suppose a possible drift `μ ≠ 0` starts at unknown change-time `τ ≤ t`. The two-sided GLR test statistic over a sliding window of length `W`:

```
T_t  =  max_{t-W ≤ τ ≤ t-1}   sup_{μ}   2 · log  L(ν_τ:t | μ) / L(ν_τ:t | 0)
     =  max_{t-W ≤ τ ≤ t-1}   ‖ Σ_{i=τ}^{t} S_i^{−1} ν_i ‖²_{(Σ_{i=τ}^{t} S_i^{−1})^{−1}}
```

Closed form for constant `S = S̄`:

```
T_t  =  max_{t-W ≤ τ ≤ t-1}  (1 / (t - τ + 1)) · ‖ Σ_{i=τ}^{t} ν_i ‖²_{S̄^{-1}}
```

For time-varying `S_i`, we keep a running sum `Σ S_i⁻¹ ν_i` and `Σ S_i⁻¹` per window position and recompute incrementally.

### 7.2 Class

```python
@dataclass
class GLRConfig:
    window: int = 200
    h_alarm: float = 30.0
    h_clear: float = 8.0
    cooldown_steps: int = 200
    mu_min_norm: float = 0.05    # below this, drift estimate is treated as "no drift"

@dataclass
class GLRState:
    statistic: float
    alarm: bool
    severity: float            # statistic / h_alarm, clipped to [0, 10]
    drift_estimate: np.ndarray
    time_since_alarm: int

class GLRDetector:
    def __init__(self, n_dim: int, cfg: GLRConfig): ...
    def update(self, nu: np.ndarray, S: np.ndarray) -> GLRState: ...
    def reset(self): ...
```

### 7.3 FDDDetector composition

`detector.py` is updated:

```python
@dataclass
class FDDOutput:
    fault_present: bool
    fault_kind: Literal["none", "abrupt", "gradual", "compound"]
    severity_abrupt: float
    severity_gradual: float
    severity: float                      # = max(abrupt, gradual)  (Phase 1 compat)
    confidence: float
    innovation_norm: float
    time_since_event: float
    glr_drift_estimate: np.ndarray | None = None
    mmae_probs: np.ndarray | None = None  # Phase 5 — currently None

class FDDDetector:
    def __init__(self, ..., glr: GLRDetector | None = None): ...
    def step(self, x_meas, u_prev) -> FDDOutput:
        kalman_step = self.kalman.step(x_meas, u_prev)
        cpd_state = self.cpd.update(kalman_step.d_t)
        glr_state = self.glr.update(kalman_step.nu, kalman_step.S) if self.glr else None

        # Aggregate
        abrupt = cpd_state.alarm
        gradual = bool(glr_state and glr_state.alarm)
        kind = ("compound" if abrupt and gradual
                else "abrupt" if abrupt
                else "gradual" if gradual
                else "none")
        sev_a = cpd_state.severity
        sev_g = float(glr_state.severity) if glr_state else 0.0
        return FDDOutput(
            fault_present=(abrupt or gradual),
            fault_kind=kind,
            severity_abrupt=sev_a,
            severity_gradual=sev_g,
            severity=max(sev_a, sev_g),
            confidence=1.0 - math.exp(-(sev_a + sev_g)),
            innovation_norm=float(np.linalg.norm(kalman_step.nu)),
            time_since_event=cpd_state.time_since_alarm * self.dt,
            glr_drift_estimate=(glr_state.drift_estimate if glr_state and glr_state.alarm else None),
        )
```

## 8. UFTCController integration

In `predict()`:

```python
u_indi = self.inner.predict(...)
if self.cfg.enable_l1_shield:
    u_safe = self.l1.filter(x_obs, u_indi, self._last_fdd, self._monitor_alarm).u_safe
else:
    u_safe = u_indi
self._last_u_safe = u_safe
return u_safe
```

In `learn()`:

```python
fdd_out = self.fdd.step(next_x, self._last_u_safe)   # uses u_safe, not u_indi
```

`save()` adds `l1/value_fn.pt`, `l1/value_fn.json`, `l1/conformal.json`, `fdd/glr.npz`.

## 9. Tests

| File | Coverage |
|---|---|
| `tests/agents/uftc/l1/test_value_fn.py` | DeepReach converges on toy double-integrator (analytical V* known); Lipschitz upper-bound holds on grid sample. |
| `tests/agents/uftc/l1/test_deepreach_train_smoke.py` | 5-epoch training run completes, loss decreases monotonically. |
| `tests/agents/uftc/l1/test_conformal.py` | `eps_t` monotone in each severity component; `eps_t = eps_0` when fault is none. |
| `tests/agents/uftc/l1/test_shield_qp.py` | passthrough when V > h_clear; modifies u minimally in L²-norm under boundary; `u_safe ∈ [u_min, u_max]`; QP-solver failure → fall-back to u_nominal without crash. |
| `tests/agents/uftc/l1/test_shield_bank.py` | `ValueBank` worst-case fallback under ambiguous FDD; abrupt+high-prob mode → corresponding V_θ. |
| `tests/agents/uftc/l1/test_request_actuator_hold.py` | macro-action call freezes `u_safe` for one tick. |
| `tests/agents/uftc/fdd/test_glr.py` | nominal χ²: T_t below h_alarm with target ARL₀; pure ramp drift → alarm within expected ARL₁; hysteresis between h_alarm and h_clear. |
| `tests/agents/uftc/fdd/test_fdd_extended.py` | FDDOutput.fault_kind ∈ {none,abrupt,gradual,compound} all reachable; severity == max(severities); compound when both fire same tick. |
| `tests/agents/uftc/test_uftc_l1_engine_drift.py` | F-16 + new ENGINE_THRUST_DRIFT preset (1 % thrust loss/s): GLR alarms ≤ 3 s after onset; L1 shield does not block ω-tracking; L3 RLS adapts. |
| `tests/agents/uftc/test_uftc_l1_phase1_invariance.py` | `enable_l1_shield=False, enable_glr=False` → predict()/learn() bit-identical to Phase 1 over 1000-step seed-fixed F-16 roll-out. |

Coverage target: ≥ 80 % line coverage on `tensoraerospace/agent/uftc/l1/` and `tensoraerospace/agent/uftc/fdd/glr.py`.

## 10. Pre-training tooling

```bash
# Train a single mode
python -m tensoraerospace.agent.uftc.l1.deepreach_train \
    --plant f16-nonlinear-angular --mode nominal --epochs 200 \
    --out artifacts/v_hj/nominal/

# Build a bank (loops over presets and trains with linearised dynamics
# matching each preset's effective F̂, Ĝ)
python example/uftc/build_v_hj_bank.py \
    --presets nominal,elevator_jam_neutral,wing_strike_left_tip,engine_flameout \
    --out artifacts/v_hj_bank/

# Calibrate conformal eps_0 from holdout nominal trajectories
python example/uftc/calibrate_conformal.py \
    --bank artifacts/v_hj_bank/ \
    --rollouts artifacts/nominal_rollouts.npz \
    --target-false-block-rate 0.05
```

`README.md` in `l1/` walks through the full pipeline including expected runtime (~30 min/mode on a single A100; ~3 hr/mode on CPU) and known failure modes (boundary-set leakage when `state_bounds` too tight).

## 11. Out of scope for Phase 2

- α-β-CROWN formal Lipschitz certificate (deferred to Phase 2.1; power-iteration upper bound is sufficient for MVP).
- MMAE bank classification of fault classes (deferred to Phase 5).
- Online retraining of `V_θ` while flying (V_θ is treated as offline-trained artifact).
- 3D-vis / dashboard for shield interventions (separate visualization work).

## 12. References

- Bansal, S., Chen, M., Herbert, S., Tomlin, C. (2017) "Hamilton-Jacobi reachability: A brief overview and recent advances," CDC.
- Bansal, S. & Tomlin, C. (2021) "DeepReach: A Deep Learning Approach to High-Dimensional Reachability," ICRA.
- Wabersich, K. & Zeilinger, M. (2018) "Linear MPC-based predictive safety filter for arbitrary feedback controllers," CDC — conformal margin pattern.
- Basseville, M. & Nikiforov, I. (1993) Detection of Abrupt Changes — GLR baseline.
- Willsky, A. (1976) "A survey of design methods for failure detection in dynamic systems," Automatica.
- Master spec: `2026-05-08-uftc-cascade-extension-design.md`.
