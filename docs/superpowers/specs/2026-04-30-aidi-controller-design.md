# Adaptive Incremental Dynamic Inversion (AIDI) controller — design

**Date:** 2026-04-30
**Status:** Draft, pending implementation plan

## 1. Background

This spec covers the addition of a new model-agnostic flight-control agent —
**Adaptive Incremental Dynamic Inversion (AIDI)** — to TensorAeroSpace. The
algorithm follows:

> Ul Haq, R. S., Atmaca, D., and van Kampen, E.,
> *"Adaptive Incremental Dynamic Inversion for Fault-tolerant Flight Control of a
> Flying Wing"*, AIAA SciTech 2026 Forum, Article AIAA 2026-1744,
> [10.2514/6.2026-1744](https://doi.org/10.2514/6.2026-1744).

The paper applies an INDI inner loop with online adaptation of the
**control-effectiveness (CE) matrix** through a Recursive-Least-Squares
identifier with a variable forgetting factor (VFF-RLS), wrapped by a C\*
longitudinal / roll-rate / sideslip-compensator outer loop. Adaptation is
**multiplicative scaling** over a known onboard CE model (`G̃ = Θ ⊙ G_nominal`)
rather than full structural identification — this generalises across plants and
keeps the implementation compact.

The repo already contains an `aa_indi` agent based on Sun et al. (TU Delft,
structural identification of `G`). AIDI is a **separate** agent — different
identification model, different VFF formulation, different outer-loop
architecture, and a different fault-tolerance angle.

### 1.1 Aircraft scope

First integration target: **F-16 nonlinear angular** model
(`tensoraerospace.envs.f16.nonlinear_angular.NonlinearAngularF16`).

The F-16 has only three rate-producing surfaces (`stab`, `ail`, `dir`), so
`G_nominal` is a square 3×3 matrix and the redundant-allocation aspects of the
paper (Moore-Penrose pseudoinverse, cross-axis consistency on a 3×5 layout) do
**not** show their full benefit on this plant. The agent is implemented as
**model-agnostic** — driven by an `OnboardCEModel` Protocol — so a future
Flying-V model would slot in without changes to the agent.

### 1.2 Out of scope

- Adding a Flying-V aerodynamic model (separate, much larger sub-project).
- MOPS (Multi-Objective Parameter Synthesis) automated tuning of outer-loop
  gains. We provide reasonable defaults and document the tunable parameters;
  a MOPS routine can be a follow-up.
- Replacing the `aa_indi` agent. AIDI is additive.

## 2. High-level architecture

The agent decomposes into well-bounded units that mirror the paper's block
diagram (Fig. 3). Each unit is independently testable and replaceable.

```
tensoraerospace/agent/aidi/
├── __init__.py             # Re-exports
├── model.py                # AIDIAgent + AIDIConfig (orchestrator + persistence)
├── scaling_rls.py          # ScalingRLS estimator
├── onboard_ce.py           # OnboardCEModel Protocol + F16NonlinearOnboardCE adapter
├── allocator.py            # MoorePenroseAllocator
├── pch.py                  # PseudoControlHedge
└── ref_models.py           # CStarController, RollReferenceModel,
                            # SideslipCompensator, SpeedController, LinearController
```

Tests under `tests/agent/aidi/` mirror this layout one-to-one.

### 2.1 Per-tick signal flow

1. **Outer loop.** The user supplies a reference dict
   `{"C_star": ..., "phi": ..., "beta": ..., "V": ...}`. The four
   reference blocks plus the linear controller produce the virtual-control
   vector `ν = (ω̇_x_des, ω̇_y_des, ω̇_z_des)`. PCH subtracts the hedging
   signal from each reference model before integration.
2. **Inner loop (AIDI law).**
   - `ω̇_meas` from a low-pass differentiator of the measured rates (we reuse
     `aa_indi.LowPassDerivative`).
   - Scaled CE: `G̃ = Θ ⊙ G_nominal(x, u₀)` where `Θ` is the live scaling matrix
     held by `ScalingRLS` (initialised to all-ones).
   - Increment: `Δu = G̃⁺ · (ν − ω̇_meas)`. Total command:
     `u = u₀ + Δu`, with rate and magnitude clamps applied.
3. **After `env.step(u)` (called from `learn`).**
   - Read new `ω̇_meas` (causal differentiator step).
   - Per-row VFF-RLS update over `Θ`, with the information-content forgetting
     factor of Eq. 26–27 of the paper.
   - Cross-axis consistency check (paper §III.C): per surface column `j`
     compute the mean update across rows; replace any row update that deviates
     by more than `1e-6` with the column mean.

## 3. Algorithm details

### 3.1 ScalingRLS (`scaling_rls.py`)

Holds:

- `theta` ∈ `(n_y, n_u)` — multiplicative scaling parameters, init `1`.
- `P_i` ∈ `(n_u, n_u)` for each row `i` — independent covariance per axis.
- `lam_i` for each row.

For each step `(Δu, Δω̇, G_nominal)` and each row `i ∈ [0, n_y)`:

1. Build regressor `φ_i = G_nominal[i, :] ⊙ Δu` (Hadamard product).
2. Compute residual `ε_i = Δω̇_i − Θ[i, :] · φ_i`.
3. Gain: `K_i = P_i · φ_i / (λ_i + φ_i^T · P_i · φ_i)`.
4. Tentative update: `Δθ_i = K_i · ε_i`.
5. Covariance: `P_i ← (P_i − K_i · φ_i^T · P_i) / λ_i`, then symmetrise
   (`P_i ← 0.5·(P_i + P_i.T)`).
6. **Information-content VFF (Eq. 26–27 of the paper):**
   - `λ_i = 1 − (1 − φ_i^T · K_i) · ε_i² / Σ₀`, with `Σ₀ = σ₀² · N₀`.
   - Clamp `λ_i ∈ [λ_min, λ_max]`.

After running all rows, **cross-axis consistency check** (paper §III.C):

```
for j in range(n_u):
    delta_bar_j = mean_i Δθ[i, j]
    for i in range(n_y):
        if abs(Δθ[i, j] − delta_bar_j) > 1e-6:
            Δθ[i, j] = delta_bar_j
```

Then commit `Θ ← Θ + Δθ`. The check coincides with the paper's stated rule
that updates per row should be proportionally similar; otherwise the mean is
applied.

Configuration parameters (`AIDIConfig` proxies these):

| name | default | meaning |
|------|---------|---------|
| `vff_lambda_min` | 0.7 | lower bound on forgetting factor (fast adaptation) |
| `vff_lambda_max` | 0.999 | upper bound (noise rejection) |
| `vff_sigma0` | 1e-3 | sensor noise variance σ₀² |
| `vff_memory_length` | 100 | nominal memory length N₀ in samples |
| `rls_cov_init` | 1.0 | initial scale of P_i (small — we trust onboard) |
| `consistency_threshold` | 1e-6 | per-paper threshold for the cross-axis check |

### 3.2 OnboardCEModel (`onboard_ce.py`)

```python
class OnboardCEModel(Protocol):
    n_state: int
    n_control: int
    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...
        # returns G ∈ (n_state, n_control)
```

Concrete implementation `F16NonlinearOnboardCE`:
- Numerically computes `G = ∂ω̇/∂u` by central finite differences around
  `(x, u)` over the existing F-16 nonlinear ODE
  (`tensoraerospace.aerospacemodel.f16.nonlinear.angular.AngularF16`),
  with a configurable perturbation `ε_u` (default `1e-3` rad).
- Result is cached per call (immutable per tick) so the agent's
  `predict`/`learn` does not pay for repeated FD calls.

A second concrete `LinearOnboardCE(B_lookup)` is added for use in tests and as
a fallback for any plant exposing an `(A, B)` matrix.

### 3.3 MoorePenroseAllocator (`allocator.py`)

Thin wrapper around `np.linalg.pinv(G, rcond=cfg.pinv_rcond)`. Returns the
control increment `Δu = G̃⁺ · (ν − ω̇_meas)`. Falls back to a zero increment
(and logs a warning) when `cond(G̃) > cfg.cond_threshold` — keeps the agent
quiet during the RLS warm-up.

### 3.4 PseudoControlHedge (`pch.py`)

State: `nu_des_prev` (per-axis virtual control from the previous tick).

```
nu_h = nu_des_prev − omega_dot_meas
```

Subtracted from the reference-model derivatives before integration:

```
r_dot_ref ← r_dot_ref − nu_h
```

This freezes the reference model whenever the inner loop fails to deliver the
demanded acceleration (actuator saturation, severe fault) and prevents
integrator wind-up of the reference. When saturation persists for more than
`pch_freeze_steps` ticks the reference rate is hard-frozen until the gap
closes.

### 3.5 Outer-loop blocks (`ref_models.py`)

| block | input | output | math |
|-------|-------|--------|------|
| `CStarController` | `C*_cmd, n_z, q, V` | `q_des` | `C* = n_z + (V/V_co)·q`; PI on error to `q_des`. `V_co = 122.6 m/s` (MIL-STD). |
| `RollReferenceModel` | `phi_cmd, phi, p` | `p_des` | 2nd-order: `p_dot = −2ζω_n p + ω_n² (φ_cmd − φ)`. |
| `SideslipCompensator` | `beta_cmd, beta` | `r_des` | PI on `(β_cmd − β)`. |
| `SpeedController` | `V_cmd, V` | `δ_throttle` | PID. No-op when env exposes constant airspeed. |
| `LinearController` | `(q_des, p_des, r_des)` | `ν` | Identity in the rate basis used by the inner loop, plus a configurable rate-feedback gain on `(ω_des − ω_meas)`. |

All gains live in `AIDIConfig` with named groups (`cstar_kp`, `cstar_ki`,
`roll_wn`, `roll_zeta`, etc.) and have defaults tuned for sub-sonic
fixed-wing dynamics.

### 3.6 AIDIAgent orchestrator (`model.py`)

API parity with existing online agents (`aa_indi`, `et_dhp`, `im_gdhp`):

```python
class AIDIAgent:
    def __init__(
        self,
        n_state: int,
        n_control: int,
        onboard_ce: OnboardCEModel,
        config: AIDIConfig | None = None,
    ) -> None: ...
    def reset(self) -> None: ...
    def predict(
        self,
        observation: dict[str, np.ndarray | float],
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray: ...   # shape (n_control,)
    def learn(
        self,
        next_observation: dict[str, np.ndarray | float],
        time_step: int = 0,
    ) -> dict[str, float]: ...  # diagnostic metrics
    def save(self, path: str | Path | None = None) -> str: ...
    @classmethod
    def from_pretrained(
        cls, repo_name: str, access_token: str | None = None,
        version: str | None = None,
    ) -> "AIDIAgent": ...
    def publish_to_hub(self, repo_name: str, folder_path: str | Path,
                       access_token: str | None = None) -> None: ...
```

`observation` is a dict with required keys `omega`, `alpha`, `beta`, `theta`,
`phi`, `V`. Optional `n_z` — if absent, reconstructed from
`(α, α̇, q, V, θ, φ)` using the standard body-frame load-factor identity
`n_z ≈ (V/g)·(q·cos α − α̇) + cos θ · cos φ` (small-β approximation). The
reconstruction lives in a small helper `aidi.utils.reconstruct_n_z`; a
downstream consumer with a true accelerometer just supplies `n_z` directly
and bypasses the helper. The exact form (which terms are kept) is
finalised in the implementation plan against the F-16 ODE.

References pattern (matches existing agents):

```python
agent.predict(obs, time_step=t)        # returns u shape (n_control,)
env.step(u)
metrics = agent.learn(next_obs, time_step=t)
```

`predict` does not advance the differentiator (consistent with `aa_indi`);
`learn` does. The cached `ω̇_meas` from the last `learn` is what `predict`
uses for the inner loop residual. A reset via `reset()` clears differentiator,
ref-model, integrator, and PCH state — but not `Θ` or `P` (lifelong adaptation).

`save` writes a self-contained directory:

```
{date}_AIDIAgent/
├── config.json          # AIDIConfig + n_state, n_control, onboard_ce class path
├── scaling_rls.npz      # theta, P stack, lambdas, num_updates
├── outer_state.npz      # CStar PI integrator, sideslip PI integrator, speed PID, etc.
├── pch_state.npz        # nu_des_prev, freeze counter
├── ref_state.npz        # roll-ref model state, etc.
├── deriv_state.npz      # LowPassDerivative state
└── loop_state.npz       # u_prev, omega_prev, step counter, omega_dot_cached
```

`from_pretrained` accepts a local path or a HuggingFace `namespace/repo` id
and reconstructs the agent bit-identically.

## 4. Fault injection

We do **not** add a new fault type — the existing F-16 damage subsystem
already supports the exact failure model used by the paper:
`failure.mode == "efficiency_loss"` multiplies the commanded surface
deflection by `failure.efficiency` (see
`tensoraerospace/aerospacemodel/f16/nonlinear/damage/controls.py`).

We **add**:

- One or two CE-loss **presets** in
  `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`
  reproducing the paper's incremental fault schedule
  `[1, 0.75, 0.5, 0.25, 0]` on the inboard elevon, and similarly for the
  outboard elevon.
- Helpers in `aidi/_ce_loss_presets.py` that wire those presets to a
  `DamageProfile` for use in the example notebook and benchmark CLI.

Outside F-16 the agent works without any fault wrapper at all — the user
supplies their own env-side fault mechanism if any.

## 5. Error handling

| failure mode | response |
|--------------|----------|
| `G̃` ill-conditioned (`cond > cfg.cond_threshold`) | allocator falls back to `Δu = 0`, logs warning at most once per tick. |
| RLS divergence | `λ` clamped to `[λ_min, λ_max]`; `P` symmetrised; optional reset of `P` triggered when `‖ε‖ > eps_max` for `> persist_steps` ticks (off by default — opt-in). |
| Outer-loop saturation | PCH freezes the offending reference rate; freeze auto-clears when the inner loop reaches the demanded acceleration. |
| Missing observation key | `KeyError` with a list of required keys. |
| `n_z` reconstruction fallback | When neither `n_z` nor `α̇` is available, drops the cosine term and uses the small-angle approximation; logs a warning at agent construction. |
| `save` / `from_pretrained` round-trip | Covered by a unit test that asserts identical post-step output before and after reload. |

## 6. Testing strategy

### 6.1 Unit tests (`tests/agent/aidi/`)

- `test_scaling_rls.py`
  - converges to `Θ = I` on synthetic `Δω̇ = G·Δu` with no model error.
  - tracks step `Θ → 0.25` on a single channel after 50 steps with VFF active.
  - cross-axis consistency rejects rogue rows (one channel injected with noise
    twice the magnitude of the others).
  - VFF drops below `λ_max` after a step fault and recovers within
    `O(N_0)` steps.
- `test_onboard_ce_f16.py` — `F16NonlinearOnboardCE` matches a reference
  finite-difference computed at higher precision (h/h2 Richardson) within
  `1e-4`.
- `test_allocator.py` — Moore-Penrose returns the minimum-norm solution;
  tested on 3×3 (square) and 3×5 (redundant) cases; ill-conditioned fallback
  logs and returns zero.
- `test_ref_models.py` — C\* tracker, roll ref, sideslip PI behave correctly
  on synthetic inputs; speed controller is a no-op when constant-airspeed
  flag is set.
- `test_pch.py` — reference rate frozen during forced saturation; auto-clears.
- `test_aidi_agent.py` — full step exercises the API; `save`/`from_pretrained`
  round-trip; nominal F-16 tracking holds RMSE within a documented bound.

### 6.2 Integration test

Marker `@pytest.mark.integration`, target run-time ~30 s.

- F-16 nonlinear angular env, doublet command on pitch axis.
- At `t = 5 s` inject `efficiency_loss` on `stab_left` with `μ = 0.25`.
- Assertions:
  - AIDI RMSE-on-rate stays within `1.2× nominal` over the 60-s episode.
  - AIDI beats a non-adaptive INDI baseline by at least 15% on RMSE-on-rate.

## 7. Example notebook + benchmark CLI

### 7.1 `example/aidi_f16_fault_recovery.ipynb`

Reproduces paper-style figures on F-16:

- Reference vs achieved rates (`p`, `q`, `r`).
- Control surface deflections over time.
- Heat-map of `Θ(t)` showing scaling parameter evolution before/after fault.
- Side-by-side INDI (no adapt) vs AIDI (adapt) under the same scenario.
- 3D fly-out via the existing `flight_3d_viewer.html` to visualise the recovery.

### 7.2 `tensoraerospace/scripts/benchmark_aidi.py`

CLI invocation:

```
python -m tensoraerospace.scripts.benchmark_aidi \
    --env f16_nonlinear_angular \
    --baselines indi_no_adapt,aa_indi \
    --scenarios nominal,stab_50,stab_25,stab_lost,rudder_lost \
    --episodes 5 \
    --out report.md
```

Produces:

- `report.md` — Markdown table in the style of Table 8 of the paper:
  RMSE per `(method, scenario, axis)`.
- `report.csv` — same data, tidy long form.

The CLI is a thin orchestrator: it instantiates the env with each scenario's
damage profile, runs each agent, computes RMSE, and emits the report. No new
test methodology; reuses the integration-test infrastructure under the hood.

## 8. Documentation

- Module docstrings on every new file (we follow the style of `aa_indi` —
  see existing `model.py` / `vff_rls.py` / `sensor_filter.py`).
- New mkdocs page `docs/algorithms/aidi.md` covering:
  - Architecture diagram (text-based ASCII or a small SVG).
  - Per-tick signal flow.
  - Math (scaling-RLS, info-content VFF, consistency check, PCH, C\*).
  - Reference to Ul Haq et al. paper (DOI).
  - Worked F-16 fault-recovery example.
- One-line entry in `tensoraerospace/agent/__init__.py`:
  `from .aidi.model import AIDIAgent as AIDIAgent`.

## 9. Build sequence (PR-sized chunks)

1. `scaling_rls.py` + `tests/agent/aidi/test_scaling_rls.py`.
2. `allocator.py`, `ref_models.py`, `pch.py` + per-block tests.
3. `onboard_ce.py` (Protocol + F-16 adapter, `LinearOnboardCE`) + tests.
4. `model.py` (`AIDIAgent`, `AIDIConfig`, save/load) + integration test.
5. CE-loss damage presets + `example/aidi_f16_fault_recovery.ipynb`.
6. CLI benchmark + report emitter.
7. mkdocs page + `__init__.py` re-export.

Each chunk lands as a self-contained PR; later chunks may also add small
helpers (e.g. `aidi.utils.reconstruct_n_z`) when first needed.

## 10. Open questions for the implementation plan stage

- Whether the F-16 nonlinear ODE exposes `α̇` directly, or we need a small
  second differentiator just for `n_z` reconstruction. (Defer to plan; affects
  test for `aidi.utils.reconstruct_n_z`.)
- Exact tolerance for `test_onboard_ce_f16.py` against the analytic derivatives
  (the model exposes most aero terms as look-up tables; FD against itself at
  h₁/h₂ Richardson is the right baseline).
- Whether the integration-test assertion (`≥15%` improvement) needs softening
  for CI flakiness — start strict, relax if needed.
