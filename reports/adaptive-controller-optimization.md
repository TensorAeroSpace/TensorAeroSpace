> Historical low-level optimizer experiment. The callback-free API and current notebook are documented in [declarative-controller-tuning.md](declarative-controller-tuning.md). The flight condition and search protocol below differ from that newer experiment.

# Shared adaptive-controller optimization

## Implementation

- `ControlOptimizer` extends the existing Optuna backend, with seeded TPE,
  explicit Float/Int/Categorical spaces, known-parameter initialization,
  per-case constraints, mean/worst aggregation, median pruning between cases,
  target/patience/timeout stopping, persistence and held-out validation.
- `AgentClass.optimize(...)` is available on AA-INDI, AIDI, iADP, IHDP, IM-GDHP,
  ET-DHP, HDP, MPC and agents inheriting BaseRLModel. It reconstructs fresh
  agents from constructor settings rather than editing trained instances.
- Dotted parameter paths rebuild dataclasses, mappings and indexed sequences.
  Optimizer learning rates and network dimensions are applied at construction.
  Fresh resource factories preserve environment identity; evaluation callbacks
  own training, simulation and cleanup.
- `StepResponseMetric` delegates CPI to the native ControlBenchmark. It rejects
  incomplete/nonfinite trajectories, exposes command-relative settling and
  overshoot, pre-step hold error, and mean/maximum tail errors. Constraints must
  pass on every scenario/seed. No feasible candidate is an explicit search
  failure, not a fallback to a bad controller.
- The original `HyperParamOptimizationOptuna` API remains compatible and now
  forwards study/run options. Optional search backends load on demand.

## Measured nonlinear B737 experiment

Executed notebook:
`example/optimization/adaptive_controller_tuning.ipynb`.
Full code, plots and results are also in the EN/RU optimization tutorials.

AA-INDI controls pitch through the existing proportional outer loop, with
continuous sensor observation and moment identification. RK4, dt=0.02 s, total
40 s, a +1° pitch step at 20 s. The nominal derivative prior and physical servo
limits are identical for every candidate. No pretraining or adaptation freeze.
The simulator is deterministic; one seed avoids redundant identical runs.

Search: 16 trials, TPE seed=42, 5 startup trials, baseline queued first.
Ranges: rate feedback [1, 12] s^-1 and acceleration cutoff [2, 15] Hz, both log.
Limits: relative tail maximum error <=1%, command settling <=10 s,
command overshoot <=10%, pre-step maximum relative error <=1%.

CPI is computed after scaling the response by its step amplitude. It must not
be compared numerically with earlier unnormalized radian-CPI experiments.

| Metric | Baseline | Selected |
|---|---:|---:|
| Normalized CPI | 3.7522434359 | 3.3068872498 |
| Command-relative 5% settling [s after step] | 7.90 | 6.68 |
| Command-relative overshoot [%] | 0 | 0 |
| Absolute mean tail error [% of step] | 0.414582 | 0.3013 |
| Maximum tail error [% of step] | 0.434119 | 0.3050 |
| Maximum pre-step error [% of step] | 0.038605 | 0.0384 |

CPI improves by **11.87%**. Selected rate feedback is
1.3467532529516106 s^-1, cutoff 4.551830162586262 Hz. Other controller settings
and adaptation equations are fixed. This is an observed result for this
experiment, not a global optimality claim.

### Held-out cases, no parameter reselection

All start from a fresh controller and the healthy prior, with online adaptation:

| Case | CPI | Settling [s] | Tail maximum [% of step] | Same bounds |
|---|---:|---:|---:|---|
| -1° step | 3.293468 | 6.66 | 0.3022 | PASS |
| +1.5° step | 3.284867 | 6.68 | 0.2619 | PASS |
| 20% elevator effectiveness loss at 27 s | 3.412059 | 6.68 | 0.3042 | PASS |

The failure is applied only to the plant through the SDK's
`ElevatorEffectiveness`. Step tests do not establish behavior under arbitrary
waveforms, sensor noise or other flight conditions.

### Runtime

The same seeded 16-trial search took 170.16 s with the environment's default
BLAS threading and 32.03 s in the executed notebook with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` set before starting
its kernel. Both selected exactly the same settings. This is a local observation
on small matrices; background load and hardware affect the comparison. The SDK
does not change process-wide thread settings. Statistical pruning across cases
has a separate test showing that unnecessary later seed evaluations are skipped.
No universal speedup over random search is claimed.

## Unsuccessful iADP exploratory search

An independent nonlinear longitudinal F16 pitch-rate step search rejected all
50 candidates in its specified parameter ranges. This is a useful negative
validation: the optimizer refused to label a controller with unacceptable
remaining error as successful.

Protocol: dt=0.02 s, duration32 s, +0.5°/s q step at20 s, initial q perturbation
uniformly within +/-0.005°/s from seed0, continuous adaptation. Scalar observed
q, four physical longitudinal/servo states, nominal F/G and discounted Riccati
P initialization as in the existing iADP small-fault tutorial. Fixed gamma=.99
for the initial prior, policy window300, warmup40 updates, R based on the
nominal gain, command envelope10° and60°/s. Search ranges: gamma[.9,.999],
gamma_rls[.99,1], phi_init[1,1e4] logarithmic, policy_eval_every10..100 step10.
Constraints: tail maximum<=5%, command settling<=10 s, overshoot<=20%,
pre-step maximum<=5%. Seed1 would run only after seed0 passed.

The baseline had CPI16.7378 and relative tail maximum50.43%. The measured
candidates and rejections are in `adaptive-controller-iadp-search.txt`.
This search does not prove the existence or absence of a satisfactory tuning
outside its bounds/budget and is not evidence of an error in the paper's law.
The optimizer does not change controller equations to force feasibility.

## Verification

- 1498 existing agent tests passed after integration.
- 116 optimization/benchmark tests passed, including real constructors for all
  eight named controller families, native step metrics, storage/resume checks,
  RNG isolation, independent resources, rejected/failed trials and pruning.
- All 6 notebook code cells executed; reference, before/after plots and validation
  tables remain embedded. Formatting changes preserve each executed cell's AST.
- EN/RU tutorials contain the complete notebook code and images.
- MkDocs strict EN/RU build passed (50.75 s).
- Ruff, Black, mypy on the new optimization code, and `git diff --check` passed.

Sources for search behavior:
[Optuna median pruning](https://optuna.readthedocs.io/en/v3.6.2/reference/generated/optuna.pruners.MedianPruner.html),
[Optuna process workers](https://optuna.readthedocs.io/en/v3.6.2/tutorial/10_key_features/004_distributed.html).
