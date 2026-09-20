# Controller search: parallel execution benchmark

## Result

For the measured iHDP/B737 experiment, four CPU processes reduced the mean runtime from **44.71 s to 14.92 s (3.00×)**. Timing includes process startup, all 24 candidate evaluations and the automatic full-horizon replay of the winner. Imports and initial tuner construction in the parent were outside the timer.

| Run | Serial, 1 process | Parallel, 4 processes |
|---|---:|---:|
| 1 | 44.665 s | 14.984 s |
| 2 | 44.761 s | 14.863 s |

The same 24 parameter dictionaries, trial states, per-seed metrics and rejection reasons matched exactly in all four searches. The selected pitch trajectory and all actuator commands also matched bit for bit on replay. Best CPI was **4.534721801** in each run; this measures execution equivalence, not an improvement to policy quality or convergence to zero.

[Raw timings, parameters and all candidate metrics](controller-search-speed.json).

## Experiment

- Native environment: `NonlinearB737-v0`, trim at 10,000 ft / 600 ft/s.
- Native controller: `ihdp`; the eight-parameter default search space.
- Step: pitch +1° at 20 s; dt=0.02 s; full horizon 40 s, 2,000 simulation updates.
- Sampler: `random`, seed 42, 24 trials; each controller simulation uses seed 0.
- Torch CPU threads: 1 per simulation; unchanged NumPy/BLAS settings.
- Machine reported 16 logical CPUs. Python 3.11.10, Optuna 3.6.2, Torch 2.11.0+cu130, NumPy 1.26.4.
- Serial/parallel runs alternated twice. No other test suite ran during the timing experiment.

Random search was used to compare exactly the same candidates. TPE and genetic search use batched feedback when parallelized, so their serial and parallel candidate sequences can differ. More processes consume more memory; startup overhead can dominate very short runs. This measured speedup is not a universal factor.

A separate baseline simulation with the original eight Torch threads and with one thread produced exactly equal pitch/action arrays on this experiment.

## SDK usage

```python
from tensoraerospace.optimization import ControllerTuner, Step

tuner = ControllerTuner(
    env="NonlinearB737-v0",
    env_kwargs={"trim_at": (10000.0, 600.0), "dt": 0.02},
    reference={"theta": Step(1.0, at=20.0, unit="deg")},
    controller="ihdp",
    method="tpe",
    duration=40.0,
    seed=42,
    torch_threads=1,
)
result = tuner.optimize(n_trials=120, n_jobs=4, target=3.7)
```

Use the cell directly in Jupyter. In a Python script, protect construction/search with `if __name__ == "__main__":`. Restart a running notebook kernel after updating the SDK. Do not use notebook-local functions or lambdas in parallel experiment settings; workers need importable, picklable objects.

## Execution contract

- TPE, genetic and random search support `n_jobs>1`; annealing requires `n_jobs=1` to preserve sequential acceptance/proposal decisions.
- Each candidate uses a fresh native controller and environment. Step timing, integration, training budgets, full horizon, metric normalization and physical constraints are unchanged.
- Workers reuse the serial objective evaluator. Constraint failures remain PRUNED with complete evaluated-case diagnostics; programming exceptions propagate.
- One parent samples candidates and updates Optuna in trial-number order. Fixed worker count, sampler seed and deterministic numerical backend give reproducible batches.
- Parallel TPE defaults to `constant_liar=True` to account for running candidates; `method_options` can override this.
- Warm-start parameters finish first. Target, patience and timeout stop new batches; the current batch finishes. At most `n_jobs - 1` additional candidates can finish after an early-stop condition.
- Torch thread count and RNG state in the calling process are restored after simulations, including failed simulations. Each worker owns its own state.
- The progress bar retains ETA/best-feasible reporting. Saved search results include execution settings and per-worker evaluation timing.

The scheduler uses [Optuna's public ask-and-tell batch interface](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html).
