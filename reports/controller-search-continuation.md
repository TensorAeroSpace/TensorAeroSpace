# Continuing controller parameter search

## API

```python
# Start once; choose method="tpe" on the tuner for parallel search.
result = tuner.optimize(n_trials=120, n_jobs=8)

# Add a budget to the same study and sampler. The eight workers are retained.
result = tuner.resume(n_trials=120)
# Also available: result = result.resume(n_trials=120)

# Save once the search has stopped.
tuner.save_checkpoint("ihdp-search.pkl")
```

After restarting the kernel:

```python
from tensoraerospace.optimization import ControllerTuner

tuner = ControllerTuner.load_checkpoint("ihdp-search.pkl")
result = tuner.resume(n_trials=200)
```

`n_trials` is additional. A supplied `target` considers the best feasible result
from the whole history; an already reached target starts no new candidates.
Timeout and patience restart for each call. The progress bar counts the new
budget and shows the existing best immediately. Every continuation returns an
updated selected result and automatically replays it at the complete horizon.

History, sampler RNG, TPE state, genetic population/history and annealing's
accepted point/temperature persist. Changing worker count is allowed; changing
the experiment or sampler settings requires a new search. Batch size changes
can change future adaptive proposals. `optimize()` intentionally starts a new
study, while `resume()` continues the active one.

The checkpoint uses pickle and is intended for trusted files in the same
compatible software environment. It includes experiment settings and all study
trials, and loads without simulations. Writes are atomic. The JSON report from
`result.save(...)` remains distinct from a resumable checkpoint. This feature
continues **hyperparameter search**, not the neural weights of a selected policy.
Fresh controllers still receive the same per-candidate online learning budget.

## Full physical experiment

Native iHDP with `NonlinearB737-v0`, trim 10,000 ft / 600 ft/s, dt=0.02 s,
40 s duration, pitch step +1° at 20 s. TPE seed 42, controller seed 0, four
processes, default eight-dimensional search space. No user metric ceilings
were applied in this equivalence experiment; native finite-state, physical
envelope, actuator and complete-horizon checks remained active.

| Stage | Trials in history | Best CPI |
|---|---:|---:|
| Initial search | 8 | 9.273017643 |
| Restored checkpoint + 16 additional trials | 24 | 5.776007631 |
| Independent uninterrupted search | 24 | 5.776007631 |

The restored and uninterrupted searches matched exactly in sampled parameters,
trial states and per-seed metrics. The winner's pitch output and all actuator
commands matched bit for bit; both replays reached 40 s. This validates retained
search state and equivalent simulation, not a general convergence guarantee.

The selected controller still had a mean tail error of
5.79% of the step and did not
settle within the command's ±5% band. CPI=3.7 and a 2% tail-error requirement
were **not achieved** by this 24-trial example.

[Exact parameters and metrics](controller-search-continuation.json).

## Verification

- Unit/integration checks compare uninterrupted, in-memory continued and
  checkpoint-restored searches for TPE, genetic, random and annealing methods.
- Continuation preserves the original sampler object; annealing cools further
  rather than resetting its temperature.
- Other checks cover worker-count changes, rejected-candidate histories,
  unchanged hard constraints, nested array/configuration comparisons, changed
  reference channel order, stale result objects, reached targets, running-trial
  guards, and preserving an existing file when serialization fails.
- Two independent Jupyter kernels saved four trials, then restored and added
  four more with the same two-process execution setting and a working progress
  bar. The resulting best CPI was 1.072873613 for that short API check; its
  0.3 s horizon is not a control-quality benchmark.
