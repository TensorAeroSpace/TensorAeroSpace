# Agent Hyperparameter Search

Automatic hyperparameter optimization tools for control algorithms. Supports Optuna and Ray Tune.

## Quick Start (Optuna)

```python
import numpy as np
from tensoraerospace.optimization import HyperParamOptimizationOptuna

def objective(trial):
    # Example: minimize the quadratic error of a model with hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    # ... train/evaluate your model ...
    score = np.random.rand()  # replace with the real quality metric
    return score

opt = HyperParamOptimizationOptuna(direction="minimize")
opt.run_optimization(objective, n_trials=20)

best = opt.get_best_param()
print("Best parameters:", best)
opt.plot_parms(figsize=(12, 4))
```

## Classes

=== "Optuna"

    ::: tensoraerospace.optimization.HyperParamOptimizationOptuna

=== "Ray Tune"

    ::: tensoraerospace.optimization.HyperParamOptimizationRay

## Notes

- `direction`: choose `"minimize"` or `"maximize"` for the target metric.
- For Ray Tune, use `run_optimization(func, param_space, tune_config=...)` and then post-process `self.results` (see the Ray Tune docs).
