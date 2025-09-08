import matplotlib

matplotlib.use("Agg")

import optuna

from tensoraerospace.optimization.base import HyperParamOptimizationOptuna


def test_optuna_plot_parms_smoke():
    def objective(trial: optuna.trial.Trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return (x - 0.2) ** 2

    hpo = HyperParamOptimizationOptuna(direction="minimize")
    hpo.run_optimization(objective, n_trials=5)
    # Should not raise
    hpo.plot_parms(figsize=(5, 3))
