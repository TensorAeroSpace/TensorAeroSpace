import optuna

from tensoraerospace.optimization.base import HyperParamOptimizationOptuna


def test_optuna_basic_minimize():
    # Deterministic objective: minimize x^2
    def objective(trial: optuna.trial.Trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return x * x

    hpo = HyperParamOptimizationOptuna(direction="minimize")
    hpo.run_optimization(func=objective, n_trials=10)
    best = hpo.get_best_param()
    assert "x" in best
    assert abs(best["x"]) <= 0.5  # should be close to zero with a few trials
