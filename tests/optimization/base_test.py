import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import optuna.trial
import pytest

from tensoraerospace.optimization import base


class _DummyTrial:
    def __init__(self, value, params, state=None):
        self.value = value
        self.params = params
        self.state = state if state is not None else optuna.trial.TrialState.COMPLETE


class _DummyStudy:
    def __init__(self):
        self.trials = []
        self.optimize_called_with = None
        self.best_trial = type("BT", (), {"params": {"a": 1, "b": 2}})()

    def optimize(self, func, n_trials):
        self.optimize_called_with = (func, n_trials)
        # populate trials for plot
        self.trials = [
            _DummyTrial(1.0, {"p": 0.1}),
            _DummyTrial(0.5, {"p": 0.2}),
        ]


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        base.HyperParamOptimizationOptuna(direction="bad")


def test_run_optimization_and_best_param(monkeypatch):
    dummy = _DummyStudy()

    def fake_create_study(direction):
        return dummy

    monkeypatch.setattr(base.optuna, "create_study", fake_create_study)

    opt = base.HyperParamOptimizationOptuna(direction="minimize")
    assert opt.study is dummy

    def obj(trial):
        return 0.0

    opt.run_optimization(func=obj, n_trials=3)
    assert dummy.optimize_called_with == (obj, 3)
    assert opt.get_best_param() == {"a": 1, "b": 2}


def test_plot_parms_runs(monkeypatch):
    dummy = _DummyStudy()
    dummy.trials = [
        _DummyTrial(1.0, {"p": 0.1}),
        _DummyTrial(0.8, {"p": 0.2}),
    ]

    def fake_create_study(direction):
        return dummy

    monkeypatch.setattr(base.optuna, "create_study", fake_create_study)

    opt = base.HyperParamOptimizationOptuna(direction="maximize")
    # Use small figsize to keep plot light; ensure no crash
    opt.plot_parms(figsize=(2, 2))
    # Verify ticks match number of trials
    fig = plt.gcf()
    ax = fig.axes[0]
    assert len(ax.get_xticks()) == len(dummy.trials)


def test_plot_parms_skips_pruned_and_failed_trials(monkeypatch):
    """plot_parms should skip pruned/failed trials that have value=None."""
    dummy = _DummyStudy()
    dummy.trials = [
        _DummyTrial(1.0, {"p": 0.1}, state=optuna.trial.TrialState.COMPLETE),
        _DummyTrial(None, {"p": 0.3}, state=optuna.trial.TrialState.PRUNED),
        _DummyTrial(None, {}, state=optuna.trial.TrialState.FAIL),
        _DummyTrial(0.5, {"p": 0.2}, state=optuna.trial.TrialState.COMPLETE),
    ]

    def fake_create_study(direction):
        return dummy

    monkeypatch.setattr(base.optuna, "create_study", fake_create_study)

    opt = base.HyperParamOptimizationOptuna(direction="minimize")
    # Should not raise despite pruned/failed trials with None values
    opt.plot_parms(figsize=(2, 2))
    fig = plt.gcf()
    ax = fig.axes[0]
    # Only 2 COMPLETE trials should be plotted
    assert len(ax.get_xticks()) == 2
