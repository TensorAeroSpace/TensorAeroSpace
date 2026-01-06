import types

import pytest

from tensoraerospace.optimization import ray as ray_opt


class _DummyResult:
    def __init__(self, config=None, params=None, **kwargs):
        self.config = config
        self.params = params or {}


class _DummyResultsGrid:
    def __init__(self, best=None, iterable=None):
        self._best = best
        self._iterable = iterable or []

    def get_best_result(self, metric=None, mode=None):
        if self._best is None:
            raise RuntimeError("no best")
        return self._best

    def __iter__(self):
        return iter(self._iterable)


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        ray_opt.HyperParamOptimizationRay(direction="wrong")


def test_run_optimization_uses_defaults(monkeypatch):
    calls = {}

    class DummyTuner:
        def __init__(self, func, param_space, tune_config, **kwargs):
            calls["init"] = (func, param_space, tune_config, kwargs)

        def fit(self):
            calls["fit"] = True
            return _DummyResultsGrid(best=_DummyResult(config={"x": 1}))

    def fake_tuner(func, param_space, tune_config=None, **kwargs):
        return DummyTuner(func, param_space, tune_config, **kwargs)

    class DummyTuneConfig:
        def __init__(self, num_samples):
            self.num_samples = num_samples

    monkeypatch.setattr(ray_opt.tune, "Tuner", fake_tuner)
    monkeypatch.setattr(ray_opt.tune, "TuneConfig", DummyTuneConfig)

    opt = ray_opt.HyperParamOptimizationRay(direction="minimize", metric="m")

    def obj(cfg):
        return 0

    opt.run_optimization(func=obj, param_space={"lr": 0.1})
    assert "fit" in calls
    # Default num_samples=5 should be used
    assert isinstance(calls["init"][2], DummyTuneConfig)
    assert calls["init"][2].num_samples == 5
    assert opt.get_best_param() == {"x": 1}


def test_get_best_param_fallbacks(monkeypatch):
    # Case 1: get_best_result raises -> fallback to iter
    best = _DummyResult(config={"a": 10})
    grid = _DummyResultsGrid(best=None, iterable=[best])

    opt = ray_opt.HyperParamOptimizationRay(direction="max")
    opt.results = grid
    assert opt.get_best_param() == {"a": 10}

    # Case 2: no config, but params dict
    res_with_params = _DummyResult(config=None)
    res_with_params.params = {"p": 3}
    grid2 = _DummyResultsGrid(best=res_with_params)
    opt.results = grid2
    assert opt.get_best_param() == {"p": 3}

    # Case 3: not run -> raises
    opt.results = None
    with pytest.raises(RuntimeError):
        opt.get_best_param()


def test_plot_parms_not_implemented():
    opt = ray_opt.HyperParamOptimizationRay(direction="min")
    with pytest.raises(NotImplementedError):
        opt.plot_parms()


import pytest

from tensoraerospace.optimization.ray import HyperParamOptimizationRay


class _DummyResult:
    def __init__(self, config):
        self.config = config


class _DummyResultGrid:
    def __init__(self, results):
        self._results = results

    def __iter__(self):
        return iter(self._results)

    def get_best_result(self, metric=None, mode=None):  # noqa: ARG002
        # Always return first for predictability
        return self._results[0]


class _DummyTuner:
    def __init__(
        self, func, param_space=None, tune_config=None, **kwargs
    ):  # noqa: ARG002
        self.func = func
        self.param_space = param_space
        self.tune_config = tune_config
        self.kwargs = kwargs

    def fit(self):
        return _DummyResultGrid([_DummyResult({"x": 1}), _DummyResult({"x": 2})])


def test_ray_hpo_direction_validation():
    with pytest.raises(ValueError):
        HyperParamOptimizationRay(direction="bad")


def test_ray_hpo_run_and_get_best_param(monkeypatch):
    import tensoraerospace.optimization.ray as raymod

    monkeypatch.setattr(raymod.tune, "Tuner", _DummyTuner)

    hpo = HyperParamOptimizationRay(direction="minimize")
    hpo.run_optimization(func=lambda cfg: cfg, param_space={"x": [1, 2]})
    best = hpo.get_best_param()
    assert best == {"x": 1}


def test_ray_hpo_get_best_param_requires_run():
    hpo = HyperParamOptimizationRay(direction="maximize")
    with pytest.raises(RuntimeError, match="run_optimization"):
        hpo.get_best_param()
