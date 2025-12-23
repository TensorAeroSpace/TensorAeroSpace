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
