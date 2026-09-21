"""Controller tuning, native step metrics and optional Optuna/Ray backends.

Public exports are loaded on demand, so importing agent optimization methods
neither initializes Ray nor imports the search/plotting stack.
"""

from importlib import import_module

_EXPORTS = {
    "ControllerTuner": (".experiment", "ControllerTuner"),
    "Step": (".experiment", "Step"),
    "ControlRun": (".experiment", "ControlRun"),
    "TuningResult": (".experiment", "TuningResult"),
    "AnnealingSampler": (".samplers", "AnnealingSampler"),
    "HyperParamOptimizationOptuna": (".base", "HyperParamOptimizationOptuna"),
    "HyperParamOptimizationRay": (".ray", "HyperParamOptimizationRay"),
    "ControlOptimizer": (".control", "ControlOptimizer"),
    "OptimizationResult": (".control", "OptimizationResult"),
    "StepResponseMetric": (".metrics", "StepResponseMetric"),
    "TrialRejected": (".metrics", "TrialRejected"),
    "Float": ("optuna.distributions", "FloatDistribution"),
    "Int": ("optuna.distributions", "IntDistribution"),
    "Categorical": ("optuna.distributions", "CategoricalDistribution"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


# Preserve wildcard imports without making Ray a mandatory dependency.
__all__ = [name for name in _EXPORTS if name != "HyperParamOptimizationRay"]
try:
    from importlib.util import find_spec

    if find_spec("ray") is not None:
        __all__.append("HyperParamOptimizationRay")
except (ImportError, ValueError):
    pass
