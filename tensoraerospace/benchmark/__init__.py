"""Module for evaluating aerospace system control quality.

This module provides tools for analyzing and evaluating the performance
of control algorithms, including quality metrics, benchmarks and comparative
analysis of various approaches to aircraft control.

Main components:
    - ControlBenchmark: Class for comprehensive control quality assessment
"""

from .bench import ControlBenchmark as ControlBenchmark


# Lazy exports avoid importing environments while the package registers them.
def __getattr__(name):
    if name == "B737PitchStepBenchmark":
        from .pitch import B737PitchStepBenchmark

        return B737PitchStepBenchmark
    if name == "B747EngineFailureBenchmark":
        from .engine_failure import B747EngineFailureBenchmark

        return B747EngineFailureBenchmark
    raise AttributeError(name)


__all__ = ["ControlBenchmark", "B737PitchStepBenchmark", "B747EngineFailureBenchmark"]
