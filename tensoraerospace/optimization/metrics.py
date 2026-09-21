"""Step-response objectives using the native control benchmark."""

from __future__ import annotations

import numpy as np

from tensoraerospace.benchmark import ControlBenchmark


class TrialRejected(Exception):
    """A candidate left the declared operating envelope or did not finish."""


class StepResponseMetric:
    """CPI and accuracy metrics for a complete, sampled step response.

    ``reference`` and the evaluated output have shape (samples,) or
    (samples, channels), at identical timestamps, including the initial state.
    Every channel must have exactly one nonzero step after the initial sample.
    CPI is the benchmark's ``performance_index``; channel CPIs are averaged.
    ``normalize=True`` expresses each response relative to its step amplitude
    before computing CPI, allowing comparisons across units/amplitudes.
    Accuracy constraints still use original units, or dimensionless relative
    errors. A missing settling time remains None (an upper-bound constraint
    rejects it). Pre-step hold error is reported separately from post-step CPI.
    """

    def __init__(
        self, reference, dt, *, normalize=False, tolerance=0.05, tail_fraction=0.1
    ):
        reference = np.asarray(reference, dtype=float)
        if reference.ndim == 1:
            reference = reference[:, None]
        if (
            reference.ndim != 2
            or reference.shape[0] < 3
            or reference.shape[1] < 1
            or not np.isfinite(reference).all()
        ):
            raise ValueError("reference must be finite with shape (N,) or (N, C)")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(tail_fraction) or not 0 < tail_fraction <= 1:
            raise ValueError("tail_fraction must be in (0, 1]")
        self.reference = reference.copy()
        self.dt = float(dt)
        self.normalize = bool(normalize)
        self.tolerance = float(tolerance)
        self.tail_fraction = float(tail_fraction)
        self._starts = []
        for channel in reference.T:
            changes = np.flatnonzero(np.diff(channel) != 0) + 1
            if len(changes) != 1 or changes[0] >= len(channel) - 1:
                raise ValueError(
                    "Each channel needs one step and >= 2 post-step samples"
                )
            self._starts.append(int(changes[0]))

    def __call__(self, output) -> dict[str, float | None]:
        """Score a full trajectory; reject short/nonfinite simulations."""
        output = np.asarray(output, dtype=float)
        if output.ndim == 1:
            output = output[:, None]
        if output.shape != self.reference.shape or not np.isfinite(output).all():
            raise TrialRejected(
                "Expected a complete finite trajectory matching reference"
            )
        channels = []
        benchmark = ControlBenchmark()
        for c, start in enumerate(self._starts):
            ref, response = self.reference[:, c], output[:, c]
            amplitude = abs(float(ref[-1] - ref[0]))
            if self.normalize:
                r = (ref - ref[0]) / (ref[-1] - ref[0])
                y = (response - ref[0]) / (ref[-1] - ref[0])
            else:
                r, y = ref, response
            # Use the exact same native CPI definition as published benchmarks.
            metrics = benchmark.benchmarking_step_response(
                r, y, signal_val=float(r[0]), dt=self.dt, tolerance=self.tolerance
            )
            tail_count = max(1, int(np.ceil((len(ref) - start) * self.tail_fraction)))
            tail_error = ref[-tail_count:] - response[-tail_count:]
            metrics.update(
                tail_error=float(abs(np.mean(tail_error))),
                tail_max_error=float(np.max(abs(tail_error))),
                relative_tail_error=float(abs(np.mean(tail_error)) / amplitude),
                relative_tail_max_error=float(np.max(abs(tail_error)) / amplitude),
                pre_step_max_error=float(np.max(abs(ref[:start] - response[:start]))),
                pre_step_relative_error=float(
                    np.max(abs(ref[:start] - response[:start])) / amplitude
                ),
            )
            channels.append(metrics)
        result: dict[str, float | None] = {}
        for key in channels[0]:
            values = [m[key] for m in channels]
            if key == "performance_index":
                result[key] = float(np.mean(values))
            else:
                result[key] = (
                    None if any(v is None for v in values) else float(max(values))
                )
        result["cpi"] = result["performance_index"]
        if len(channels) > 1:
            for c, metrics in enumerate(channels):
                result.update({f"channel_{c}.{k}": v for k, v in metrics.items()})
        return result
