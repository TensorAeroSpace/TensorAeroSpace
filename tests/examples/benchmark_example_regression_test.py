"""The published benchmark demo must generate the response it advertises."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "example/utilities/example_benchmark_usage.py"
)
spec = importlib.util.spec_from_file_location("benchmark_example", MODULE_PATH)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_zero_overshoot_produces_finite_monotonic_response():
    times = np.linspace(0, 6, 601)
    reference, response = example.generate_sample_system_response(
        times, overshoot=0.0, settling_time=2.0, noise_level=0.0
    )
    assert np.all(np.isfinite(response))
    assert np.all(np.diff(response) >= -1e-14)
    assert response.max() <= 1.0
    assert response[-1] > 0.999


def test_requested_settling_time_controls_the_decay_envelope():
    times = np.linspace(0, 10, 10001)
    _, response = example.generate_sample_system_response(
        times, overshoot=0.2, settling_time=2.0, noise_level=0.0
    )
    assert response.max() - 1.0 == pytest.approx(0.2, abs=1e-5)
    assert np.max(np.abs(response[times >= 3.2] - 1.0)) < 0.02


def test_example_preserves_fractional_response_on_integer_time_grid():
    times = np.arange(8)
    _, actual = example.generate_sample_system_response(times, noise_level=0.0)
    _, expected = example.generate_sample_system_response(
        times.astype(float), noise_level=0.0
    )
    np.testing.assert_allclose(actual, expected)
