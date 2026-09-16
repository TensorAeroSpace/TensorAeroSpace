"""Numeric benchmarks must respect step direction, sampling time and observation horizon."""

import re

import numpy as np
import pytest

from tensoraerospace.benchmark import ControlBenchmark
from tensoraerospace.benchmark.function import (
    find_step_function,
    get_lower_upper_bound,
    overshoot,
    rise_time,
    settling_time,
)


def test_integral_metrics_are_in_seconds():
    reference = np.ones(10)
    response = np.zeros(10)
    metrics = ControlBenchmark().benchmarking_one_step(reference, response, 0.0, 0.1)
    assert metrics["iae"] == pytest.approx(1.0)
    assert metrics["ise"] == pytest.approx(1.0)
    assert metrics["itae"] == pytest.approx(0.45)
    assert metrics["performance_index"] == pytest.approx(0.58)


def test_never_settled_returns_none_instead_of_index_past_horizon():
    response = np.ones(20)
    response[-1] = 5
    assert settling_time(np.ones(20), response) is None
    metrics = ControlBenchmark().benchmarking_one_step(np.ones(20), response, 0, 0.1)
    assert metrics["settling_time"] is None


@pytest.mark.parametrize("metric", [overshoot, rise_time, settling_time])
def test_metrics_are_invariant_under_sign_reversal(metric):
    response = np.r_[0.0, 0.2, 0.6, 1.2, np.ones(36)]
    reference = np.ones_like(response)
    assert metric(-reference, -response) == pytest.approx(metric(reference, response))


@pytest.mark.parametrize("dtype", [int, float])
def test_negative_setpoint_tolerance_band_is_ordered_and_fractional(dtype):
    lower, upper = get_lower_upper_bound(np.array([0, -2, -2], dtype=dtype))
    np.testing.assert_allclose(lower, [-2.1] * 3)
    np.testing.assert_allclose(upper, [-1.9] * 3)


def test_downward_step_is_detected_at_crossing():
    reference = np.array([0, 0, -1, -1])
    response = np.array([0.0, 0.0, -0.2, -0.8])
    actual_reference, actual_response = find_step_function(reference, response, -0.5)
    np.testing.assert_array_equal(actual_reference, [-1, -1])
    np.testing.assert_array_equal(actual_response, [-0.2, -0.8])


def test_immediate_settling_is_not_reported_as_slow():
    report = ControlBenchmark().generate_report(np.ones(20), np.ones(20), 0.0, 0.1)
    assert re.search(r"Быстродействие:\s+Быстро\b", report)
