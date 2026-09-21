"""Long-horizon metrics must neither double count boundaries nor use future data."""

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_long_horizon_f16 as example,
)


def test_window_metrics_excludes_left_endpoint_and_reports_peak_time():
    trace = np.zeros((5, 8))
    trace[:, 0] = [100, 101, 102, 103, 104]
    trace[:, 1] = 1
    trace[:, 2] = [1000, 2, 3, 4, 1000]
    m = example.window_metrics(trace, 100, 103)
    assert m["samples"] == 3
    assert m["rmse_deg_s"] == pytest.approx(np.sqrt(14 / 3))
    assert m["peak_error_deg_s"] == 3
    assert m["peak_time_s"] == 103
    assert example.window_metrics(trace, 105, 110) is None


def test_rolling_rmse_is_causal_and_uses_full_windows_only():
    trace = np.zeros((5, 8))
    trace[:, 0] = np.arange(1, 6)
    trace[:, 2] = [1, -1, 1, -1, 10]
    time, error = example.rolling_rmse(trace, 3)
    np.testing.assert_array_equal(time, [3, 4, 5])
    np.testing.assert_allclose(error, [1, 1, np.sqrt(102 / 3)])
