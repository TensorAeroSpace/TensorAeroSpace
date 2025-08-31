import numpy as np

from tensoraerospace.benchmark.function import (
    damping_degree,
    find_step_function,
    rise_time,
    settling_time,
)


def test_find_step_function_no_indices_returns_original():
    c = np.zeros(5)
    s = np.arange(5)
    c2, s2 = find_step_function(c, s, signal_val=1.0)
    assert np.array_equal(c2, c)
    assert np.array_equal(s2, s)


def test_damping_degree_no_peaks_returns_zero():
    s = np.linspace(0, 1, 10)
    assert damping_degree(s) == 0.0


def test_rise_time_none_when_thresholds_not_reached():
    c = np.ones(10)
    s = np.zeros(10)
    assert rise_time(c, s, low_threshold=0.1, high_threshold=0.9) is None


def test_settling_time_full_length_when_never_within_range():
    c = np.ones(10)
    s = np.zeros(10)
    assert settling_time(c, s, threshold=0.01) == len(s)
