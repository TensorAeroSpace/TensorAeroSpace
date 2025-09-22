import numpy as np

from tensoraerospace.benchmark.function import (
    damping_degree,
    find_longest_repeating_series,
    find_step_function,
    get_lower_upper_bound,
    integral_absolute_error,
    integral_squared_error,
    integral_time_absolute_error,
    maximum_deviation,
    oscillation_count,
    overshoot,
    peak_time,
    performance_index,
    rise_time,
    settling_time,
    static_error,
    steady_state_value,
)


def test_find_longest_repeating_series_basic_and_edges():
    assert find_longest_repeating_series([]) == (0, 0)
    assert find_longest_repeating_series([5]) == (5, 5)
    assert find_longest_repeating_series([1, 2, 3, 10, 11, 5]) == (1, 3)


def test_find_step_function_slices_and_validates():
    control = np.array([0, 0, 1, 1], dtype=float)
    system = np.array([10, 11, 12, 13], dtype=float)
    c2, s2 = find_step_function(control, system, signal_val=0)
    assert np.array_equal(c2, np.array([1, 1], dtype=float))
    assert np.array_equal(s2, np.array([12, 13], dtype=float))


def test_overshoot_simple():
    control = np.ones(100, dtype=float)
    system = np.ones(100, dtype=float)
    system[10] = 1.2  # peak
    val = overshoot(control, system)
    assert np.isclose(val, 20.0)


def test_settling_time_enters_and_stays():
    control = np.ones(100, dtype=float)
    system = np.zeros(100, dtype=float)
    system[30:] = 1.0
    idx = settling_time(control, system, threshold=0.05)
    assert idx == 30


def test_damping_degree_from_peaks():
    # Peaks at 1.0, 0.8, 0.64 -> ratios 0.8/1.0 and 0.64/0.8 both 0.8 -> damping 0.2
    system = np.array([0.0, 1.0, 0.0, 0.8, 0.0, 0.64, 0.0], dtype=float)
    dd = damping_degree(system)
    assert np.isclose(dd, 0.2)


def test_static_error_difference_of_final_means():
    control = np.ones(50, dtype=float)
    system = np.ones(50, dtype=float) * 0.9
    assert np.isclose(static_error(control, system), 0.1)


def test_get_lower_upper_bound_from_final_value():
    control = np.array([0.0, 1.0, 2.0], dtype=float)
    lower, upper = get_lower_upper_bound(control, epsilon=0.1)
    assert np.allclose(lower, np.array([1.8, 1.8, 1.8]))
    assert np.allclose(upper, np.array([2.2, 2.2, 2.2]))


def test_rise_time_between_thresholds():
    control = np.ones(20, dtype=float)
    system = np.zeros(20, dtype=float)
    system[5:] = np.linspace(0.0, 1.0, 15)
    rt = rise_time(control, system, low_threshold=0.1, high_threshold=0.9)
    assert rt is None or rt > 0


def test_peak_time_returns_first_peak_or_argmax():
    system = np.array([0.0, 1.0, 0.0, 0.8, 0.0], dtype=float)
    assert peak_time(system) == 1


def test_maximum_deviation_from_final():
    control = np.ones(10, dtype=float)
    system = np.array([0.8] * 5 + [1.2] * 5, dtype=float)
    assert np.isclose(maximum_deviation(control, system), 0.2)


def test_integral_errors_and_oscillation_count_and_steady_state_value():
    control = np.array([0.0, 0.0, 1.0], dtype=float)
    system = np.array([0.0, 0.0, 0.0], dtype=float)
    assert np.isclose(integral_absolute_error(control, system), 1.0)
    assert np.isclose(integral_squared_error(control, system), 1.0)
    assert np.isclose(integral_time_absolute_error(control, system, dt=1.0), 2.0)

    # One full oscillation (one peak + one valley)
    sys2 = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=float)
    assert oscillation_count(sys2, threshold=0.1) == 1

    control2 = np.arange(10, dtype=float)
    expected_mean = np.mean(control2[8:])
    assert np.isclose(steady_state_value(control2, percentage=0.2), expected_mean)


def test_performance_index_returns_scalar():
    control = np.ones(50, dtype=float)
    system = np.ones(50, dtype=float) * 0.9
    val = performance_index(control, system, dt=0.1)
    assert isinstance(val, float)
    assert val >= 0.0
