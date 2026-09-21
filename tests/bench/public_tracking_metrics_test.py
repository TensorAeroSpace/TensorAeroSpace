"""Command-based metrics distinguish correct tracking from a constant offset."""

import numpy as np
import pytest

from tensoraerospace.benchmark import ControlBenchmark


@pytest.mark.parametrize("initial,final", [(0.0, 1.0), (4.0, 6.0), (4.0, 2.0)])
def test_step_metrics_use_command_amplitude_and_direction(initial, final):
    reference = np.r_[np.full(5, initial), np.full(40, final)]
    response = np.r_[
        np.full(5, initial), initial + (final - initial) * np.r_[0.5, 1.2, np.ones(38)]
    ]
    result = ControlBenchmark().benchmarking_step_response(
        reference, response, initial, 0.1
    )
    assert result["command_overshoot"] == pytest.approx(20)
    assert result["command_settling_time"] == pytest.approx(0.2)


def test_multichannel_tracking_excludes_start_and_uses_applied_interval_actions():
    output = np.array([[99, 99], [99, 99], [0.2, -0.3], [0.01, 0.02], [0.0, 0.0]])
    actions = np.array([[999], [1], [2], [4]])
    m = ControlBenchmark().tracking_metrics(
        0, output, 0.5, start=0.5, end=2, tolerance=0.05, actions=actions
    )
    assert m["iae"] == pytest.approx(0.265)
    assert m["combined_rmse"] == pytest.approx(np.sqrt(0.1305 / 3))
    assert m["recovery_time"] == 1.0
    assert m["control_variation"] == 3.0
    assert m["control_peak"] == 4.0
    np.testing.assert_allclose(m["final_error"], [0, 0])


def test_tracking_reports_reference_minus_output_and_no_false_recovery():
    m = ControlBenchmark().tracking_metrics(1, np.full(11, 0.8), 0.1, tolerance=0.05)
    assert m["recovery_time"] is None
    assert m["final_error"][0] == pytest.approx(0.2)
    assert m["mae"][0] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"start": 2},
        {"end": 5},
        {"tolerance": 0},
        {"actions": np.zeros((2, 1))},
    ],
)
def test_invalid_windows_or_actions_are_rejected(kwargs):
    args = dict(reference=0, output=np.zeros(11), dt=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        ControlBenchmark().tracking_metrics(**args)
