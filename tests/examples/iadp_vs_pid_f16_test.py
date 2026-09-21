"""PID is tuned on healthy data; the paired fault comparison cannot retune it."""

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_vs_pid_f16 as example,
)


@pytest.fixture(scope="module")
def comparison():
    cfg = example.baseline.Experiment(duration=6.0, fault_time=2.0)
    return cfg, *example.run_comparison(cfg)


def test_both_algorithms_see_the_same_reference_and_fault(comparison):
    cfg, traces, report = comparison
    np.testing.assert_array_equal(
        traces["iadp_fault"][:, :2], traces["pid_fault"][:, :2]
    )
    for algorithm in ("iadp", "pid"):
        np.testing.assert_allclose(
            traces[f"{algorithm}_fault"][: cfg.fault_step],
            traces[f"{algorithm}_healthy"][: cfg.fault_step],
            atol=1e-10,
            rtol=0,
        )
        assert report["metrics"][f"{algorithm}_healthy"]["events"] == []
        assert (
            report["metrics"][f"{algorithm}_fault"]["events"][0]["time"]
            == cfg.fault_time
        )
    assert (
        report["metrics"]["iadp_fault"]["events"]
        == report["metrics"]["pid_fault"]["events"]
    )


def test_pid_gains_are_fixed_but_integral_and_feedback_keep_working(comparison):
    cfg, traces, report = comparison
    metrics = report["metrics"]["pid_fault"]
    assert (
        metrics["gains_before"]
        == metrics["gains_after"]
        == list(example.DEFAULT_PID_GAINS)
    )
    assert abs(metrics["final_integral"]) > 1e-6
    assert np.ptp(traces["pid_fault"][cfg.fault_step :, 6]) > 0.1
    assert metrics["after_fault"]["rmse_deg_s"] > 0.0


def test_pid_obeys_same_increment_and_magnitude_limits_as_iadp(comparison):
    cfg, traces, _ = comparison
    for trace in traces.values():
        previous_applied = np.concatenate(([0.0], trace[:-1, 7]))
        assert np.max(np.abs(trace[:, 6] - previous_applied)) <= 60 * cfg.dt + 1e-10
        assert np.max(np.abs(trace[:, 6])) <= 10 + 1e-10
        assert np.max(np.abs(trace[:, 4])) <= 25 + 1e-10
        assert np.max(np.abs(trace[:, 5])) <= 60 + 1e-10


def test_tuning_never_uses_the_fault_or_evaluation_reference(monkeypatch):
    real_rollout = example.rollout_pid
    observed = []

    def observed_rollout(cfg, initial_state, reference, gains, *, fault):
        assert fault is False
        assert cfg.loss == 0.0
        assert not np.allclose(reference, example.reference_signal(cfg))
        trace, diagnostics = real_rollout(
            cfg, initial_state, reference, gains, fault=fault
        )
        assert diagnostics["events"] == []
        observed.append(gains.copy())
        return trace, diagnostics

    monkeypatch.setattr(example, "rollout_pid", observed_rollout)
    gains, report = example.tune_pid(max_evaluations=4)
    assert len(observed) == report["evaluations"] == 4
    assert report["fault_used"] is False
    assert np.isfinite(gains).all()
    assert any(np.array_equal(gains, candidate) for candidate in observed)


def test_pid_zero_loss_and_refined_integrator_are_consistent():
    cfg = example.baseline.Experiment(
        duration=1.0, fault_time=0.4, loss=0.0, integration_substeps=2
    )
    traces, _ = example.run_comparison(cfg)
    np.testing.assert_allclose(
        traces["pid_fault"], traces["pid_healthy"], atol=1e-10, rtol=0
    )
    assert traces["pid_fault"].shape == (cfg.steps, len(example.TRACE_COLUMNS))
    assert traces["pid_fault"][-1, 0] == cfg.duration
