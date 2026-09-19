"""Verify the explicit integral-state experiment without changing iADP equations."""

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_tuned_f16 as example,
)


def test_integral_model_matches_causal_error_update():
    cfg = example.baseline.Experiment()
    _, agent, _ = example.make_agent(cfg, example.INTEGRAL_TUNING)
    state = np.array([0.02, -0.03, 0.01, 0.0])  # q, z, q_ref, zero
    predicted = agent.F @ state + agent.G[:, 0] * 0.2
    expected = example.integrate_error(state[1], state[0], state[2], cfg.dt)
    assert predicted[1] == pytest.approx(expected)
    assert agent.n_state == 2
    np.testing.assert_array_equal(agent.Q, np.diag([1.0, 30.0]))
    assert agent.cfg.policy_eval_min_samples == agent.cfg.policy_eval_window == 300
    assert not hasattr(agent.cfg, "policy_eval_blend")


def test_integral_feature_is_zero_for_perfect_tracking():
    integral = 0.0
    for q in (0.1, -0.2, 0.3):
        integral = example.integrate_error(integral, q, q, 0.02)
    assert integral == 0.0
    assert example.integrate_error(0.0, 0.0, 0.01, 0.02) == pytest.approx(0.0002)


def test_frozen_ablation_keeps_integral_action_and_measurements():
    cfg = example.baseline.Experiment(duration=8.0, fault_time=2.0)
    adaptive, diagnostics = example.rollout(cfg, example.INTEGRAL_TUNING, fault=True)
    frozen, frozen_diagnostics = example.rollout(
        cfg, example.INTEGRAL_TUNING, fault=True, frozen_after_event=True
    )
    np.testing.assert_array_equal(adaptive[: cfg.fault_step], frozen[: cfg.fault_step])
    assert frozen_diagnostics["model_change_after_event"] == 0.0
    assert frozen_diagnostics["critic_change_after_event"] == 0.0
    assert diagnostics["model_change_after_event"] > 0.0
    assert diagnostics["critic_change_after_event"] > 0.0
    assert abs(frozen_diagnostics["final_integral_error_rad"]) > 1e-6
    assert np.ptp(frozen[cfg.fault_step :, 6]) > 0.01


def test_rate_profile_keeps_original_feature_space():
    cfg = example.baseline.Experiment(duration=1.0, fault_time=0.4)
    initial, agent, _ = example.make_agent(cfg, example.RATE_TUNING)
    assert agent.n_state == 1
    reference = example.comparison.reference_signal(cfg)
    expected, _ = example.baseline.rollout(
        cfg, initial, agent, reference, fault=True, adaptive=True
    )
    actual, _ = example.rollout(cfg, example.RATE_TUNING, fault=True)
    np.testing.assert_array_equal(actual, expected[:, :8])


def test_zero_fault_and_integrator_refinement_preserve_integral_controller():
    cfg = example.baseline.Experiment(
        duration=1.0, fault_time=0.4, loss=0.0, integration_substeps=2
    )
    healthy, _ = example.rollout(cfg, example.INTEGRAL_TUNING, fault=False)
    faulty, _ = example.rollout(cfg, example.INTEGRAL_TUNING, fault=True)
    np.testing.assert_allclose(healthy, faulty, rtol=0, atol=1e-10)
    previous_applied = np.concatenate(([0.0], faulty[:-1, 7]))
    assert np.max(np.abs(faulty[:, 6] - previous_applied)) <= 60 * cfg.dt + 1e-10


def test_recovery_requires_staying_in_band_until_end():
    cfg = example.baseline.Experiment(duration=10.0, fault_time=2.0)
    trace = np.zeros((10, 8))
    trace[:, 0] = np.arange(1, 11)
    trace[:3, 2] = 0.1
    assert example.recovery_time(trace, cfg, band=0.05) == 2.0
    trace[7, 2] = 0.06
    # A late return inside the band must not qualify with <5 seconds remaining.
    assert example.recovery_time(trace, cfg, band=0.05) is None


def test_learning_telemetry_is_passive_and_records_frozen_parameters():
    cfg = example.baseline.Experiment(duration=2.0, fault_time=0.4)
    expected, _ = example.rollout(
        cfg, example.INTEGRAL_TUNING, fault=True, frozen_after_event=True
    )
    rows = []
    actual, _ = example.rollout(
        cfg,
        example.INTEGRAL_TUNING,
        fault=True,
        frozen_after_event=True,
        telemetry=rows,
    )
    data = np.asarray(rows)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data[:, :8], expected)
    assert data.shape == (cfg.steps, 8 + len(example.LEARNING_COLUMNS))
    assert np.all(data[cfg.fault_step :, 12:14] == 0)
    assert np.ptp(data[cfg.fault_step :, 8]) > 0  # Integral still responds.
    assert np.all(data[:, 10] < 0)  # Physical sign of the identified input gain.
    assert np.all(data[:, 14] > 0)  # RLS covariance remains positive definite.


def test_telemetry_keeps_completed_steps_if_plant_fails(monkeypatch):
    cfg = example.baseline.Experiment(duration=1.0, fault_time=0.4)
    make_env = example.baseline.make_environment

    def failing_environment(*args, **kwargs):
        env = make_env(*args, **kwargs)
        step = env.step
        calls = 0

        def fail_on_third_step(action):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("Injected plant failure")
            return step(action)

        env.step = fail_on_third_step
        return env

    monkeypatch.setattr(example.baseline, "make_environment", failing_environment)
    rows = []
    with pytest.raises(RuntimeError, match="Injected plant failure"):
        example.rollout(cfg, example.INTEGRAL_TUNING, fault=True, telemetry=rows)
    assert len(rows) == 2
    np.testing.assert_array_equal(np.asarray(rows)[:, 0], [0.02, 0.04])
