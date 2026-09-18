"""Validate event-blind controllers and the physical scope of fault scenarios."""

from dataclasses import replace

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_fault_scenarios_f16 as example,
)


def test_schedules_are_reproducible_and_independent_of_controller_results():
    first = example.scenarios(seed=7)
    second = example.scenarios(seed=7)
    different = example.scenarios(seed=8)
    assert {k: v.to_dict() for k, v in first.items()} == {
        k: v.to_dict() for k, v in second.items()
    }
    assert first["late_gain_loss"].to_dict() != different["late_gain_loss"].to_dict()
    for profile in first.values():
        for event in profile.events:
            assert 0 < event.trigger_time < 500
            assert event.trigger_time / 0.02 == pytest.approx(
                round(event.trigger_time / 0.02)
            )
    ramp = first["progressive_gain_loss"].events
    efficiencies = [e.payload["efficiency"] for e in ramp]
    assert np.all(np.diff(efficiencies) < 0)
    assert efficiencies[-1] == pytest.approx(0.8)


def test_continuous_iadp_cannot_use_evaluation_fault_time():
    cfg = example.tuned.baseline.Experiment(duration=2.0, fault_time=0.4)
    profile = example.DamageProfile(
        [
            example.DamageEvent(
                0.8,
                "control_failure",
                {
                    "surface": "stab_left",
                    "mode": "efficiency_loss",
                    "efficiency": 0.85,
                },
            )
        ]
    )
    rows, other_rows = [], []
    first, first_d = example.tuned.rollout(
        cfg,
        example.tuned.INTEGRAL_TUNING,
        fault=True,
        damage_profile=profile,
        telemetry=rows,
    )
    second, second_d = example.tuned.rollout(
        replace(cfg, fault_time=1.4),
        example.tuned.INTEGRAL_TUNING,
        fault=True,
        damage_profile=profile,
        telemetry=other_rows,
    )
    # cfg.fault_time only sets the diagnostic reference point; it cannot alter
    # actions, model estimates, critic or integral feedback in the online arm.
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        np.asarray(rows)[:, :12], np.asarray(other_rows)[:, :12]
    )
    assert first_d["events"] == second_d["events"]
    assert first_d["events"][0]["time"] == 0.8
    assert first_d["model_change_after_event"] > 0


def test_pid_custom_profile_and_passive_telemetry_keep_fixed_gains():
    cfg = example.tuned.baseline.Experiment(duration=1.0, fault_time=0.4)
    initial, _, _ = example.tuned.baseline.trim_and_agent(cfg)
    reference = example.tuned.comparison.reference_signal(cfg)
    profile = example.DamageProfile([])
    rows = []
    trace, diagnostics = example.tuned.comparison.rollout_pid(
        cfg,
        initial,
        reference,
        example.tuned.comparison.DEFAULT_PID_GAINS,
        fault=True,
        damage_profile=profile,
        telemetry=rows,
    )
    healthy, _ = example.tuned.comparison.rollout_pid(
        cfg,
        initial,
        reference,
        example.tuned.comparison.DEFAULT_PID_GAINS,
        fault=False,
    )
    np.testing.assert_array_equal(trace, healthy)
    np.testing.assert_array_equal(trace, rows)
    assert diagnostics["events"] == []
    assert diagnostics["gains_before"] == diagnostics["gains_after"]


def test_symmetric_tip_loss_changes_native_plant_parameters_at_event():
    cfg = example.tuned.baseline.Experiment(duration=1.0, fault_time=0.4)
    initial, _, _ = example.tuned.baseline.trim_and_agent(cfg)
    reference = example.tuned.comparison.reference_signal(cfg)
    events = [
        replace(e, trigger_time=0.4)
        for e in example.scenarios()["symmetric_wing_tip_loss"].events
    ]
    env = example.tuned.baseline.make_environment(
        cfg,
        initial,
        reference,
        fault=True,
        damage_profile=example.DamageProfile(events),
    )
    try:
        observation, _ = env.reset()
        assert observation.shape == (4,)
        baseline = (env.model.param.m, env.model.param.S, env.model.param.Jz)
        for _ in range(cfg.fault_step - 1):
            env.step(np.zeros(1))
        assert (env.model.param.m, env.model.param.S, env.model.param.Jz) == baseline
        env.step(np.zeros(1))
        damaged = (env.model.param.m, env.model.param.S, env.model.param.Jz)
        assert all(0 < new < old for new, old in zip(damaged, baseline))
        state = env.damage_manager.state
        assert state.section_loss["left_tip"] == state.section_loss["right_tip"] == 0.3
        assert len(env.damage_events_log) == 2
        assert all(e["time"] == 0.4 for e in env.damage_events_log)
    finally:
        env.close()
