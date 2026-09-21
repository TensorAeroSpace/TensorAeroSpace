"""The demo must isolate adaptation and use the real nonlinear servo/fault path."""

from dataclasses import replace

import numpy as np
import pytest

from example.reinforcement_learning.incremental_adp import (
    example_iadp_small_fault_f16 as demo,
)


@pytest.fixture(scope="module")
def short_experiment():
    cfg = demo.Experiment(duration=8.0, fault_time=2.0)
    return cfg, *demo.run_experiment(cfg)


def test_controllers_share_the_complete_prefault_history(short_experiment):
    cfg, traces, report = short_experiment
    healthy = traces["healthy_adaptive"][: cfg.fault_step]
    for trace in traces.values():
        np.testing.assert_array_equal(trace[: cfg.fault_step], healthy)
    assert report["prefault_histories_match"]
    assert report["metrics"]["fault_adaptive"]["events"][0]["time"] == cfg.fault_time
    assert report["metrics"]["healthy_adaptive"]["events"] == []


def test_freezing_parameters_keeps_feedback_and_control_active(short_experiment):
    cfg, traces, _ = short_experiment
    frozen = traces["fault_frozen"][cfg.fault_step :]
    adaptive = traces["fault_adaptive"][cfg.fault_step :]
    # Both the incremental model and value matrix are fixed only in the ablation.
    np.testing.assert_array_equal(frozen[:, 11:13], 0.0)
    assert np.max(adaptive[:, 11]) > 1e-7
    assert np.max(adaptive[:, 12]) > 1e-4
    # A frozen policy still responds to changing observations after the fault.
    assert np.ptp(frozen[:, 6]) > 0.01
    assert np.isfinite(frozen).all()


def test_native_fault_scales_total_servo_target_once_at_the_boundary():
    cfg = demo.Experiment(duration=0.1, fault_time=0.04)
    initial, _, _ = demo.trim_and_agent(cfg)
    reference = np.zeros((1, cfg.steps + 1))
    healthy = demo.make_environment(cfg, initial, reference, fault=False)
    faulty = demo.make_environment(cfg, initial, reference, fault=True)
    healthy.reset()
    faulty.reset()
    try:
        for _ in range(cfg.fault_step):
            healthy.step(np.zeros(1))
            faulty.step(np.zeros(1))
            np.testing.assert_array_equal(
                healthy.model.current_state, faulty.model.current_state
            )
        healthy.step(np.zeros(1))
        faulty.step(np.zeros(1))
        assert float(faulty.model.u_history[-1][0, 0]) == pytest.approx(
            (1 - cfg.loss) * initial[2]
        )
        assert len(faulty.damage_events_log) == 1
        # No instantaneous state teleport: the second-order servo moves toward
        # the attenuated target. A zero delta command still disturbs the trim.
        assert initial[2] < faulty.model.current_state[2] < (1 - cfg.loss) * initial[2]
        assert abs(faulty.model.current_state[1]) > 1e-5
        assert faulty.model.param.Jz == healthy.model.param.Jz
        assert faulty.model.param.m == healthy.model.param.m
    finally:
        healthy.close()
        faulty.close()


def test_zero_loss_is_the_healthy_aircraft_with_identical_learning():
    cfg = demo.Experiment(duration=1.0, fault_time=0.4, loss=0.0)
    traces, _ = demo.run_experiment(cfg)
    for mode in ("adaptive", "frozen"):
        np.testing.assert_array_equal(
            traces[f"fault_{mode}"], traces[f"healthy_{mode}"]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"loss": -0.1},
        {"loss": 0.5},
        {"dt": 0.04},
        {"dt": 0.01},
        {"integration_substeps": 0},
        {"integration_substeps": 1.5},
        {"dt": 0.0},
        {"phase": np.nan},
        {"fault_time": 20.005},
        {"duration": 20.0},
    ],
)
def test_invalid_scenarios_fail_before_simulation(changes):
    with pytest.raises(ValueError):
        replace(demo.Experiment(), **changes)
