import numpy as np

from example.lqr_controllers.example_lqr_f16 import (
    Experiment,
    damage_profile,
    make_setup,
    rollout,
)


def test_f16_pmrac_executes_matched_fault_and_frozen_shares_pre_event_trace():
    setup = make_setup()
    cfg = Experiment(
        duration=0.4, fault_time=0.2, loss_fraction=0.3, fault_kind="actuator"
    )
    online, online_report = rollout(setup, cfg, "pmrac_online")
    frozen, frozen_report = rollout(setup, cfg, "pmrac_frozen")
    baseline, baseline_report = rollout(setup, cfg, "lqr")
    assert online_report["status"] == "complete"
    assert frozen_report["status"] == baseline_report["status"] == "complete"
    assert online.shape[1] == 12
    assert np.isfinite(online).all()
    np.testing.assert_allclose(online[:10], frozen[:10], rtol=0, atol=1e-10)
    assert not np.allclose(online[10:, 9:], frozen[10:, 9:])


def test_actuator_fault_is_an_input_effectiveness_event():
    event = damage_profile(
        Experiment(loss_fraction=0.45, fault_kind="actuator")
    ).events[0]
    assert event.event_type == "control_failure"
    assert event.payload["efficiency"] == 0.55
