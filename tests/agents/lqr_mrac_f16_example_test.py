import numpy as np

from example.lqr_controllers.example_lqr_f16 import Experiment, make_setup, rollout


def test_f16_examples_execute_and_frozen_shares_pre_event_trace():
    setup = make_setup()
    cfg = Experiment(duration=0.4, fault_time=0.2, loss_fraction=0.3)
    online, online_report = rollout(setup, cfg, "mrac_online")
    frozen, frozen_report = rollout(setup, cfg, "mrac_frozen")
    baseline, baseline_report = rollout(setup, cfg, "lqr")
    assert (
        online_report["status"]
        == frozen_report["status"]
        == baseline_report["status"]
        == "complete"
    )
    assert np.isfinite(online).all()
    np.testing.assert_allclose(online[:10], frozen[:10], rtol=0, atol=1e-10)
    assert not np.allclose(online[10:, 9:], frozen[10:, 9:])
