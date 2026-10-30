import numpy as np

from article.experiments.run_et_dhp_f16_study import Scenario
from article.experiments.run_iadp_f16_study import q_reference
from article.experiments.run_lqr_mrac_f16_study import paired_intervals
from example.lqr_controllers.example_lqr_f16 import Experiment
from example.lqr_controllers.example_lqr_f16 import q_reference as mrac_reference


def test_study_uses_the_same_two_frequency_reference_as_iadp():
    scenario = Scenario(seed=8003, duration=2.0, amplitude_deg=2.0, frequency_hz=0.1)
    cfg = Experiment(
        duration=scenario.duration,
        amplitude_deg=scenario.amplitude_deg,
        frequency_hz=scenario.frequency_hz,
        phase=(scenario.seed % 17) * 0.1,
    )
    np.testing.assert_allclose(
        np.rad2deg(mrac_reference(cfg)),
        q_reference(None, scenario)[0, : cfg.steps + 1],
    )


def test_bootstrap_pairs_only_completed_matching_scenarios():
    reports = []
    for seed, value in ((1, 1.0), (2, 3.0)):
        for method, offset, status in (
            ("lqr", 0.0, "complete"),
            ("mrac_frozen", 1.0, "complete"),
            ("mrac_online", -0.5, "complete" if seed == 1 else "failed"),
        ):
            reports.append(
                {
                    "scenario": {"seed": seed},
                    "method": method,
                    "fault": False,
                    "diagnostics": {"status": status},
                    "metrics": {"q_late_rmse": value + offset},
                }
            )
    result = paired_intervals(reports)
    assert result["mrac_online_minus_lqr"]["scenarios"] == 1
    assert result["mrac_online_minus_lqr"]["mean_difference"] == -0.5
    assert result["mrac_frozen_minus_lqr"]["scenarios"] == 2
    assert result["mrac_frozen_minus_lqr"]["mean_difference"] == 1.0
