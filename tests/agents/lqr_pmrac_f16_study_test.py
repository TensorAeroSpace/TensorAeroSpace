import pytest

from article.experiments.run_lqr_pmrac_f16_study import paired_intervals


def test_pmrac_bootstrap_uses_post_fault_pitch_rate_and_completed_pairs():
    reports = []
    for method, value, status in (
        ("lqr", 1.0, "complete"),
        ("mrac_online", 0.8, "complete"),
        ("pmrac_frozen", 0.9, "complete"),
        ("pmrac_online", 0.5, "complete"),
        ("mrac_frozen", 1.0, "complete"),
    ):
        reports.append(
            {
                "scenario": {"seed": 1},
                "method": method,
                "fault": True,
                "diagnostics": {"status": status},
                "metrics": {"q_post_fault_rmse": value},
            }
        )
    reports.append(
        {
            "scenario": {"seed": 2},
            "method": "pmrac_online",
            "fault": True,
            "diagnostics": {"status": "failed"},
            "metrics": {"q_post_fault_rmse": -100.0},
        }
    )
    result = paired_intervals(reports)
    assert result["pmrac_online_minus_lqr"]["mean_difference"] == pytest.approx(-0.5)
    assert result["pmrac_online_minus_mrac_online"]["mean_difference"] == pytest.approx(
        -0.3
    )
    assert result["pmrac_online_minus_pmrac_frozen"][
        "mean_difference"
    ] == pytest.approx(-0.4)
    assert result["pmrac_online_minus_lqr"]["scenarios"] == 1
