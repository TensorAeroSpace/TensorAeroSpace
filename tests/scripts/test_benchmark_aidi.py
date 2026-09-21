"""Smoke test for the AIDI benchmark CLI — short scenario, structured output."""

import csv
import sys

import pytest


@pytest.mark.integration
def test_benchmark_aidi_emits_report(tmp_path, monkeypatch):
    out_md = tmp_path / "report.md"
    out_csv = tmp_path / "report.csv"
    argv = [
        "benchmark_aidi",
        "--env",
        "f16_nonlinear_angular",
        "--baselines",
        "frozen",
        "--scenarios",
        "nominal,stab_25",
        "--episodes",
        "1",
        "--steps",
        "300",
        "--out",
        str(out_md),
        "--csv",
        str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    from tensoraerospace.scripts.benchmark_aidi import main

    main()
    assert out_md.exists() and out_md.stat().st_size > 50
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header + 2 scenarios × 2 methods (adaptive + frozen) = 5 rows minimum.
    assert len(rows) >= 5


def test_trim_matches_the_angular_plant_and_level_attitude():
    import numpy as np

    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
        f16_ode_6dof,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )
    from tensoraerospace.scripts.benchmark_aidi import _solve_trim

    alpha, stab = _solve_trim()
    x = np.zeros(14)
    x[0] = x[7] = alpha
    x[8] = stab
    dx = f16_ode_6dof(x, [stab, 0, 0], 0, default_parameters())
    np.testing.assert_allclose(dx[[0, 4]], 0, atol=1e-8)
    # Lateral interpolation residuals at zero beta are below 1e-7.
    np.testing.assert_allclose(dx, 0, atol=1e-7)


@pytest.mark.parametrize("method", ["adaptive", "frozen"])
def test_benchmark_only_freezes_identifier_and_uses_actual_feedback(
    monkeypatch, method
):
    import numpy as np

    from tensoraerospace.scripts import benchmark_aidi as benchmark

    agent = benchmark._build_agent(method)
    feedback = []
    original = agent.learn

    def learn(obs, *args, **kwargs):
        feedback.append((kwargs["applied_action"].copy(), agent._last_u_cmd.copy()))
        assert obs["V"] == 120.0
        return original(obs, *args, **kwargs)

    monkeypatch.setattr(agent, "learn", learn)
    monkeypatch.setattr(benchmark, "_build_agent", lambda _: agent)
    benchmark._run_episode(method, "nominal", 210, *benchmark._solve_trim())
    assert len(feedback) == 210
    assert any(np.max(abs(applied - command)) > 1e-8 for applied, command in feedback)
    if method == "frozen":
        assert agent.rls.num_updates == 0
        np.testing.assert_array_equal(agent.rls.theta, np.ones((3, 3)))
    else:
        assert agent.rls.num_updates == 209


@pytest.mark.parametrize(
    "args", [["--episodes", "0"], ["--steps", "100"], ["--scenarios", ""]]
)
def test_benchmark_rejects_empty_evaluations(tmp_path, args):
    from tensoraerospace.scripts.benchmark_aidi import main

    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "invalid.md"), *args])
    assert not (tmp_path / "invalid.md").exists()
