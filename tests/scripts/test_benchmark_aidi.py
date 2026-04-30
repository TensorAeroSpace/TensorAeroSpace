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
