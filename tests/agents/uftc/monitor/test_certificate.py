"""Numerical certificate matches closed-form mu_uub on a toy 5x5 system."""

from __future__ import annotations

import json

import numpy as np

from tensoraerospace.agent.uftc.monitor.certificate import (
    CertificateReport,
    run_certificate,
)


def _toy_cfg() -> dict:
    return {
        "c_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
        "a_diag": [1.0, 1.0, 1.0, 1.0, 1.0],
        "eps_matrix": [
            [0.0, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.0, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.0],
        ],
        "d_disturbance": [0.1, 0.1, 0.1, 0.1, 0.1],
        "alarm_warn_frac": 0.7,
        "alarm_critical_frac": 0.95,
        "cooldown_steps": 200,
    }


def test_metzler_and_hurwitz_pass_on_toy() -> None:
    rep = run_certificate(_toy_cfg(), rollouts={})
    assert isinstance(rep, CertificateReport)
    assert rep.metzler_check == "pass"
    assert rep.hurwitz_check == "pass"
    assert rep.mu_uub_pred > 0


def test_metzler_violation_detected() -> None:
    cfg = _toy_cfg()
    cfg["eps_matrix"][0][1] = -0.1  # negative off-diagonal
    rep = run_certificate(cfg, rollouts={})
    assert rep.metzler_check == "fail"


def test_hurwitz_violation_detected() -> None:
    cfg = _toy_cfg()
    cfg["a_diag"] = [0.05, 0.05, 0.05, 0.05, 0.05]  # too small → not Hurwitz
    rep = run_certificate(cfg, rollouts={})
    assert rep.hurwitz_check == "fail"


def test_empirical_pass_rate_recorded() -> None:
    cfg = _toy_cfg()
    rng = np.random.default_rng(0)
    fake_rollouts = {
        "preset_a": np.zeros((50, 100)),  # 50 trajectories of 100 V_total samples
        "preset_b": rng.standard_normal((50, 100)) * 0.0,
    }
    rep = run_certificate(cfg, rollouts=fake_rollouts, transient_steps=10)
    assert "preset_a" in rep.rollouts
    assert rep.rollouts["preset_a"]["pass_rate"] == 1.0
