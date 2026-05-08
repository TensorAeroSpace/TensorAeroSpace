"""UFTCController integration smoke: L1 + GLR flags toggle correctly."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _build_controller(*, enable_l1: bool, enable_glr: bool) -> UFTCController:
    cfg = UFTCConfig(
        dt=0.01,
        fdd_warmup_steps=20,
        enable_l1_shield=enable_l1,
        enable_glr=enable_glr,
    )
    return UFTCController(n_state=3, n_control=2, config=cfg)


def test_flags_persist_through_predict_learn() -> None:
    ctl = _build_controller(enable_l1=False, enable_glr=False)
    x = np.array([0.1, 0.0, -0.05])
    r = np.zeros(3)
    u = ctl.predict(x, r, time_step=0)
    info = ctl.learn(x + np.array([0.001, 0.0, 0.0]), r, time_step=0)
    assert isinstance(u, np.ndarray)
    assert u.shape == (2,)
    assert isinstance(info, dict)


def test_l1_shield_runs_when_enabled() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    ctl = _build_controller(enable_l1=True, enable_glr=False)
    rng = np.random.default_rng(0)
    for k in range(50):
        x = rng.standard_normal(3) * 0.05
        r = np.zeros(3)
        u = ctl.predict(x, r, time_step=k)
        ctl.learn(x, r, time_step=k)
    diag = ctl.diagnostics()
    assert "l1" in diag


def test_glr_severity_appears_in_diag_when_enabled() -> None:
    ctl = _build_controller(enable_l1=False, enable_glr=True)
    rng = np.random.default_rng(0)
    for k in range(60):
        x = rng.standard_normal(3) * 0.05
        ctl.predict(x, np.zeros(3), time_step=k)
        ctl.learn(x, np.zeros(3), time_step=k)
    diag = ctl.diagnostics()
    assert "fdd" in diag
    assert "severity_gradual" in diag["fdd"]
