"""1000-step smoke test on a mock plant — no NaN, all keys filled."""

from __future__ import annotations

import math

import numpy as np

from tensoraerospace.agent import UFTCConfig, UFTCController


def test_thousand_step_smoke_no_nan() -> None:
    rng = np.random.default_rng(0)
    n_state = 3
    n_control = 2
    F = np.array([[-0.1, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, -0.2]])
    G = np.array([[0.5, 0.0], [0.0, 0.4], [0.1, 0.1]])
    ctl = UFTCController(
        n_state=n_state,
        n_control=n_control,
        nominal_F=F,
        nominal_G=G,
        config=UFTCConfig(fdd_warmup_steps=50),
    )
    x = np.zeros(n_state)
    for k in range(1000):
        ref = 0.1 * np.array(
            [math.sin(k * 0.05), math.cos(k * 0.07), math.sin(k * 0.03)]
        )
        u = ctl.predict(x, ref, time_step=k)
        assert np.isfinite(u).all()
        x = x + (F @ x + G @ u) * 0.01 + 0.005 * rng.normal(size=n_state)
        ctl.learn(x, ref, time_step=k)

    diag = ctl.diagnostics()
    for key in (
        "step",
        "fault_present",
        "severity",
        "confidence",
        "innovation_norm",
        "rls_gamma",
        "mode",
        "fdd_active",
    ):
        assert key in diag
    assert diag["step"] == 1000
