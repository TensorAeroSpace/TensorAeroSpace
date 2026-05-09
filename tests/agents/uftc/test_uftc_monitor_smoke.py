"""enable_monitor wiring: collect_vstate, monitor.step, dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi.model import AAINDIConfig
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_monitor_emits_diagnostics_block() -> None:
    cfg = UFTCConfig(
        dt=0.01,
        fdd_warmup_steps=20,
        enable_monitor=True,
        inner_cfg=AAINDIConfig(seed=0),
    )
    ctl = UFTCController(n_state=3, n_control=2, config=cfg)
    rng = np.random.default_rng(0)
    for k in range(40):
        x = rng.standard_normal(3) * 0.05
        ctl.predict(x, np.zeros(3), time_step=k)
        ctl.learn(x, np.zeros(3), time_step=k)
    diag = ctl.diagnostics()
    assert "monitor" in diag
    assert diag["monitor"]["alarm"] in ("OK", "WARN", "CRITICAL")
    assert "V_total" in diag["monitor"]
    assert "mu_uub_pred" in diag["monitor"]


def test_monitor_off_invariance_with_phase123() -> None:
    seed = 999

    def rollout(enable_monitor: bool):
        rng = np.random.default_rng(seed)
        cfg = UFTCConfig(
            dt=0.01,
            fdd_warmup_steps=20,
            enable_monitor=enable_monitor,
            inner_cfg=AAINDIConfig(seed=0),
        )
        ctl = UFTCController(n_state=4, n_control=2, config=cfg)
        xs, us = [], []
        x = rng.standard_normal(4) * 0.1
        for k in range(200):
            u = ctl.predict(x, np.zeros(4), time_step=k)
            x = x + 0.01 * (rng.standard_normal(4) * 0.05 - 0.1 * x)
            ctl.learn(x, np.zeros(4), time_step=k)
            xs.append(x.copy())
            us.append(np.asarray(u, dtype=np.float64).copy())
        return np.stack(xs), np.stack(us)

    x_off, u_off = rollout(enable_monitor=False)
    x_on, u_on = rollout(enable_monitor=True)
    # With no L4/L1 active, monitor's macro-actions are no-ops (no L1/L4 to call).
    np.testing.assert_array_equal(x_off, x_on)
    np.testing.assert_array_equal(u_off, u_on)
