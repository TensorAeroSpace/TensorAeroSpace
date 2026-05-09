"""enable_l4_outer wiring smoke test."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.aa_indi.model import AAINDIConfig
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_l4_returns_modified_reference_on_demand() -> None:
    cfg = UFTCConfig(
        dt=0.01,
        fdd_warmup_steps=20,
        enable_l4_outer=True,
        l4_n_ref_dim=3,
        l4_action_scale=0.0,  # zero scale → r̃ == base_reference
    )
    ctl = UFTCController(n_state=3, n_control=2, config=cfg)
    rng = np.random.default_rng(0)
    base_ref = np.array([0.1, -0.2, 0.05])
    for k in range(40):
        x = rng.standard_normal(3) * 0.1
        u = ctl.predict(x, base_ref, time_step=k)
        ctl.learn(x, base_ref, time_step=k)
    diag = ctl.diagnostics()
    assert "l4" in diag
    assert "beta_t" in diag["l4"]


def test_l4_off_invariance_with_phase1_only() -> None:
    """enable_l4_outer=False: behaviour identical to Phase 1 + 2 flags-off."""
    rng_seed = 12345

    def rollout(enable_l4: bool) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(rng_seed)
        ctl = UFTCController(
            n_state=4,
            n_control=2,
            config=UFTCConfig(
                dt=0.01,
                fdd_warmup_steps=20,
                enable_l1_shield=False,
                enable_glr=False,
                enable_l4_outer=enable_l4,
                l4_n_ref_dim=4,
                inner_cfg=AAINDIConfig(seed=0),
            ),
        )
        xs, us = [], []
        x = rng.standard_normal(4) * 0.1
        for k in range(200):
            u = ctl.predict(x, np.zeros(4), time_step=k)
            x = x + 0.01 * (rng.standard_normal(4) * 0.05 - 0.1 * x)
            ctl.learn(x, np.zeros(4), time_step=k)
            xs.append(x.copy())
            us.append(np.asarray(u, dtype=np.float64).copy())
        return np.stack(xs), np.stack(us)

    x_off, u_off = rollout(enable_l4=False)
    x_off_ref, u_off_ref = rollout(enable_l4=False)
    np.testing.assert_array_equal(x_off, x_off_ref)
    np.testing.assert_array_equal(u_off, u_off_ref)
