"""Trim-free wrapper enabled: alpha/q indices in r_eff come from actor."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_trim_free_overwrites_alpha_q_indices_in_r_eff() -> None:
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=10,
        enable_l4_outer=True, l4_n_ref_dim=4, l4_action_scale=0.0,
        l4_trim_free={"V_idx": 0, "gamma_idx": 1, "alpha_idx": 2, "q_idx": 3},
    )
    ctl = UFTCController(n_state=4, n_control=2, config=cfg)
    base = np.array([100.0, 0.05, 9999.0, -9999.0])
    ctl.predict(np.zeros(4), base, time_step=0)
    assert ctl._last_r_eff is not None
    # V and gamma preserved; alpha and q replaced by something *other* than 9999.
    assert ctl._last_r_eff[0] == base[0]
    assert ctl._last_r_eff[1] == base[1]
    assert ctl._last_r_eff[2] != 9999.0
    assert ctl._last_r_eff[3] != -9999.0
