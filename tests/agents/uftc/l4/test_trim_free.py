"""LongitudinalTrimFreeWrapper passthrough vs replace behaviour."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.l4.trim_free import (
    LongitudinalTrimFreeConfig,
    LongitudinalTrimFreeWrapper,
)


def test_disabled_passthrough() -> None:
    w = LongitudinalTrimFreeWrapper(
        LongitudinalTrimFreeConfig(V_idx=0, gamma_idx=1, alpha_idx=2,
                                   q_idx=3, enabled=False),
    )
    base = np.array([1.0, 2.0, 3.0, 4.0])
    out = w.apply(np.array([0.5, 0.6]), x_obs=np.zeros(4), base_reference=base)
    assert np.allclose(out, base)


def test_enabled_replaces_alpha_q_indices() -> None:
    w = LongitudinalTrimFreeWrapper(
        LongitudinalTrimFreeConfig(V_idx=0, gamma_idx=1, alpha_idx=2,
                                   q_idx=3, enabled=True),
    )
    base = np.array([100.0, 0.05, 999.0, 999.0])
    out = w.apply(np.array([0.04, 0.01]), x_obs=np.zeros(4), base_reference=base)
    assert out[0] == base[0]
    assert out[1] == base[1]
    assert out[2] == 0.04
    assert out[3] == 0.01
