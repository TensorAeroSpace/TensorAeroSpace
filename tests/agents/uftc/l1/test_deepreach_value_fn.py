"""DeepReachValueFn satisfies HJValueFunction; save/load round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l1.value_fn import (
    DeepReachConfig,
    DeepReachValueFn,
    HJValueFunction,
)


def test_value_fn_satisfies_protocol() -> None:
    cfg = DeepReachConfig(n_state=3, hidden_sizes=(32, 32))
    fn = DeepReachValueFn.from_config(cfg)
    assert isinstance(fn, HJValueFunction)


def test_value_returns_scalar_and_gradient_matches_state_dim() -> None:
    cfg = DeepReachConfig(n_state=4, hidden_sizes=(16, 16))
    fn = DeepReachValueFn.from_config(cfg, seed=0)
    x = np.array([0.1, -0.2, 0.0, 0.4])
    v = fn.value(x)
    g = fn.gradient(x)
    assert isinstance(v, float)
    assert g.shape == (4,)


def test_save_load_round_trip(tmp_path) -> None:
    cfg = DeepReachConfig(n_state=2, hidden_sizes=(8, 8))
    fn = DeepReachValueFn.from_config(cfg, seed=42)
    x = np.array([0.3, -0.1])
    v_before = fn.value(x)

    fn.save(tmp_path / "v.pt")
    fn2 = DeepReachValueFn.load(tmp_path / "v.pt")
    v_after = fn2.value(x)
    assert abs(v_before - v_after) < 1e-6
    meta = json.loads((tmp_path / "v.json").read_text())
    assert meta["n_state"] == 2


def test_lipschitz_const_returns_finite() -> None:
    cfg = DeepReachConfig(n_state=3, hidden_sizes=(8, 8))
    fn = DeepReachValueFn.from_config(cfg, seed=1)
    L = fn.lipschitz_const()
    assert isinstance(L, float)
    assert L > 0.0
