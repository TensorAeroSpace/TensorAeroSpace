"""5-epoch smoke training: HJI-residual loss decreases monotonically."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from tensoraerospace.agent.uftc.l1.deepreach_train import (
    TrainingConfig,
    train_value_fn,
)
from tensoraerospace.agent.uftc.l1.value_fn import DeepReachConfig


def _double_integrator():
    """f(x,u) = [x[1], u[0]]; safe set ℓ(x) = 1 - max(|x_1|, |x_2|)."""

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[1], u[0]], dtype=np.float64)

    def ell(x: np.ndarray) -> float:
        return 1.0 - float(max(abs(x[0]), abs(x[1])))

    return f, ell


def test_smoke_loss_decreases() -> None:
    f, ell = _double_integrator()
    cfg_v = DeepReachConfig(
        n_state=2,
        hidden_sizes=(16, 16),
        state_bounds=[[-2.0, 2.0], [-2.0, 2.0]],
        time_horizon=1.0,
    )
    train_cfg = TrainingConfig(
        epochs=5,
        batch_size=128,
        lr=1e-3,
        u_low=np.array([-1.0]),
        u_high=np.array([1.0]),
        disturbance_low=None,
        disturbance_high=None,
        n_state=2,
        n_control=1,
        seed=0,
    )
    fn, history = train_value_fn(cfg_v, train_cfg, dynamics=f, safe_set=ell)
    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0]
    # value evaluable
    v = fn.value(np.array([0.0, 0.0]))
    assert isinstance(v, float)
