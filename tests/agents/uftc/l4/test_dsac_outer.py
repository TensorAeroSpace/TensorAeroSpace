"""DSACOuter end-to-end on a 2-state mock plant."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l4 import (
    DSACConfig,
    DSACOuter,
    Transition,
)


def _zero_fdd() -> FDDOutput:
    return FDDOutput(
        False,
        0.0,
        0.0,
        0.0,
        0.0,
        fault_kind="none",
        severity_abrupt=0.0,
        severity_gradual=0.0,
    )


def test_predict_returns_tuple_in_eval_mode() -> None:
    cfg = DSACConfig(
        n_state=3,
        n_ref_dim=3,
        n_action=3,
        critic_hidden=(8, 8),
        actor_hidden=(8, 8),
        n_quantiles=4,
        batch_size=4,
        replay_capacity=32,
    )
    dsac = DSACOuter(cfg)
    r_tilde, beta, reset_hint = dsac.predict(
        x_obs=np.zeros(3),
        base_reference=np.zeros(3),
        fdd=_zero_fdd(),
        monitor_alarm="OK",
    )
    assert r_tilde.shape == (3,)
    assert 0.0 <= beta <= 1.0
    assert reset_hint in (False, True)


def test_freeze_learning_blocks_updates() -> None:
    cfg = DSACConfig(
        n_state=2,
        n_ref_dim=2,
        n_action=2,
        critic_hidden=(4,),
        actor_hidden=(4,),
        n_quantiles=4,
        batch_size=2,
        replay_capacity=8,
    )
    dsac = DSACOuter(cfg)
    # Push some transitions.
    for _ in range(4):
        dsac.learn(
            Transition(
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                0.0,
                np.zeros(2),
                False,
                _zero_fdd(),
                "OK",
            )
        )
    # Get current actor parameters.
    p0 = next(dsac.actor.parameters()).detach().clone()
    dsac.freeze_learning(until_step=10**6)
    for _ in range(8):
        dsac.learn(
            Transition(
                np.random.randn(2),
                np.random.randn(2),
                np.random.randn(2),
                1.0,
                np.random.randn(2),
                False,
                _zero_fdd(),
                "OK",
            )
        )
    p1 = next(dsac.actor.parameters()).detach().clone()
    assert torch.allclose(p0, p1)


def test_degrade_reference_to_hold_passthrough() -> None:
    cfg = DSACConfig(
        n_state=3,
        n_ref_dim=3,
        n_action=3,
        critic_hidden=(4,),
        actor_hidden=(4,),
        n_quantiles=4,
        batch_size=2,
        replay_capacity=8,
    )
    dsac = DSACOuter(cfg)
    base = np.array([0.5, -0.2, 0.1])
    dsac.degrade_reference_to_hold()
    r_tilde, _, _ = dsac.predict(np.zeros(3), base, _zero_fdd(), "OK")
    assert np.allclose(r_tilde, base)


def test_save_load_round_trip(tmp_path) -> None:
    cfg = DSACConfig(
        n_state=2,
        n_ref_dim=2,
        n_action=2,
        critic_hidden=(4,),
        actor_hidden=(4,),
        n_quantiles=4,
        batch_size=2,
        replay_capacity=8,
    )
    dsac = DSACOuter(cfg)
    r0, _, _ = dsac.predict(np.zeros(2), np.zeros(2), _zero_fdd(), "OK")
    dsac.save(tmp_path)
    dsac2 = DSACOuter.from_pretrained(tmp_path, cfg=cfg)
    r1, _, _ = dsac2.predict(np.zeros(2), np.zeros(2), _zero_fdd(), "OK")
    assert np.allclose(r0, r1, atol=1e-5)
