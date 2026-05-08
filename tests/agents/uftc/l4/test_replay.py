"""PrioritizedReplay storage and weight semantics."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l4.replay import (
    PrioritizedReplay,
    Transition,
)


def _t(reward: float = 0.0) -> Transition:
    return Transition(
        s=np.zeros(3), a_actual=np.zeros(2), r_used=np.zeros(3),
        reward=reward, s_next=np.zeros(3), done=False,
        fdd=FDDOutput(False, 0.0, 0.0, 0.0, 0.0,
                      fault_kind="none",
                      severity_abrupt=0.0, severity_gradual=0.0),
        alarm="OK",
    )


def test_push_and_len() -> None:
    rep = PrioritizedReplay(capacity=10, alpha=0.6)
    rep.push(_t(0.1))
    rep.push(_t(0.2))
    assert len(rep) == 2


def test_capacity_evicts_oldest() -> None:
    rep = PrioritizedReplay(capacity=3, alpha=0.6)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rep.push(_t(r))
    assert len(rep) == 3
    rewards = [t.reward for t in rep.snapshot()]
    assert rewards == [0.3, 0.4, 0.5]


def test_sample_returns_indices_and_weights() -> None:
    rep = PrioritizedReplay(capacity=20, alpha=0.6, beta_init=0.4)
    for r in range(20):
        rep.push(_t(float(r)), priority=1.0 + r)
    transitions, idx, w = rep.sample(8)
    assert len(transitions) == 8
    assert len(idx) == 8
    assert w.shape == (8,)
    assert (w > 0).all()


def test_a_actual_is_stored_unchanged() -> None:
    rep = PrioritizedReplay(capacity=10, alpha=0.6)
    t = _t()
    t.a_actual = np.array([0.7, -0.3])
    rep.push(t)
    snap = rep.snapshot()
    assert np.allclose(snap[0].a_actual, [0.7, -0.3])
