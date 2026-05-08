"""With L1 active, replay must record u_safe (post-shield), not u_indi."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_replay_records_post_shield_action() -> None:
    """The off-policy correction: ``Transition.a_actual`` is the action that
    actually entered the env (i.e. ``u_safe`` after L1), not the inner-loop
    ``u_indi``.

    With the ``_Identity`` placeholder value-fn the QP becomes infeasible once
    the conformal margin exceeds 1.0 (severity-driven), so the shield falls
    back to nominal — meaning bounds are not strictly enforced. What this
    test guarantees is the *off-policy correction itself*: whatever leaves
    the controller is what lands in the replay's ``a_actual``.
    """
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=10,
        enable_l1_shield=True, enable_glr=False,
        enable_l4_outer=True, l4_n_ref_dim=3, l4_action_scale=0.05,
        l1_u_min=[-0.1, -0.1], l1_u_max=[0.1, 0.1],
        l1_h_clear=2.0,
    )
    ctl = UFTCController(n_state=3, n_control=2, config=cfg)
    rng = np.random.default_rng(0)
    us_seen: list[np.ndarray] = []
    for k in range(50):
        x = rng.standard_normal(3) * 0.05
        u = ctl.predict(x, np.zeros(3), time_step=k)
        us_seen.append(np.asarray(u, dtype=np.float64).copy())
        ctl.learn(x, np.zeros(3), time_step=k)

    snap = ctl.l4.replay.snapshot()
    assert len(snap) >= 30
    # Replay's ``a_actual`` must match the action returned by predict() —
    # the post-shield ``u_safe`` (projected into the actor's action-space
    # by right-pad / truncate to ``n_action``).
    n_action = int(ctl.l4.cfg.n_action)
    for i, tr in enumerate(snap):
        u = us_seen[i].reshape(-1)
        expected = np.zeros(n_action)
        m = min(n_action, u.size)
        expected[:m] = u[:m]
        np.testing.assert_allclose(tr.a_actual, expected, atol=1e-9)
