"""Unit tests for the IM-GDHP agent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensoraerospace.agent.im_gdhp import (
    GDHPActor,
    GDHPCritic,
    IMGDHPAgent,
    IMGDHPConfig,
    IncrementalModelRLS,
)


# ---------------------------------------------------------------------------
# IncrementalModelRLS
# ---------------------------------------------------------------------------
def test_rls_siso_converges_on_linear_system():
    """A and B should recover the true coefficients of y_{t+1}=0.8y+0.5u."""
    rls = IncrementalModelRLS(n_y=1, n_u=1, forgetting=0.999, cov_init=1e3, seed=0)
    rng = np.random.default_rng(0)
    y_hist = [np.array([0.0]), np.array([0.0])]
    u_hist = [np.array([0.0])]
    for t in range(300):
        u_t = np.array([float(rng.normal(0, 0.3))])
        y_next = 0.8 * y_hist[-1] + 0.5 * u_t
        y_hist.append(y_next)
        u_hist.append(u_t)
        if t >= 1:
            rls.update(
                y_prev=y_hist[-3],
                y_curr=y_hist[-2],
                y_next=y_hist[-1],
                u_prev=u_hist[-2],
                u_curr=u_hist[-1],
            )
    assert abs(rls.A[0, 0] - 0.8) < 0.05
    assert abs(rls.B[0, 0] - 0.5) < 0.05


def test_rls_mimo_converges_on_linear_system():
    """Two-state, one-input linear system recovers the full (A, B)."""
    A_true = np.array([[0.9, 0.1], [0.0, 0.8]])
    B_true = np.array([[0.0], [0.3]])
    rls = IncrementalModelRLS(n_y=2, n_u=1, forgetting=0.999, cov_init=1e3, seed=0)
    rng = np.random.default_rng(1)
    y_hist = [np.zeros(2), np.zeros(2)]
    u_hist = [np.zeros(1)]
    for t in range(600):
        u_t = rng.normal(0, 0.5, size=(1,))
        y_next = A_true @ y_hist[-1] + B_true @ u_t
        y_hist.append(y_next)
        u_hist.append(u_t)
        if t >= 1:
            rls.update(
                y_prev=y_hist[-3],
                y_curr=y_hist[-2],
                y_next=y_hist[-1],
                u_prev=u_hist[-2],
                u_curr=u_hist[-1],
            )
    assert np.allclose(rls.A, A_true, atol=0.05)
    assert np.allclose(rls.B, B_true, atol=0.05)


def test_rls_update_returns_prediction_error_shape():
    rls = IncrementalModelRLS(n_y=3, n_u=2, seed=0)
    y_prev = np.zeros(3)
    y_curr = np.zeros(3)
    y_next = np.array([0.1, 0.2, 0.3])
    u_prev = np.zeros(2)
    u_curr = np.array([0.5, -0.5])
    eps = rls.update(y_prev, y_curr, y_next, u_prev, u_curr)
    assert eps.shape == (3,)
    assert rls.num_updates == 1


def test_rls_reset_covariance_restores_identity_scaled_P():
    rls = IncrementalModelRLS(n_y=1, n_u=1, cov_init=100.0, seed=0)
    # Drive a broad set of (y, u) transitions so both rows of P shrink.
    rng = np.random.default_rng(0)
    y_hist = [np.zeros(1), np.zeros(1)]
    u_hist = [np.zeros(1)]
    for _ in range(50):
        u_t = rng.normal(0, 1.0, size=(1,))
        y_next = 0.5 * y_hist[-1] + 0.2 * u_t
        y_hist.append(y_next)
        u_hist.append(u_t)
        rls.update(y_hist[-3], y_hist[-2], y_hist[-1], u_hist[-2], u_hist[-1])
    p_before = rls.P.copy()
    rls.reset_covariance()
    expected = np.eye(2) * 100.0
    assert np.allclose(rls.P, expected)
    assert not np.allclose(p_before, expected)


def test_rls_predict_next_uses_current_estimates():
    rls = IncrementalModelRLS(n_y=1, n_u=1, seed=0)
    rls.theta[0, 0] = 1.0  # A = 1
    rls.theta[1, 0] = 2.0  # B = 2
    pred = rls.predict_next(
        y_curr=np.array([1.0]),
        y_prev=np.array([0.5]),
        u_curr=np.array([1.0]),
        u_prev=np.array([0.0]),
    )
    # y_next = 1 + 1*(1-0.5) + 2*(1-0) = 3.5
    assert pred[0] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# GDHPActor / GDHPCritic
# ---------------------------------------------------------------------------
def test_gdhp_actor_output_shape_and_bounds():
    actor = GDHPActor(in_features=4, n_u=2, u_max=10.0)
    out = actor(torch.randn(4))
    assert out.shape == (2,)
    assert torch.all(out.abs() <= 10.0 + 1e-6)


def test_gdhp_critic_returns_two_heads():
    critic = GDHPCritic(in_features=5, n_y=3)
    j, lam = critic(torch.randn(5))
    assert j.shape == (1,)
    assert lam.shape == (3,)


def test_gdhp_critic_backward_updates_all_params():
    critic = GDHPCritic(in_features=5, n_y=2, hidden_sizes=(8, 8))
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    x = torch.randn(5)
    j, lam = critic(x)
    loss = j.pow(2).sum() + lam.pow(2).sum()
    opt.zero_grad()
    loss.backward()
    # Every parameter should have a gradient of matching shape.
    for p in critic.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape
    opt.step()


# ---------------------------------------------------------------------------
# IMGDHPAgent
# ---------------------------------------------------------------------------
def _make_agent(n_obs=2, n_action=1, **cfg_kwargs) -> IMGDHPAgent:
    cfg = IMGDHPConfig(
        actor_hidden=(8, 8),
        critic_hidden=(16, 16),
        actor_lr=1e-3,
        critic_lr=1e-3,
        track_Q=[1.0],
        warmup_steps=2,
        forgetting=0.99,
        cov_init=1e2,
        u_max=1.0,
        seed=0,
        **cfg_kwargs,
    )
    return IMGDHPAgent(
        n_obs=n_obs,
        n_action=n_action,
        reference_size=1,
        tracking_indices=[0],
        config=cfg,
    )


def test_agent_predict_returns_bounded_action():
    agent = _make_agent()
    obs = np.array([0.0, 0.0])
    ref = np.zeros((1, 10))
    action = agent.predict(obs, ref, time_step=0)
    assert action.shape == (1,)
    assert abs(action[0]) <= 1.0 + 1e-6


def test_agent_learn_without_predict_raises():
    agent = _make_agent()
    with pytest.raises(RuntimeError):
        agent.learn(np.zeros(2), np.zeros((1, 5)), time_step=0)


def test_agent_full_online_step_updates_all_components():
    """One full predict-learn cycle on a tiny linear plant should update
    the actor, critic, and incremental model without raising, produce
    finite losses after warm-up, and start updating the RLS identifier.

    Convergence of the RLS block is verified separately in the dedicated
    RLS tests — here we do NOT depend on actor exploration being rich
    enough to identify the plant, only on the plumbing being correct.
    """
    agent = _make_agent(exploration_noise_std=0.2)  # excite the plant
    A_true = np.array([[0.9, 0.0], [0.0, 0.8]])
    B_true = np.array([[0.1], [0.2]])
    ref = np.zeros((1, 50))
    y = np.array([0.05, 0.0])

    losses = []
    for t in range(30):
        action = agent.predict(y, ref, time_step=t)
        y_next = A_true @ y + B_true @ action
        metrics = agent.learn(y_next, ref, time_step=t)
        y = y_next
        if t >= agent.cfg.warmup_steps + 1:
            losses.append(metrics["critic_loss"])

    assert agent._total_steps == 30
    assert all(np.isfinite(l) for l in losses), f"non-finite losses: {losses}"
    # RLS should have processed (30 - 1) transitions after the first
    # step warms the history buffers.
    assert agent.incremental_model.num_updates >= 28


def test_agent_train_requires_reference_signal_attr():
    agent = _make_agent()

    class _FakeEnv:
        def reset(self):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 0.0, True, False, {}

    with pytest.raises(AttributeError, match="reference_signal"):
        agent.train(_FakeEnv(), num_episodes=1)


def test_agent_tracking_indices_validation():
    with pytest.raises(ValueError, match="tracking_indices"):
        IMGDHPAgent(n_obs=2, n_action=1, tracking_indices=[])


def test_agent_track_Q_length_mismatch():
    with pytest.raises(ValueError, match="track_Q"):
        IMGDHPAgent(
            n_obs=2,
            n_action=1,
            tracking_indices=[0, 1],
            config=IMGDHPConfig(track_Q=[1.0]),  # length 1, expected 2
        )


def test_target_critic_polyak_update_moves_params_toward_online():
    """After one critic update the target critic should be interpolated
    τ of the way toward the online critic parameters."""
    tau = 0.5
    agent = _make_agent(target_update_tau=tau)
    # Capture initial (identical) params.
    src0 = [p.detach().clone() for p in agent.critic.parameters()]
    tgt0 = [p.detach().clone() for p in agent.target_critic.parameters()]
    for s, t in zip(src0, tgt0):
        assert torch.allclose(s, t)

    # Run one full predict/learn cycle past the warmup so the critic
    # optimiser fires at least once.
    ref = np.zeros((1, 30))
    for t in range(agent.cfg.warmup_steps + 2):
        obs = np.array([0.01, 0.0]) * (t + 1)
        action = agent.predict(obs, ref, t)
        next_obs = obs + 0.001 * np.array([1.0, -0.5]) + 0.0005 * action[0]
        agent.learn(next_obs, ref, t)

    # Online critic has moved; target must have moved *toward* it by τ.
    for p_src, p_tgt, p_tgt0 in zip(
        agent.critic.parameters(), agent.target_critic.parameters(), tgt0
    ):
        moved = (p_tgt.detach() - p_tgt0).norm()
        gap_src = (p_src.detach() - p_tgt0).norm()
        if gap_src.item() > 1e-6:
            # Target should have moved but NOT all the way (unless τ=1).
            assert moved.item() > 0.0
            assert moved.item() <= gap_src.item() + 1e-6


def test_critic_weight_decay_wired_into_optimizer():
    """``IMGDHPConfig.critic_weight_decay`` must reach the critic Adam
    optimiser's parameter group and must not affect the actor's."""
    agent = _make_agent(critic_weight_decay=5e-3)
    assert agent.critic_opt.param_groups[0]["weight_decay"] == pytest.approx(5e-3)
    # Actor optimiser is untouched by the config key.
    assert agent.actor_opt.param_groups[0]["weight_decay"] == pytest.approx(0.0)


def test_target_update_tau_zero_leaves_target_frozen():
    """With ``tau=0`` the target critic must never deviate from its
    initial snapshot, regardless of how many critic updates run."""
    agent = _make_agent(target_update_tau=0.0)
    snap = [p.detach().clone() for p in agent.target_critic.parameters()]
    ref = np.zeros((1, 20))
    for t in range(agent.cfg.warmup_steps + 3):
        obs = np.array([0.01, 0.005]) * (t + 1)
        a = agent.predict(obs, ref, t)
        next_obs = obs + 0.001 * a[0]
        agent.learn(next_obs, ref, t)
    for p_now, p0 in zip(agent.target_critic.parameters(), snap):
        assert torch.allclose(p_now.detach(), p0)
