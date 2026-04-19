"""Tests for the Incremental Approximate Dynamic Programming (iADP) agent.

Covers:
    * :class:`IncrementalRLS` — convergence, predict/update validation.
    * :class:`IADPAgent` — end-to-end predict/learn cycle on a toy
      linear plant, reference slicing, validation, rate/magnitude
      limiting, model-learning open-loop phase.
    * Persistence: save/load/from_pretrained/publish_to_hub round-trip
      including the rolling loop state and the policy-evaluation
      window.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tensoraerospace.agent.iadp import IADPAgent, IADPConfig, IncrementalRLS


# ---------------------------------------------------------------------------
# IncrementalRLS
# ---------------------------------------------------------------------------
def test_incremental_rls_converges_on_known_linear_system():
    """Synthetic ΔX_{t+1} = F·ΔX_t + G·Δu with white-noise inputs."""
    rng = np.random.default_rng(0)
    F_true = np.array([[0.9, 0.05], [0.0, 0.8]], dtype=np.float64)
    G_true = np.array([[-0.5], [0.1]], dtype=np.float64)
    rls = IncrementalRLS(n_output=2, n_regressor=3, gamma_rls=0.999, phi_init=1e3)
    for _ in range(1500):
        dX = rng.normal(size=2) * 0.3
        du = rng.normal(size=1) * 0.3
        W = np.concatenate([dX, du])
        y = F_true @ dX + G_true @ du + rng.normal(size=2) * 1e-4
        rls.update(W, y)

    F_hat = rls.extract_F(n_state=2)
    G_hat = rls.extract_G(n_state=2)
    np.testing.assert_allclose(F_hat, F_true, atol=5e-2)
    np.testing.assert_allclose(G_hat, G_true, atol=5e-2)


def test_incremental_rls_predict_matches_parameters():
    rls = IncrementalRLS(n_output=2, n_regressor=3)
    rls.theta = np.array([[1.0, 0.5], [0.0, -1.0], [2.0, 1.5]], dtype=np.float64)
    W = np.array([0.1, 0.2, -0.3])
    # ΔX̂ = Θ^T W = [(1, 0, 2)·W; (0.5, -1, 1.5)·W]
    expected = np.array(
        [1.0 * 0.1 + 0.0 * 0.2 + 2.0 * -0.3, 0.5 * 0.1 + -1.0 * 0.2 + 1.5 * -0.3]
    )
    np.testing.assert_allclose(rls.predict(W), expected)


def test_incremental_rls_validates_inputs():
    with pytest.raises(ValueError, match="gamma_rls"):
        IncrementalRLS(n_output=1, n_regressor=1, gamma_rls=1.1)
    with pytest.raises(ValueError, match="phi_init"):
        IncrementalRLS(n_output=1, n_regressor=1, phi_init=0.0)
    rls = IncrementalRLS(n_output=2, n_regressor=3)
    with pytest.raises(ValueError, match="W"):
        rls.update(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="y"):
        rls.update(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="W"):
        rls.predict(np.zeros(2))


def test_incremental_rls_reset_covariance():
    rls = IncrementalRLS(n_output=1, n_regressor=2, phi_init=10.0)
    for _ in range(5):
        rls.update(np.array([1.0, 0.5]), np.array([0.1]))
    before = rls.Phi.copy()
    rls.reset_covariance()
    assert not np.allclose(before, rls.Phi)
    np.testing.assert_allclose(rls.Phi, 10.0 * np.eye(2))


# ---------------------------------------------------------------------------
# IADPAgent helpers
# ---------------------------------------------------------------------------
def _mk_agent(**cfg_kwargs) -> IADPAgent:
    defaults = dict(
        dt=0.01,
        gamma=0.8,
        gamma_rls=0.995,
        Q=np.array([[10.0]]),
        R=np.array([[0.1]]),
        policy_eval_window=60,
        policy_eval_every=25,
        policy_eval_warmup_updates=10,
        G_init=np.array([[-0.5], [0.0]]),
        F_init=np.eye(2),
        P_init=np.eye(2),
        u_magnitude_limit=5.0,
        u_rate_limit=30.0,
        seed=0,
    )
    defaults.update(cfg_kwargs)
    cfg = IADPConfig(**defaults)
    return IADPAgent(n_state=1, n_control=1, config=cfg)


def _toy_scalar_plant(x: float, u: float, dt: float = 0.01) -> float:
    """First-order: x_{t+1} = x_t + dt*(G·u - a·x)."""
    return x + dt * (-0.5 * u - 2.0 * x)


def _run_closed_loop(agent: IADPAgent, ref_value: float, n_steps: int) -> np.ndarray:
    x = 0.0
    traj = np.zeros(n_steps)
    for k in range(n_steps):
        ref = np.array([ref_value])
        u = agent.predict(np.array([x]), ref, k)
        x = _toy_scalar_plant(x, float(u[0]), agent.cfg.dt)
        agent.learn(np.array([x]), ref, k)
        traj[k] = x
    return traj


# ---------------------------------------------------------------------------
# Agent lifecycle
# ---------------------------------------------------------------------------
def test_agent_dimensions_and_defaults():
    agent = IADPAgent(n_state=2, n_control=3)
    assert agent.n_state == 2
    assert agent.n_control == 3
    assert agent.n_aug == 4
    # Default Q, R are identity of the right size.
    np.testing.assert_allclose(agent.Q, np.eye(2))
    np.testing.assert_allclose(agent.R, np.eye(3))
    # Default P is identity.
    np.testing.assert_allclose(agent.P, np.eye(4))
    # RLS theta has the right shape.
    assert agent.rls.theta.shape == (4 + 3, 4)


def test_agent_predict_returns_control_of_right_shape():
    agent = _mk_agent()
    u = agent.predict(np.array([0.1]), np.array([0.0]), 0)
    assert u.shape == (1,)
    assert np.isfinite(u).all()


def test_agent_rejects_wrong_state_shape():
    agent = _mk_agent()
    with pytest.raises(ValueError, match="x_obs"):
        agent.predict(np.zeros(2), np.array([0.0]), 0)


def test_agent_rejects_3d_reference():
    agent = _mk_agent()
    with pytest.raises(ValueError, match="reference"):
        agent.predict(np.array([0.0]), np.zeros((1, 1, 1)), 0)


def test_agent_learn_rejects_wrong_state_shape():
    agent = _mk_agent()
    agent.predict(np.array([0.0]), np.array([0.0]), 0)
    with pytest.raises(ValueError, match="next_x_obs"):
        agent.learn(np.zeros(2), np.array([0.0]), 0)


def test_agent_q_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Q"):
        IADPAgent(
            n_state=2,
            n_control=1,
            config=IADPConfig(Q=np.zeros((3, 3))),
        )


def test_agent_r_shape_mismatch_raises():
    with pytest.raises(ValueError, match="R"):
        IADPAgent(
            n_state=1,
            n_control=2,
            config=IADPConfig(R=np.zeros((3, 3))),
        )


def test_agent_g_init_shape_mismatch_raises():
    with pytest.raises(ValueError, match="G_init"):
        IADPAgent(
            n_state=1,
            n_control=1,
            config=IADPConfig(G_init=np.zeros((3, 3))),
        )


def test_agent_f_init_shape_mismatch_raises():
    with pytest.raises(ValueError, match="F_init"):
        IADPAgent(
            n_state=1,
            n_control=1,
            config=IADPConfig(F_init=np.zeros((3, 3))),
        )


def test_agent_p_init_shape_mismatch_raises():
    with pytest.raises(ValueError, match="P_init"):
        IADPAgent(
            n_state=1,
            n_control=1,
            config=IADPConfig(P_init=np.zeros((3, 3))),
        )


def test_agent_closed_loop_runs_without_error():
    agent = _mk_agent()
    traj = _run_closed_loop(agent, ref_value=0.2, n_steps=80)
    assert np.isfinite(traj).all()


def test_agent_learn_reports_metrics():
    agent = _mk_agent()
    u = agent.predict(np.array([0.0]), np.array([0.1]), 0)
    x_next = _toy_scalar_plant(0.0, float(u[0]))
    metrics = agent.learn(np.array([x_next]), np.array([0.1]), 0)
    for key in ("rls_pred_error_norm", "cost", "F_norm", "G_norm", "P_norm"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_agent_reference_can_be_scalar_or_schedule():
    agent = _mk_agent()
    # 0-D
    agent.predict(np.array([0.0]), np.array(0.1), 0)
    # (T,) schedule
    ref_schedule = np.linspace(0.0, 0.2, 50)
    agent.predict(np.array([0.0]), ref_schedule, 10)
    # (n_state, T) schedule
    ref_2d = ref_schedule[None, :]
    agent.predict(np.array([0.0]), ref_2d, 10)


def test_agent_rate_and_magnitude_limits_enforced():
    agent = _mk_agent(u_rate_limit=1.0, u_magnitude_limit=0.05)
    # Large reference demands large control; limits must cap it.
    for k in range(20):
        u = agent.predict(np.array([0.0]), np.array([5.0]), k)
        assert abs(float(u[0])) <= 0.05 + 1e-9
        _ = agent.learn(np.array([0.0]), np.array([5.0]), k)


def test_agent_reset_clears_rolling_state():
    agent = _mk_agent()
    _run_closed_loop(agent, ref_value=0.1, n_steps=10)
    agent.reset()
    assert agent._X_prev is None
    assert agent._last_X is None
    np.testing.assert_allclose(agent._delta_prev, 0.0)
    np.testing.assert_allclose(agent._last_dX, 0.0)
    assert agent._step == 0
    assert len(agent._window) == 0


def test_agent_model_learning_phase_uses_excitation():
    """During model_learning_only_steps, the policy must stay in open
    loop and output the excitation signal instead of eq. (11)."""
    exc = 0.4 * np.sin(2 * np.pi * 0.8 * np.arange(10) * 0.01)
    agent = _mk_agent(
        model_learning_only_steps=10,
        excitation_signal=exc,
    )
    for k in range(10):
        u = agent.predict(np.array([0.0]), np.array([0.0]), k)
        # Excitation is absolute, so u should match exc[k] modulo rate
        # limiting (but the rate limit is 30 ⇒ 0.3 per step, large
        # enough here).
        assert abs(float(u[0]) - float(exc[k])) < 1e-6
        _ = agent.learn(np.array([0.0]), np.array([0.0]), k)


def test_agent_model_learning_phase_default_zero_output():
    """No excitation given ⇒ zero control output for the open-loop window."""
    agent = _mk_agent(model_learning_only_steps=5)
    for k in range(5):
        u = agent.predict(np.array([0.0]), np.array([0.1]), k)
        np.testing.assert_allclose(u, 0.0)
        _ = agent.learn(np.array([0.0]), np.array([0.1]), k)


def test_agent_policy_evaluation_runs_on_schedule():
    """P̃ must be updated after the warm-up and stride conditions are met."""
    agent = _mk_agent(
        policy_eval_warmup_updates=5,
        policy_eval_every=10,
        P_init=np.eye(2) * 3.0,
    )
    P_before = agent.P.copy()
    for k in range(40):
        agent.predict(np.array([0.0]), np.array([0.1]), k)
        x_next = _toy_scalar_plant(0.0, 0.0) + 0.001 * k
        agent.learn(np.array([x_next]), np.array([0.1]), k)
    # P should have moved off its initial value by now.
    assert not np.allclose(P_before, agent.P)


def test_agent_policy_eval_blend_smooths_p_trajectory():
    """With ``policy_eval_blend < 1`` the kernel matrix ``P̃`` should
    change more gradually between successive policy-evaluation ticks
    than with the default ``blend = 1``. We run two identical rollouts
    (same seed, same state sequence) and compare the step-to-step jump
    in ``‖P̃‖`` over policy updates."""

    def track_p_jumps(blend):
        agent = _mk_agent(
            policy_eval_warmup_updates=5,
            policy_eval_every=5,
            policy_eval_blend=blend,
            P_init=np.eye(2) * 3.0,
        )
        p_norms = []
        for k in range(80):
            agent.predict(np.array([0.0]), np.array([0.1]), k)
            x_next = _toy_scalar_plant(0.0, 0.0) + 0.001 * k
            agent.learn(np.array([x_next]), np.array([0.1]), k)
            p_norms.append(float(np.linalg.norm(agent.P)))
        diffs = np.abs(np.diff(p_norms))
        return diffs.max()

    hard_update = track_p_jumps(1.0)
    soft_update = track_p_jumps(0.2)
    # Soft update must strictly reduce the largest per-tick ||P̃|| jump.
    assert soft_update < hard_update, (
        f"soft blend did not smooth updates: hard={hard_update:.3f}, "
        f"soft={soft_update:.3f}"
    )


def test_agent_policy_eval_blend_validates_range():
    """The blend coefficient is clipped to ``[0, 1]``; values outside that
    range should be accepted (clipped) without raising, and ``blend = 0``
    should freeze ``P̃`` at its initial value."""
    agent = _mk_agent(
        policy_eval_warmup_updates=3,
        policy_eval_every=5,
        policy_eval_blend=0.0,  # freeze P̃
        P_init=np.eye(2) * 7.0,
    )
    P_before = agent.P.copy()
    for k in range(40):
        agent.predict(np.array([0.0]), np.array([0.1]), k)
        x_next = _toy_scalar_plant(0.0, 0.0) + 0.001 * k
        agent.learn(np.array([x_next]), np.array([0.1]), k)
    np.testing.assert_allclose(agent.P, P_before)


def test_agent_p_stays_psd_with_enforcement():
    agent = _mk_agent(enforce_psd=True, psd_floor=1e-5)
    _run_closed_loop(agent, ref_value=0.2, n_steps=100)
    eigs = np.linalg.eigvalsh(agent.P)
    assert eigs.min() >= 1e-5 - 1e-9


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_agent_save_creates_expected_files(tmp_path):
    agent = _mk_agent()
    _run_closed_loop(agent, 0.1, n_steps=30)
    run_dir = Path(agent.save(path=tmp_path))
    for name in (
        "config.json",
        "rls.npz",
        "value.npz",
        "weights.npz",
        "loop_state.npz",
        "window.npz",
    ):
        assert (run_dir / name).exists(), f"missing {name}"


def test_agent_save_config_roundtrips_arrays(tmp_path):
    agent = _mk_agent(
        F_init=np.array([[0.9, 0.1], [0.0, 1.0]]),
        P_init=np.array([[2.0, -1.0], [-1.0, 3.0]]),
    )
    run_dir = Path(agent.save(path=tmp_path))
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    stored_cfg = cfg["policy"]["config"]
    assert isinstance(stored_cfg["Q"], list)
    assert isinstance(stored_cfg["R"], list)
    assert isinstance(stored_cfg["F_init"], list)
    assert isinstance(stored_cfg["G_init"], list)
    assert isinstance(stored_cfg["P_init"], list)


def test_agent_round_trip_preserves_state(tmp_path):
    agent = _mk_agent()
    _run_closed_loop(agent, ref_value=0.2, n_steps=50)
    run_dir = agent.save(path=tmp_path)
    restored = IADPAgent.from_pretrained(run_dir)

    np.testing.assert_allclose(restored.rls.theta, agent.rls.theta)
    np.testing.assert_allclose(restored.rls.Phi, agent.rls.Phi)
    assert restored.rls.num_updates == agent.rls.num_updates
    np.testing.assert_allclose(restored.P, agent.P)
    np.testing.assert_allclose(restored.Q, agent.Q)
    np.testing.assert_allclose(restored.R, agent.R)
    np.testing.assert_allclose(restored._delta_prev, agent._delta_prev)
    assert restored._step == agent._step
    assert len(restored._window) == len(agent._window)


def test_agent_mid_episode_save_produces_bitidentical_next_control(tmp_path):
    """Save mid-episode, reload, feed the same next measurement: the
    commanded control must match the live agent bit-for-bit."""
    agent = _mk_agent()
    _run_closed_loop(agent, ref_value=0.2, n_steps=40)

    run_dir = agent.save(path=tmp_path)
    restored = IADPAgent.from_pretrained(run_dir)

    live_obs = np.array([0.05])
    ref = np.array([0.2])
    u_live = agent.predict(live_obs, ref, 40)
    u_restored = restored.predict(live_obs, ref, 40)
    np.testing.assert_allclose(u_live, u_restored, atol=1e-12)


def test_agent_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        IADPAgent.from_pretrained(str(tmp_path / "nope"))


def test_agent_from_pretrained_snapshot_download(tmp_path, monkeypatch):
    agent = _mk_agent()
    run_dir = agent.save(path=tmp_path)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.snapshot_download = lambda repo_id, token=None, revision=None: run_dir
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    restored = IADPAgent.from_pretrained("user/iadp-demo")
    assert isinstance(restored, IADPAgent)
    np.testing.assert_allclose(restored.P, agent.P)


def test_agent_publish_to_hub_delegates_to_upload_folder(tmp_path, monkeypatch):
    agent = _mk_agent()
    run_dir = agent.save(path=tmp_path)

    called = {}

    class _FakeApi:
        def upload_folder(self, **kwargs):
            called.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    agent.publish_to_hub("me/iadp", folder_path=run_dir, access_token="tok")
    assert called["repo_id"] == "me/iadp"
    assert called["repo_type"] == "model"
    assert called["token"] == "tok"


def test_agent_load_handles_empty_window(tmp_path):
    """An agent saved before any learn() call has an empty window on
    disk; load must succeed and the window must be empty."""
    agent = _mk_agent()
    run_dir = agent.save(path=tmp_path)
    restored = IADPAgent.from_pretrained(run_dir)
    assert len(restored._window) == 0
