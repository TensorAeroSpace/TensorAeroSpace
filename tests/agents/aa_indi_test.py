"""Tests for the Active-Adaptive INDI (AA-INDI) agent.

Covers:
    * :class:`VFFRLSEstimator` — convergence, forgetting-factor behaviour,
      argument validation.
    * :class:`LowPassDerivative` and :class:`BiasEstimator` primitives.
    * :class:`AAINDIAgent` — end-to-end closed-loop tracking on a toy
      linear plant, predict/learn cycle, save/load/from_pretrained/
      publish_to_hub round-trip.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi import (
    AAINDIAgent,
    AAINDIConfig,
    BiasEstimator,
    LowPassDerivative,
    VFFRLSEstimator,
)


# ---------------------------------------------------------------------------
# VFF-RLS
# ---------------------------------------------------------------------------
def test_vff_rls_converges_on_known_linear_system():
    """Synthetic identification task: Δy = G_true · Δu with white-noise Δu."""
    rng = np.random.default_rng(0)
    G_true = np.array([[-2.0, 0.1], [0.05, -1.5]], dtype=np.float64)
    rls = VFFRLSEstimator(
        n_y=2, n_u=2,
        forgetting_min=0.8, forgetting_max=0.9999,
        eps_sensitivity=5.0, cov_init=1e3, seed=0,
    )
    for _ in range(500):
        du = rng.normal(size=2) * 0.5
        dy = G_true @ du + rng.normal(size=2) * 1e-3
        rls.update(du, dy)
    np.testing.assert_allclose(rls.G, G_true, atol=5e-2)


def test_vff_rls_lambda_drops_on_large_residuals():
    """Large residuals should pull the forgetting factor toward the lower bound.

    Only the very first update sees the full residual; by the second call the
    linear estimator has already nearly fit it, so we assert the contraction
    on the initial tick.
    """
    rls = VFFRLSEstimator(
        n_y=1, n_u=1,
        forgetting_min=0.5, forgetting_max=0.999,
        eps_sensitivity=0.1,
    )
    rls.update(np.array([1.0]), np.array([10.0]))
    assert rls.last_lambda <= 0.8
    # After the residual collapses, λ should relax back toward the upper bound.
    for _ in range(20):
        rls.update(np.array([1.0]), np.array([10.0]))
    assert rls.last_lambda >= 0.9


def test_vff_rls_validates_inputs():
    with pytest.raises(ValueError, match="forgetting"):
        VFFRLSEstimator(n_y=1, n_u=1, forgetting_min=1.5, forgetting_max=1.0)
    with pytest.raises(ValueError, match="eps_sensitivity"):
        VFFRLSEstimator(n_y=1, n_u=1, eps_sensitivity=0.0)
    rls = VFFRLSEstimator(n_y=2, n_u=2)
    with pytest.raises(ValueError, match="du"):
        rls.update(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="dy"):
        rls.update(np.zeros(2), np.zeros(3))


def test_vff_rls_reset_covariance_restores_init_scale():
    rls = VFFRLSEstimator(n_y=1, n_u=1, cov_init=50.0)
    rls.update(np.array([1.0]), np.array([0.5]))
    assert not np.allclose(rls.P, 50.0 * np.eye(1))
    rls.reset_covariance()
    np.testing.assert_allclose(rls.P, 50.0 * np.eye(1))


def test_vff_rls_predict_uses_latest_theta():
    rls = VFFRLSEstimator(n_y=1, n_u=1)
    rls.theta = np.array([[0.3]])
    np.testing.assert_allclose(rls.predict(np.array([2.0])), [0.6])


# ---------------------------------------------------------------------------
# Sensor filter primitives
# ---------------------------------------------------------------------------
def test_low_pass_derivative_on_ramp_returns_slope_eventually():
    """For x(t) = a·t + b, the derivative should converge to a."""
    d = LowPassDerivative(n=1, dt=0.01, cutoff_hz=20.0)
    out = None
    for k in range(200):
        out = d.step(np.array([2.0 * k * 0.01 + 1.0]))
    # After warm-up the filter has tracked the constant slope.
    assert abs(float(out[0]) - 2.0) < 1e-2


def test_low_pass_derivative_first_call_returns_zero():
    d = LowPassDerivative(n=3, dt=0.01, cutoff_hz=10.0)
    out = d.step(np.array([5.0, -1.0, 2.0]))
    np.testing.assert_allclose(out, np.zeros(3))


def test_low_pass_derivative_reset_clears_state():
    d = LowPassDerivative(n=1, dt=0.01, cutoff_hz=50.0)
    for _ in range(10):
        d.step(np.array([1.0]))
    d.reset()
    out = d.step(np.array([1.0]))
    np.testing.assert_allclose(out, np.zeros(1))


def test_low_pass_derivative_validates_inputs():
    with pytest.raises(ValueError, match="dt"):
        LowPassDerivative(n=1, dt=0.0)
    with pytest.raises(ValueError, match="cutoff_hz"):
        LowPassDerivative(n=1, dt=0.01, cutoff_hz=-1)
    d = LowPassDerivative(n=2, dt=0.01)
    with pytest.raises(ValueError, match="length"):
        d.step(np.array([1.0]))


def test_bias_estimator_tracks_constant_innovation():
    b = BiasEstimator(n=2, forgetting=0.9)
    for _ in range(200):
        b.update(np.array([0.1, -0.2]))
    np.testing.assert_allclose(b.bias, [0.1, -0.2], atol=1e-3)


def test_bias_estimator_validates_inputs():
    with pytest.raises(ValueError, match="forgetting"):
        BiasEstimator(n=1, forgetting=1.5)
    b = BiasEstimator(n=2)
    with pytest.raises(ValueError, match="length"):
        b.update(np.array([1.0]))


# ---------------------------------------------------------------------------
# AAINDIAgent — closed-loop behaviour
# ---------------------------------------------------------------------------
def _toy_plant():
    """Stable 3×3 linear plant used for closed-loop tests."""
    return np.array(
        [[-2.0, 0.1, 0.0], [0.05, -1.5, 0.2], [0.0, 0.05, -0.9]],
        dtype=np.float64,
    )


def _run_closed_loop(agent, G_true, ref, n_steps=500):
    omega = np.zeros(3)
    for k in range(n_steps):
        u = agent.predict(omega, ref, k)
        omega = omega + agent.cfg.dt * (G_true @ u)
        agent.learn(omega, ref, k)
    return omega


def _mk_agent(**cfg_overrides) -> AAINDIAgent:
    defaults = dict(
        dt=0.01, ref_wn=5.0, ref_zeta=0.7,
        u_magnitude_limit=25.0, u_rate_limit=200.0,
        vff_forgetting_min=0.9, vff_forgetting_max=0.999,
        vff_eps_sensitivity=2.0, vff_cov_init=10.0,
        sensor_cutoff_hz=50.0, bias_forgetting=0.995,
        enable_bias_correction=False, seed=0,
    )
    defaults.update(cfg_overrides)
    return AAINDIAgent(n_state=3, n_control=3, config=AAINDIConfig(**defaults))


def test_aaindi_tracks_constant_reference_with_warm_start():
    G_true = _toy_plant()
    agent = _mk_agent(G_init=G_true)
    omega = _run_closed_loop(agent, G_true, np.array([0.2, -0.1, 0.05]), n_steps=500)
    assert float(np.linalg.norm(omega - np.array([0.2, -0.1, 0.05]))) < 5e-3


def test_aaindi_reference_can_be_2d_schedule():
    G_true = _toy_plant()
    agent = _mk_agent(G_init=G_true)
    schedule = np.stack(
        [np.linspace(0.0, 0.2, 100), np.zeros(100), np.zeros(100)], axis=0
    )
    omega = np.zeros(3)
    for k in range(100):
        u = agent.predict(omega, schedule, k)
        omega = omega + agent.cfg.dt * (G_true @ u)
        agent.learn(omega, schedule, k)
    # Channel 0 has climbed, channels 1-2 remain near zero.
    assert omega[0] > 0.1
    assert abs(omega[1]) < 0.1 and abs(omega[2]) < 0.1


def test_aaindi_g_init_shape_mismatch_raises():
    with pytest.raises(ValueError, match="G_init"):
        AAINDIAgent(
            n_state=3, n_control=2,
            config=AAINDIConfig(G_init=np.zeros((3, 3))),
        )


def test_aaindi_reset_clears_rolling_state():
    agent = _mk_agent(G_init=_toy_plant())
    _run_closed_loop(agent, _toy_plant(), np.array([0.1, 0, 0]), n_steps=10)
    agent.reset()
    assert agent._omega_prev is None
    assert agent._omega_dot_prev is None
    np.testing.assert_allclose(agent._u_prev, np.zeros(3))
    np.testing.assert_allclose(agent._ref_rate, np.zeros(3))


def test_aaindi_learn_reports_metrics():
    agent = _mk_agent(G_init=_toy_plant())
    G_true = _toy_plant()
    ref = np.array([0.1, 0.0, 0.0])
    u = agent.predict(np.zeros(3), ref, 0)
    omega = agent.cfg.dt * (G_true @ u)
    metrics = agent.learn(omega, ref, 0)
    for key in ("rls_pred_error_norm", "vff_lambda", "G_norm", "bias_norm"):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_aaindi_control_respects_rate_and_magnitude_limits():
    # A tiny rate limit must cap ||Δu|| per step, and the magnitude limit
    # clamps the cumulative command.
    agent = _mk_agent(
        G_init=_toy_plant(),
        u_rate_limit=0.5,  # ⇒ Δu ≤ 0.005 per step
        u_magnitude_limit=0.3,
    )
    G_true = _toy_plant()
    omega = _run_closed_loop(agent, G_true, np.array([10.0, 0, 0]), n_steps=300)
    # u is constrained far below what a step of 10 rad/s would demand —
    # the plant can't reach the reference, and the commanded u stays
    # inside the hard bound.
    assert np.max(np.abs(agent._u_prev)) <= 0.3 + 1e-6
    assert float(np.linalg.norm(omega)) < 10.0


def test_aaindi_invalid_reference_shape_raises():
    agent = _mk_agent(G_init=_toy_plant())
    with pytest.raises(ValueError, match="reference"):
        agent.predict(np.zeros(3), np.zeros((3, 3, 3)), 0)


def test_aaindi_invalid_omega_shape_raises():
    agent = _mk_agent(G_init=_toy_plant())
    with pytest.raises(ValueError, match="omega"):
        agent.predict(np.zeros(2), np.zeros(3), 0)


def test_aaindi_bias_correction_branch_runs():
    agent = _mk_agent(G_init=_toy_plant(), enable_bias_correction=True)
    G_true = _toy_plant()
    omega = _run_closed_loop(agent, G_true, np.array([0.1, 0, 0]), n_steps=30)
    # Running with correction enabled must not blow up.
    assert np.all(np.isfinite(omega))


# ---------------------------------------------------------------------------
# Save / load / from_pretrained / publish_to_hub
# ---------------------------------------------------------------------------
def test_aaindi_save_creates_expected_files(tmp_path):
    agent = _mk_agent()
    run_dir = Path(agent.save(path=tmp_path))
    for name in ("config.json", "vff_rls.npz", "bias_state.npz", "deriv_state.npz"):
        assert (run_dir / name).exists(), f"missing {name}"


def test_aaindi_round_trip_preserves_state(tmp_path):
    agent = _mk_agent(G_init=_toy_plant())
    G_true = _toy_plant()
    _run_closed_loop(agent, G_true, np.array([0.2, -0.1, 0.05]), n_steps=50)

    run_dir = agent.save(path=tmp_path)
    restored = AAINDIAgent.from_pretrained(run_dir)

    np.testing.assert_allclose(restored.rls.theta, agent.rls.theta)
    np.testing.assert_allclose(restored.rls.P, agent.rls.P)
    assert restored.rls.num_updates == agent.rls.num_updates
    np.testing.assert_allclose(restored.bias_est.bias, agent.bias_est.bias)
    np.testing.assert_allclose(restored.deriv.last_output, agent.deriv.last_output)


def test_aaindi_from_pretrained_missing_local_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local directory not found"):
        AAINDIAgent.from_pretrained(str(tmp_path / "nope"))


def test_aaindi_from_pretrained_snapshot_download(tmp_path, monkeypatch):
    agent = _mk_agent()
    run_dir = agent.save(path=tmp_path)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.snapshot_download = lambda repo_id, token=None, revision=None: run_dir
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    restored = AAINDIAgent.from_pretrained("user/aaindi-demo")
    assert isinstance(restored, AAINDIAgent)


def test_aaindi_save_includes_loop_state(tmp_path):
    """``save()`` must persist reference-model + PI integrator state so a
    mid-episode checkpoint replays the same control law on reload."""
    agent = _mk_agent(G_init=_toy_plant(), ref_error_kp=0.6, ref_error_ki=0.2)
    G_true = _toy_plant()
    _run_closed_loop(agent, G_true, np.array([0.2, -0.1, 0.05]), n_steps=25)

    # Preserve the mid-episode state.
    ref_rate_before = agent._ref_rate.copy()
    int_err_before = agent._int_err.copy()
    u_prev_before = agent._u_prev.copy()
    omega_dot_cached_before = agent._omega_dot_cached.copy()

    run_dir = agent.save(path=tmp_path)
    restored = AAINDIAgent.from_pretrained(run_dir)

    np.testing.assert_allclose(restored._ref_rate, ref_rate_before)
    np.testing.assert_allclose(restored._int_err, int_err_before)
    np.testing.assert_allclose(restored._u_prev, u_prev_before)
    np.testing.assert_allclose(restored._omega_dot_cached, omega_dot_cached_before)


def test_aaindi_round_trip_behaviourally_identical_with_pi(tmp_path):
    """After save/load mid-episode with PI feedback active, the next
    ``predict`` call must produce the same control command."""
    agent = _mk_agent(G_init=_toy_plant(), ref_error_kp=0.6, ref_error_ki=0.2)
    G_true = _toy_plant()
    ref = np.array([0.2, -0.1, 0.05])
    _run_closed_loop(agent, G_true, ref, n_steps=30)

    # Capture the next ω measurement the "live" agent would see.
    live_obs = np.random.RandomState(0).randn(3) * 0.01

    run_dir = agent.save(path=tmp_path)
    restored = AAINDIAgent.from_pretrained(run_dir)

    u_live = agent.predict(live_obs, ref, 30)
    u_restored = restored.predict(live_obs, ref, 30)
    np.testing.assert_allclose(u_live, u_restored, atol=1e-10)


def test_aaindi_publish_to_hub_delegates_to_upload_folder(tmp_path, monkeypatch):
    agent = _mk_agent()
    run_dir = agent.save(path=tmp_path)

    called = {}

    class _FakeApi:
        def upload_folder(self, **kwargs):
            called.update(kwargs)

    fake_mod = types.ModuleType("huggingface_hub")
    fake_mod.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_mod)

    agent.publish_to_hub("me/aaindi", folder_path=run_dir, access_token="tok")
    assert called["repo_id"] == "me/aaindi"
    assert called["repo_type"] == "model"
    assert called["token"] == "tok"
