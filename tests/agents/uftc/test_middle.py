"""Tests for IADPMiddle: RLS reset triggered by FDD rising edge."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.iadp.model import IADPAgent, IADPConfig
from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.middle import IADPMiddle, RLSResetPolicy


def _make_middle(n_state=2, n_control=2, dt=0.01):
    cfg = IADPConfig(dt=dt, gamma_rls=0.99, phi_init=10.0)
    base = IADPAgent(n_state=n_state, n_control=n_control, config=cfg)
    pol = RLSResetPolicy(
        cov_inflation=100.0, forgetting_drop=0.9, forgetting_recover_steps=100
    )
    return IADPMiddle(base=base, reset_policy=pol)


def _nominal_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=False,
        severity=0.0,
        confidence=0.0,
        innovation_norm=0.0,
        time_since_event=0.0,
    )


def _alarm_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=True,
        severity=1.5,
        confidence=0.78,
        innovation_norm=2.0,
        time_since_event=0.0,
    )


def test_predict_returns_u_iadp_and_omega_ref() -> None:
    m = _make_middle()
    u, omega_ref = m.predict(np.zeros(2), np.zeros(2), time_step=0)
    assert u.shape == (2,)
    assert omega_ref.shape == (2,)


def test_rising_edge_inflates_phi() -> None:
    m = _make_middle()
    rng = np.random.default_rng(0)
    # Drive a few steps to seed Φ and history.
    for k in range(5):
        x = rng.normal(scale=0.1, size=2)
        ref = rng.normal(scale=0.1, size=2)
        m.predict(x, ref, time_step=k)
        m.learn(x + 0.01 * rng.normal(size=2), ref, time_step=k, fdd=_nominal_fdd())
    phi_pre = np.linalg.norm(m.base.rls.Phi)
    m.learn(rng.normal(scale=0.1, size=2), np.zeros(2), time_step=5, fdd=_alarm_fdd())
    phi_post = np.linalg.norm(m.base.rls.Phi)
    assert phi_post > phi_pre


def test_forgetting_recovers_after_drop() -> None:
    m = _make_middle()
    m.learn(np.zeros(2), np.zeros(2), time_step=0, fdd=_alarm_fdd())
    assert m.base.rls.gamma_rls == pytest.approx(0.9, abs=1e-6)
    # Step nominal updates until recovery completes.
    for k in range(101):
        m.learn(np.zeros(2), np.zeros(2), time_step=k + 1, fdd=_nominal_fdd())
    assert m.base.rls.gamma_rls == pytest.approx(0.99, abs=1e-3)


def test_reset_restores_initial_gamma_and_no_recovery_pending() -> None:
    m = _make_middle()
    m.learn(np.zeros(2), np.zeros(2), time_step=0, fdd=_alarm_fdd())
    m.reset()
    assert m.base.rls.gamma_rls == pytest.approx(0.99)
    assert m._recover_countdown == 0


def test_omega_ref_passthrough_when_no_omega_indices() -> None:
    m = _make_middle()
    ref = np.array([0.5, -0.2])
    _, omega_ref = m.predict(np.zeros(2), ref, time_step=0)
    assert np.allclose(omega_ref, ref)


def test_omega_ref_with_omega_indices_uses_state_error_lookahead() -> None:
    cfg = IADPConfig(dt=0.01)
    base = IADPAgent(n_state=2, n_control=2, config=cfg)
    pol = RLSResetPolicy()
    m = IADPMiddle(base=base, reset_policy=pol, omega_indices=[0, 1], lookahead_dt=0.1)
    x = np.array([0.0, 0.0])
    ref = np.array([1.0, -0.5])
    _, omega_ref = m.predict(x, ref, time_step=0)
    expected = (ref - x) / 0.1
    assert np.allclose(omega_ref, expected)
