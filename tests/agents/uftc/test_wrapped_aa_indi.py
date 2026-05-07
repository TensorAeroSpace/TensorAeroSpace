"""Tests for WrappedAAINDI bounded trust-region wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.aa_indi.model import AAINDIAgent, AAINDIConfig
from tensoraerospace.agent.uftc.inner import (
    ModeSwitcher,
    SuperTwistingObserver,
    WrappedAAINDI,
)


def _make_wrapped(n_state=3, n_control=3, dt=0.01):
    cfg = AAINDIConfig(
        dt=dt,
        u_magnitude_limit=1.0,
        u_rate_limit=10.0,
        G_init=np.eye(n_state, n_control) * 0.5,
    )
    base = AAINDIAgent(n_state=n_state, n_control=n_control, config=cfg)
    sm = SuperTwistingObserver(n_axes=n_state, dt=dt)
    sw = ModeSwitcher()
    return WrappedAAINDI(base=base, sm_obs=sm, mode_switch=sw,
                         trust_radius_nominal=0.05,
                         trust_radius_fault=0.5,
                         dt=dt)


def test_predict_returns_correct_shape() -> None:
    w = _make_wrapped()
    out = w.predict(
        omega_ref=np.zeros(3),
        omega_meas=np.zeros(3),
        alpha=0.0,
        u_blend_target=np.zeros(3),
        fault_severity=0.0,
        time_step=0,
    )
    assert out.shape == (3,)


def test_trust_region_clips_to_target_under_nominal() -> None:
    w = _make_wrapped()
    # Force a large omega_ref to drive INDI hard.
    out = w.predict(
        omega_ref=np.array([5.0, 0.0, 0.0]),
        omega_meas=np.zeros(3),
        alpha=0.0,
        u_blend_target=np.array([0.2, 0.0, 0.0]),
        fault_severity=0.0,
        time_step=0,
    )
    # Distance from target is bounded by trust_radius_nominal=0.05.
    assert np.linalg.norm(out - np.array([0.2, 0.0, 0.0])) <= 0.05 + 1e-9


def test_trust_region_expands_under_fault_severity() -> None:
    w = _make_wrapped()
    out = w.predict(
        omega_ref=np.array([5.0, 0.0, 0.0]),
        omega_meas=np.zeros(3),
        alpha=0.0,
        u_blend_target=np.array([0.2, 0.0, 0.0]),
        fault_severity=1.0,
        time_step=0,
    )
    assert np.linalg.norm(out - np.array([0.2, 0.0, 0.0])) <= 0.5 + 1e-9


def test_predict_then_learn_round_trip() -> None:
    w = _make_wrapped()
    rng = np.random.default_rng(0)
    u_blend_target = np.zeros(3)
    for k in range(50):
        omega = rng.normal(scale=0.1, size=3)
        ref = rng.normal(scale=0.1, size=3)
        u = w.predict(
            omega_ref=ref, omega_meas=omega, alpha=0.0,
            u_blend_target=u_blend_target,
            fault_severity=0.0, time_step=k,
        )
        u_blend_target = u
        next_omega = omega + 0.01 * rng.normal(size=3)
        w.learn(next_omega, ref, time_step=k)


def test_reset_clears_substate() -> None:
    w = _make_wrapped()
    rng = np.random.default_rng(1)
    for k in range(10):
        w.predict(omega_ref=rng.normal(size=3), omega_meas=rng.normal(size=3),
                  alpha=0.0, u_blend_target=np.zeros(3),
                  fault_severity=0.0, time_step=k)
    w.reset()
    # After reset the SM observer state should be zero.
    assert np.allclose(w.sm_obs._s, 0.0)
    assert np.allclose(w.sm_obs._z, 0.0)


def test_validates_shape_mismatches() -> None:
    w = _make_wrapped()
    with pytest.raises(ValueError):
        w.predict(
            omega_ref=np.zeros(2),  # wrong size
            omega_meas=np.zeros(3),
            alpha=0.0,
            u_blend_target=np.zeros(3),
            fault_severity=0.0,
            time_step=0,
        )
