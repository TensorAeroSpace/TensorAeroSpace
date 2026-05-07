"""Tests for the L2-inner sub-components: SM observer + mode switch."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.inner import ModeSwitcher, SuperTwistingObserver


def test_super_twisting_estimates_step_disturbance() -> None:
    obs = SuperTwistingObserver(n_axes=2, k1=3.0, k2=1.5, dt=0.01)
    nu_des = np.zeros(2)
    omega_dot_meas = np.array([1.5, -0.7])  # constant disturbance
    estimates = []
    for _ in range(2000):
        delta_hat = obs.update(omega_dot_meas, nu_des)
        estimates.append(delta_hat.copy())
    final = estimates[-1]
    target = omega_dot_meas - nu_des
    # Super-twisting converges in finite time; tolerance is generous.
    assert np.linalg.norm(final - target) < 0.3


def test_super_twisting_validates_shapes() -> None:
    obs = SuperTwistingObserver(n_axes=3, dt=0.01)
    with pytest.raises(ValueError):
        obs.update(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        obs.update(np.zeros(3), np.zeros(2))


def test_super_twisting_reset_clears_state() -> None:
    obs = SuperTwistingObserver(n_axes=2, dt=0.01)
    for _ in range(100):
        obs.update(np.array([2.0, 1.0]), np.zeros(2))
    obs.reset()
    out = obs.update(np.zeros(2), np.zeros(2))
    assert np.allclose(out, 0.0, atol=1e-6)


def test_mode_switcher_default_rate_below_threshold() -> None:
    sw = ModeSwitcher(alpha_threshold_deg=25.0, hysteresis_deg=5.0)
    assert sw.select(np.deg2rad(10.0)) == "rate"
    assert sw.select(np.deg2rad(20.0)) == "rate"


def test_mode_switcher_switches_to_angle_above_threshold() -> None:
    sw = ModeSwitcher(alpha_threshold_deg=25.0, hysteresis_deg=5.0)
    assert sw.select(np.deg2rad(30.0)) == "angle"


def test_mode_switcher_hysteresis_holds() -> None:
    sw = ModeSwitcher(alpha_threshold_deg=25.0, hysteresis_deg=5.0)
    sw.select(np.deg2rad(30.0))   # → angle
    # Just below threshold — but inside hysteresis band — stays angle.
    assert sw.select(np.deg2rad(22.0)) == "angle"
    # Below clear-band — back to rate.
    assert sw.select(np.deg2rad(15.0)) == "rate"


def test_mode_switcher_reset() -> None:
    sw = ModeSwitcher()
    sw.select(np.deg2rad(30.0))
    sw.reset()
    # After reset we revert to rate at low alpha without hysteresis carryover.
    assert sw.select(np.deg2rad(22.0)) == "rate"
