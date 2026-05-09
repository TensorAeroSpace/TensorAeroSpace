"""Tests for the adaptive 3-step Kalman filter used by UFTC FDD."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.fdd.kalman_3step import KalmanStep, NominalKalman


def _step_plant(x, u, F, G, sigma=0.0, rng=None):
    nx = x + F @ x + G @ u  # discrete linear plant: x_{t+1} = x_t + F·x_t + G·u_t
    if sigma > 0.0:
        nx = nx + rng.normal(0.0, sigma, size=nx.shape)
    return nx


def test_kalman_returns_kalman_step_namedtuple_like() -> None:
    F = np.zeros((2, 2))
    G = np.eye(2)
    kf = NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2) * 1e-3, R=np.eye(2) * 1e-2)
    out = kf.step(np.zeros(2), np.zeros(2))
    assert isinstance(out, KalmanStep)
    assert out.x_hat.shape == (2,)
    assert out.nu.shape == (2,)
    assert out.S.shape == (2, 2)
    assert out.K.shape == (2, 2)


def test_kalman_tracks_linear_plant_within_tolerance() -> None:
    rng = np.random.default_rng(0)
    F = np.array([[0.05, 0.0], [0.0, -0.03]])
    G = np.array([[0.1, 0.0], [0.0, 0.2]])
    kf = NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2) * 1e-4, R=np.eye(2) * 1e-2)
    x = np.zeros(2)
    errs = []
    for _ in range(500):
        u = rng.normal(0.0, 1.0, size=2)
        x = _step_plant(x, u, F, G, sigma=0.05, rng=rng)
        out = kf.step(x.copy(), u.copy())
        errs.append(float(np.linalg.norm(out.x_hat - x)))
    # After warm-up the filter tracks the noisy plant within a few sigma.
    assert np.mean(errs[-200:]) < 0.5


def test_kalman_innovation_zero_mean_under_nominal() -> None:
    rng = np.random.default_rng(1)
    F = np.zeros((2, 2))
    G = np.eye(2)
    kf = NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2) * 1e-3, R=np.eye(2) * 1e-2)
    x = np.zeros(2)
    nus = []
    for _ in range(1000):
        u = rng.normal(0.0, 0.5, size=2)
        x = _step_plant(x, u, F, G, sigma=0.05, rng=rng)
        out = kf.step(x, u)
        nus.append(out.nu.copy())
    nus = np.stack(nus[200:])  # post warm-up
    assert np.linalg.norm(nus.mean(axis=0)) < 0.05


def test_kalman_validates_shapes() -> None:
    F = np.zeros((2, 2))
    G = np.eye(2)
    kf = NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2), R=np.eye(2))
    with pytest.raises(ValueError):
        kf.step(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        kf.step(np.zeros(2), np.zeros(3))


def test_kalman_warm_start_replaces_F_G() -> None:
    F = np.zeros((2, 2))
    G = np.eye(2)
    kf = NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2), R=np.eye(2))
    F2 = np.eye(2) * 0.1
    G2 = np.eye(2) * 0.5
    kf.warm_start(F_nominal=F2, G_nominal=G2)
    assert np.allclose(kf.F, F2)
    assert np.allclose(kf.G, G2)


def test_kalman_reset_returns_state_to_initial() -> None:
    kf = NominalKalman(
        F_nominal=np.zeros((2, 2)),
        G_nominal=np.eye(2),
        Q=np.eye(2) * 1e-3,
        R=np.eye(2) * 1e-2,
    )
    rng = np.random.default_rng(2)
    for _ in range(50):
        kf.step(rng.normal(size=2), rng.normal(size=2))
    kf.reset()
    assert np.allclose(kf.x_hat, 0.0)
    assert np.allclose(kf.P, np.eye(kf.n_state))


def test_kalman_validates_alpha_range() -> None:
    F = np.zeros((2, 2))
    G = np.eye(2)
    with pytest.raises(ValueError):
        NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2), R=np.eye(2), alpha_Q=-0.1)
    with pytest.raises(ValueError):
        NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2), R=np.eye(2), alpha_R=1.5)
    with pytest.raises(ValueError):
        NominalKalman(F_nominal=F, G_nominal=G, Q=np.eye(2), R=np.eye(2), alpha_Q=0.0)


def test_kalman_adaptation_changes_Q_and_R() -> None:
    rng = np.random.default_rng(3)
    F = np.zeros((2, 2))
    G = np.eye(2)
    kf = NominalKalman(
        F_nominal=F,
        G_nominal=G,
        Q=np.eye(2) * 1e-3,
        R=np.eye(2) * 1e-2,
        alpha_Q=0.95,
        alpha_R=0.95,
        adapt_Q=True,
        adapt_R=True,
    )
    Q0 = kf.Q.copy()
    R0 = kf.R.copy()
    x = np.zeros(2)
    for _ in range(200):
        u = rng.normal(0.0, 0.5, size=2)
        x = x + F @ x + G @ u + rng.normal(0.0, 0.05, size=2)
        kf.step(x, u)
    # Both Q and R must adapt away from their initial values.
    assert not np.allclose(kf.Q, Q0)
    assert not np.allclose(kf.R, R0)
    # Both must remain PSD (smallest eigenvalue >= 0).
    assert np.linalg.eigvalsh(kf.Q).min() >= -1e-12
    assert np.linalg.eigvalsh(kf.R).min() >= -1e-12
