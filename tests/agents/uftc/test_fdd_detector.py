"""End-to-end FDD detector tests: nominal silence + step-fault detection."""

from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.change_point import ChangePointDetector
from tensoraerospace.agent.uftc.fdd.detector import (
    FDDConfig,
    FDDDetector,
    FDDOutput,
)
from tensoraerospace.agent.uftc.fdd.kalman_3step import NominalKalman


def _make_detector(n_state=2, n_control=2, dt=0.01, h_alarm=20.0):
    F = np.zeros((n_state, n_state))
    G = np.eye(n_state, n_control) * 0.1
    kf = NominalKalman(
        F_nominal=F,
        G_nominal=G,
        Q=np.eye(n_state) * 1e-3,
        R=np.eye(n_state) * 1e-2,
        adapt_Q=False,
        adapt_R=False,
    )
    cpd = ChangePointDetector(
        n_dim=n_state, h_alarm=h_alarm, h_clear=5.0, cooldown_steps=200
    )
    return (
        FDDDetector(n_state=n_state, n_control=n_control, kalman=kf, cpd=cpd, dt=dt),
        F,
        G,
    )


def test_fdd_output_shape() -> None:
    fdd, _, _ = _make_detector()
    out = fdd.step(np.zeros(2), np.zeros(2))
    assert isinstance(out, FDDOutput)
    assert isinstance(out.fault_present, bool)
    assert 0.0 <= out.confidence < 1.0
    assert out.severity >= 0.0


def test_nominal_flight_does_not_fire() -> None:
    rng = np.random.default_rng(0)
    fdd, F, G = _make_detector()
    x = np.zeros(2)
    fired = False
    for _ in range(2000):
        u = rng.normal(0.0, 0.5, size=2)
        x = x + F @ x + G @ u + rng.normal(0.0, 0.05, size=2)
        out = fdd.step(x, u)
        fired = fired or out.fault_present
    assert not fired


def test_step_fault_triggers_within_2s() -> None:
    rng = np.random.default_rng(1)
    fdd, F, G = _make_detector()
    x = np.zeros(2)
    for _ in range(500):
        u = rng.normal(0.0, 0.5, size=2)
        x = x + F @ x + G @ u + rng.normal(0.0, 0.05, size=2)
        fdd.step(x, u)
    # Fault: measurement bias step on the first sensor (e.g. an IMU
    # axis stuck-at offset). Pure Mahalanobis innovation distance grows
    # quadratically with the bias and CUSUM accumulates fast.
    bias = np.array([0.5, 0.0])
    fired_at = None
    for k in range(400):  # 4 s at dt=0.01
        u = rng.normal(0.0, 0.5, size=2)
        x = x + F @ x + G @ u + rng.normal(0.0, 0.05, size=2)
        out = fdd.step(x + bias, u)
        if out.fault_present and fired_at is None:
            fired_at = k
            break
    assert fired_at is not None and fired_at < 200  # within 2 s


def test_warm_start_replaces_kalman_F_G() -> None:
    fdd, F, G = _make_detector()
    F2 = F + 0.1
    G2 = G * 2.0
    fdd.warm_start(F_nominal=F2, G_nominal=G2)
    assert np.allclose(fdd.kalman.F, F2)
    assert np.allclose(fdd.kalman.G, G2)


def test_reset_clears_state() -> None:
    fdd, _, _ = _make_detector()
    rng = np.random.default_rng(2)
    for _ in range(50):
        fdd.step(rng.normal(size=2), rng.normal(size=2))
    fdd.reset()
    out = fdd.step(np.zeros(2), np.zeros(2))
    assert out.severity == 0.0
    assert not out.fault_present


def test_factory_method_builds_default_components() -> None:
    fdd = FDDDetector.from_config(
        n_state=3,
        n_control=2,
        dt=0.01,
        config=FDDConfig(h_alarm=25.0),
        F_nominal=np.zeros((3, 3)),
        G_nominal=np.zeros((3, 2)),
    )
    assert fdd.kalman.n_state == 3
    assert fdd.cpd.h_alarm == 25.0
