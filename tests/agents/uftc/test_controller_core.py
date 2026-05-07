"""UFTCController predict/learn/reset/diagnostics — core unit tests."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _make_controller(n_state=3, n_control=3, dt=0.01,
                     warmup_steps=10, **overrides):
    cfg_kwargs = dict(dt=dt, fdd_warmup_steps=warmup_steps)
    cfg_kwargs.update(overrides)
    cfg = UFTCConfig(**cfg_kwargs)
    return UFTCController(
        n_state=n_state, n_control=n_control,
        nominal_F=np.zeros((n_state, n_state)),
        nominal_G=np.eye(n_state, n_control) * 0.1,
        config=cfg,
    )


def test_predict_returns_control_vector() -> None:
    ctl = _make_controller()
    u = ctl.predict(np.zeros(3), np.zeros(3), time_step=0)
    assert u.shape == (3,)


def test_predict_then_learn_cycle_runs_clean() -> None:
    rng = np.random.default_rng(0)
    ctl = _make_controller()
    x = np.zeros(3)
    for k in range(50):
        ref = rng.normal(scale=0.05, size=3)
        u = ctl.predict(x, ref, time_step=k)
        x = x + 0.05 * rng.normal(size=3) + 0.1 * u
        ctl.learn(x, ref, time_step=k)


def test_diagnostics_keys_present() -> None:
    ctl = _make_controller()
    ctl.predict(np.zeros(3), np.zeros(3), time_step=0)
    ctl.learn(np.zeros(3), np.zeros(3), time_step=0)
    diag = ctl.diagnostics()
    for key in ("fault_present", "severity", "confidence",
                "rls_gamma", "mode", "step"):
        assert key in diag


def test_warmup_suppresses_fault_present() -> None:
    rng = np.random.default_rng(1)
    ctl = _make_controller(warmup_steps=200)
    x = np.zeros(3)
    fired = False
    for k in range(150):
        ref = rng.normal(scale=0.05, size=3)
        u = ctl.predict(x, ref, time_step=k)
        x = x + 0.05 * rng.normal(size=3) + 0.1 * u
        ctl.learn(x, ref, time_step=k)
        fired = fired or ctl.diagnostics()["fault_present"]
    assert not fired


def test_reset_zeroes_step_counter_but_keeps_weights() -> None:
    rng = np.random.default_rng(2)
    ctl = _make_controller()
    for k in range(30):
        x = rng.normal(scale=0.1, size=3)
        ctl.predict(x, np.zeros(3), time_step=k)
        ctl.learn(x, np.zeros(3), time_step=k)
    F_before = ctl.middle.base.F.copy()
    ctl.reset()
    assert ctl.diagnostics()["step"] == 0
    # Weights survive reset.
    assert np.allclose(ctl.middle.base.F, F_before)


def test_no_omega_indices_yields_ref_as_omega_ref() -> None:
    ctl = _make_controller()
    # Without omega_indices on UFTCConfig, IADPMiddle uses ref as omega_ref.
    ref = np.array([0.1, -0.05, 0.0])
    ctl.predict(np.zeros(3), ref, time_step=0)
    # Internal state cached after predict.
    assert np.allclose(ctl._last_omega_ref, ref)
