"""V_total monotonicity, alarm-level transitions with hysteresis."""

from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.monitor import (
    CompositeLyapunovMonitor,
    MonitorConfig,
    VState,
)


def _vstate(v: tuple[float, float, float, float, float], t: float = 0.0) -> VState:
    return VState(*v, timestamp=t)


def test_v_total_is_weighted_sum() -> None:
    cfg = MonitorConfig(
        c_weights=(0.1, 0.2, 0.3, 0.2, 0.2),
        a_diag=(1.0,) * 5,
        d_disturbance=(0.0,) * 5,
    )
    mon = CompositeLyapunovMonitor(cfg)
    out = mon.step(_vstate((1.0, 2.0, 3.0, 4.0, 5.0)))
    expected = 0.1 + 0.4 + 0.9 + 0.8 + 1.0
    assert abs(out.V_total - expected) < 1e-12


def test_alarm_transitions_warn_then_critical() -> None:
    cfg = MonitorConfig(
        c_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        a_diag=(1.0,) * 5,
        d_disturbance=(1.0,) * 5,
        alarm_warn_frac=0.5,
        alarm_critical_frac=0.9,
    )
    mon = CompositeLyapunovMonitor(cfg)
    mu = mon.mu_uub_pred
    # Quiet
    assert mon.step(_vstate((0.0, 0.0, 0.0, 0.0, 0.0))).alarm == "OK"
    # Above warn threshold
    out = mon.step(_vstate((0.6 * mu, 0.0, 0.0, 0.0, 0.0)))
    assert out.alarm == "WARN"
    # Above critical
    out = mon.step(_vstate((0.95 * mu, 0.0, 0.0, 0.0, 0.0)))
    assert out.alarm == "CRITICAL"


def test_hysteresis_clears_after_cooldown() -> None:
    cfg = MonitorConfig(
        c_weights=(1.0, 0.0, 0.0, 0.0, 0.0),
        a_diag=(1.0,) * 5,
        d_disturbance=(1.0,) * 5,
        alarm_warn_frac=0.5,
        alarm_critical_frac=0.9,
        cooldown_steps=10,
    )
    mon = CompositeLyapunovMonitor(cfg)
    mu = mon.mu_uub_pred
    mon.step(_vstate((0.95 * mu, 0.0, 0.0, 0.0, 0.0)))
    assert mon._alarm.level == "CRITICAL"
    cleared_at = None
    for k in range(200):
        out = mon.step(_vstate((0.0, 0.0, 0.0, 0.0, 0.0)))
        if out.alarm == "OK":
            cleared_at = k
            break
    assert cleared_at is not None and cleared_at > cfg.cooldown_steps - 1
