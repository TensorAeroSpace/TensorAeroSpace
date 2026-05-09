"""Tests for CUSUM change-point detector used by UFTC FDD."""

from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.fdd.change_point import (
    ChangePointDetector,
    ChangePointState,
)


def test_chi_square_nominal_does_not_alarm() -> None:
    rng = np.random.default_rng(0)
    n = 4
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0, cooldown_steps=200)
    fired = False
    for _ in range(2000):
        d = float((rng.standard_normal(n) ** 2).sum())  # ~chi^2_n
        st = cpd.update(d)
        fired = fired or st.alarm
    assert not fired


def test_step_shift_triggers_alarm_within_latency() -> None:
    rng = np.random.default_rng(1)
    n = 4
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0, cooldown_steps=200)
    for _ in range(500):
        d = float((rng.standard_normal(n) ** 2).sum())
        cpd.update(d)
    fired_at = None
    # Inject step shift: mean moves from n to 4·n.
    for k in range(500):
        d = float(((rng.standard_normal(n) * 2.0) ** 2).sum()) + 0.0
        st = cpd.update(d)
        if st.alarm and fired_at is None:
            fired_at = k
            break
    assert fired_at is not None and fired_at < 50


def test_returns_change_point_state_dataclass() -> None:
    cpd = ChangePointDetector(n_dim=3)
    st = cpd.update(1.0)
    assert isinstance(st, ChangePointState)
    assert isinstance(st.cusum, float)
    assert isinstance(st.alarm, bool)
    assert isinstance(st.severity, float)
    assert isinstance(st.time_since_alarm, int)


def test_hysteresis_prevents_chattering_at_threshold() -> None:
    cpd = ChangePointDetector(
        n_dim=2, drift=2.0, h_alarm=10.0, h_clear=2.0, cooldown_steps=10
    )
    # Push above alarm.
    for _ in range(20):
        cpd.update(15.0)
    assert cpd.update(15.0).alarm
    # Drop just below alarm but above clear.
    transitions = 0
    prev = True
    for _ in range(100):
        st = cpd.update(5.0)
        if st.alarm != prev:
            transitions += 1
            prev = st.alarm
    # Without hysteresis we'd see fast toggling; with it ≤ 1 transition.
    assert transitions <= 1


def test_severity_grows_past_threshold() -> None:
    cpd = ChangePointDetector(n_dim=2, drift=2.0, h_alarm=10.0)
    severities = []
    for _ in range(50):
        severities.append(cpd.update(20.0).severity)
    assert severities[0] < severities[-1]
    assert severities[-1] >= 1.0  # past threshold


def test_reset_returns_to_initial() -> None:
    cpd = ChangePointDetector(n_dim=2)
    for _ in range(50):
        cpd.update(20.0)
    cpd.reset()
    st = cpd.update(2.0)
    assert st.cusum < 1.0
    assert not st.alarm


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        ChangePointDetector(n_dim=2, h_alarm=5.0, h_clear=10.0)
