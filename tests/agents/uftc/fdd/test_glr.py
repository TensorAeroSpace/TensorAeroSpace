"""GLR detector: nominal ARL₀, ramp-drift detection latency, hysteresis."""

from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.glr import (
    GLRConfig,
    GLRDetector,
    GLRState,
)


def test_returns_glr_state_dataclass() -> None:
    n = 3
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=50))
    nu = np.zeros(n)
    S = np.eye(n)
    st = glr.update(nu, S)
    assert isinstance(st, GLRState)
    assert isinstance(st.statistic, float)
    assert isinstance(st.alarm, bool)


def test_nominal_innovations_below_threshold() -> None:
    rng = np.random.default_rng(0)
    n = 3
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=200, h_alarm=30.0))
    fired = False
    S = np.eye(n)
    for _ in range(2000):
        nu = rng.standard_normal(n)
        st = glr.update(nu, S)
        fired = fired or st.alarm
    assert not fired


def test_ramp_drift_triggers_alarm() -> None:
    rng = np.random.default_rng(1)
    n = 3
    glr = GLRDetector(
        n_dim=n, cfg=GLRConfig(window=200, h_alarm=30.0, cooldown_steps=200)
    )
    S = np.eye(n)
    # Burn-in nominal noise.
    for _ in range(300):
        glr.update(rng.standard_normal(n), S)

    # Inject ramp drift: mean grows by 0.05 per step on first axis.
    fired_at = None
    for k in range(500):
        nu = rng.standard_normal(n).copy()
        nu[0] += 0.05 * (k + 1)
        st = glr.update(nu, S)
        if st.alarm and fired_at is None:
            fired_at = k
            break
    assert fired_at is not None
    assert fired_at < 200


def test_hysteresis_clears_after_cooldown_under_clean_innovations() -> None:
    rng = np.random.default_rng(2)
    n = 2
    glr = GLRDetector(
        n_dim=n, cfg=GLRConfig(window=100, h_alarm=20.0, h_clear=5.0, cooldown_steps=50)
    )
    S = np.eye(n)
    # Force statistic high.
    for _ in range(200):
        glr.update(np.array([5.0, 0.0]), S)
    assert glr.update(np.array([5.0, 0.0]), S).alarm

    # Restore clean innovations and run past cooldown.
    cleared = False
    for _ in range(500):
        st = glr.update(rng.standard_normal(n) * 0.1, S)
        if not st.alarm and st.statistic < 5.0:
            cleared = True
            break
    assert cleared


def test_reset_clears_window_and_alarm() -> None:
    n = 2
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=100, h_alarm=10.0))
    S = np.eye(n)
    for _ in range(300):
        glr.update(np.array([3.0, 0.0]), S)
    assert glr.update(np.array([3.0, 0.0]), S).alarm
    glr.reset()
    st = glr.update(np.zeros(n), S)
    assert not st.alarm
    assert st.statistic < 1e-6
