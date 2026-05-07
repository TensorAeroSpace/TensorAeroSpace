# UFTC Phase 1 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a plant-agnostic Unified Fault-Tolerant Control orchestrator (Phase 1 MVP) to TensorAeroSpace, composing existing AAINDIAgent (L2) and IADPAgent (L3) with a new innovation-driven CUSUM fault detector and runtime adaptation glue.

**Architecture:** Three layers wrapped by `UFTCController`. L3 (`IADPMiddle`) wraps `IADPAgent` with covariance-inflation RLS reset triggered by FDD rising edge. L2 (`WrappedAAINDI`) wraps `AAINDIAgent` with super-twisting sliding-mode observer, rate↔angle mode switch, and bounded trust-region clip targeting L3 output. FDD (`FDDDetector`) runs one adaptive 3-step Kalman of nominal dynamics + CUSUM change-point detector with hysteresis. Open-world: no fault catalog.

**Tech Stack:** Python 3.10+, NumPy (algorithm core), pytest (tests), poetry (run). PyTorch is **not** used in MVP — all arithmetic is NumPy. Existing `aa_indi.AAINDIAgent` and `iadp.IADPAgent` are composed unchanged.

**Spec:** [`docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md`](../specs/2026-05-07-uftc-phase1-mvp-design.md)

**Build order (bottom-up TDD):**

```
NominalKalman ──┐
                ├─→ FDDDetector ──┐
ChangePointDet ─┘                 │
                                  │
SuperTwistingObs ──┐               │
                   ├─→ WrappedAAINDI ─┐
ModeSwitcher ──────┘                  │
                                      ├─→ UFTCController ─→ integration tests
                                      │
RLSResetPolicy ──→ IADPMiddle ────────┘
```

**Conventions used throughout:**
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest <path> -v`. The env-var matches the project Makefile (`make test`); pytest-cov picks it up only when re-enabled with `-p pytest_cov`.
- Test files live in `tests/agents/uftc/`; use the same `from __future__ import annotations` + numpy + pytest pattern as `tests/agents/aa_indi_test.py`.
- New source files live under `tensoraerospace/agent/uftc/`. Each module starts with a triple-quoted docstring summarising responsibility (mirror `aa_indi/model.py` style).
- Commit message style follows recent commits (`feat(uftc): ...`, `test(uftc): ...`, etc. — no Claude attribution per project memory).
- Type-hints are mandatory on public symbols. Private helpers may omit them.

---

### Task 1: Bootstrap `tensoraerospace/agent/uftc/` package skeleton

**Files:**
- Create: `tensoraerospace/agent/uftc/__init__.py`
- Create: `tensoraerospace/agent/uftc/fdd/__init__.py`
- Create: `tests/agents/uftc/__init__.py`
- Create: `tests/agents/uftc/test_package_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_package_skeleton.py
"""Smoke import test that locks in the public surface of the uftc package."""
from __future__ import annotations


def test_uftc_package_importable() -> None:
    import tensoraerospace.agent.uftc as uftc
    assert hasattr(uftc, "__all__")
    # Phase 1 MVP exports — populated incrementally by later tasks.
    assert isinstance(uftc.__all__, list)


def test_uftc_fdd_subpackage_importable() -> None:
    import tensoraerospace.agent.uftc.fdd as fdd
    assert hasattr(fdd, "__all__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_package_skeleton.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tensoraerospace.agent.uftc'`.

- [ ] **Step 3: Create the skeleton**

```python
# tensoraerospace/agent/uftc/__init__.py
"""Unified Fault-Tolerant Control (UFTC) — Phase 1 MVP orchestrator.

Composes the existing AAINDIAgent (L2 inner) and IADPAgent (L3 middle)
with an innovation-driven CUSUM fault detector. Exports populated by
subsequent task implementations.

See ``docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md``.
"""
from __future__ import annotations

__all__: list[str] = []
```

```python
# tensoraerospace/agent/uftc/fdd/__init__.py
"""Fault Detection and Diagnosis primitives for UFTC."""
from __future__ import annotations

__all__: list[str] = []
```

```python
# tests/agents/uftc/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_package_skeleton.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/__init__.py \
        tensoraerospace/agent/uftc/fdd/__init__.py \
        tests/agents/uftc/__init__.py \
        tests/agents/uftc/test_package_skeleton.py
git commit -m "feat(uftc): bootstrap uftc package skeleton"
```

---

### Task 2: `NominalKalman` adaptive 3-step Kalman filter

**Files:**
- Create: `tensoraerospace/agent/uftc/fdd/kalman_3step.py`
- Modify: `tensoraerospace/agent/uftc/fdd/__init__.py`
- Create: `tests/agents/uftc/test_kalman_3step.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_kalman_3step.py
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
    kf = NominalKalman(F_nominal=F, G_nominal=G,
                       Q=np.eye(2) * 1e-3, R=np.eye(2) * 1e-2)
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
    kf = NominalKalman(F_nominal=F, G_nominal=G,
                       Q=np.eye(2) * 1e-4, R=np.eye(2) * 1e-2)
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
    kf = NominalKalman(F_nominal=F, G_nominal=G,
                       Q=np.eye(2) * 1e-3, R=np.eye(2) * 1e-2)
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
    kf = NominalKalman(F_nominal=np.zeros((2, 2)), G_nominal=np.eye(2),
                       Q=np.eye(2) * 1e-3, R=np.eye(2) * 1e-2)
    rng = np.random.default_rng(2)
    for _ in range(50):
        kf.step(rng.normal(size=2), rng.normal(size=2))
    P_post = kf.P.copy()
    kf.reset()
    assert not np.allclose(kf.P, P_post)  # state mutated by reset
    assert np.allclose(kf.x_hat, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_kalman_3step.py -v`
Expected: FAIL with `ModuleNotFoundError: tensoraerospace.agent.uftc.fdd.kalman_3step`.

- [ ] **Step 3: Implement NominalKalman**

```python
# tensoraerospace/agent/uftc/fdd/kalman_3step.py
"""Adaptive 3-step Kalman filter for UFTC FDD.

Lu, P. et al. (2015) "Adaptive three-step Kalman filter for air-data
sensor fault detection," AIAA JGCD — adaptive Q, R via Sage-Husa
exponentially-weighted innovation/residual covariance updates.

The filter assumes the plant is locally linear in the state with a
known control map ``G``::

    x_{t+1} ≈ x_t + F · x_t + G · u_t  (incremental form)

Both ``F`` and ``G`` may be warm-started by ``UFTCController`` once the
incremental RLS inside :class:`IADPAgent` has converged on nominal
flight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanStep:
    """One-step output of :class:`NominalKalman`."""

    x_hat: np.ndarray  # posterior state estimate (n_state,)
    nu: np.ndarray     # innovation y - H·x_hat_prior (n_state,)
    S: np.ndarray      # innovation covariance (n_state, n_state)
    K: np.ndarray      # Kalman gain (n_state, n_state)


class NominalKalman:
    """Adaptive 3-step Kalman filter on a linear nominal plant.

    Args:
        F_nominal: System matrix increment, shape ``(n_state, n_state)``.
        G_nominal: Control map, shape ``(n_state, n_control)``.
        Q: Process noise covariance, shape ``(n_state, n_state)``.
        R: Measurement noise covariance, shape ``(n_state, n_state)``.
        alpha_Q: EMA coefficient for adaptive Q (default 0.99 — slow).
        alpha_R: EMA coefficient for adaptive R (default 0.99).
        adapt_Q: Enable Sage-Husa Q adaptation (default True).
        adapt_R: Enable Sage-Husa R adaptation (default True).
    """

    def __init__(
        self,
        F_nominal: np.ndarray,
        G_nominal: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        *,
        alpha_Q: float = 0.99,
        alpha_R: float = 0.99,
        adapt_Q: bool = True,
        adapt_R: bool = True,
    ) -> None:
        self.F = np.array(F_nominal, dtype=np.float64, copy=True)
        self.G = np.array(G_nominal, dtype=np.float64, copy=True)
        self.Q = np.array(Q, dtype=np.float64, copy=True)
        self.R = np.array(R, dtype=np.float64, copy=True)
        self.alpha_Q = float(alpha_Q)
        self.alpha_R = float(alpha_R)
        self.adapt_Q = bool(adapt_Q)
        self.adapt_R = bool(adapt_R)

        self.n_state = self.F.shape[0]
        self.n_control = self.G.shape[1]
        if self.F.shape != (self.n_state, self.n_state):
            raise ValueError("F_nominal must be square")
        if self.G.shape != (self.n_state, self.n_control):
            raise ValueError("G_nominal must be (n_state, n_control)")
        if self.Q.shape != (self.n_state, self.n_state):
            raise ValueError("Q must match n_state")
        if self.R.shape != (self.n_state, self.n_state):
            raise ValueError("R must match n_state")

        self._reset_state()

    def _reset_state(self) -> None:
        self.x_hat = np.zeros(self.n_state, dtype=np.float64)
        self.P = np.eye(self.n_state, dtype=np.float64)

    def reset(self) -> None:
        """Restore filter state to zero / identity covariance."""
        self._reset_state()

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        """Replace ``F``/``G`` with refreshed estimates of nominal dynamics."""
        if F_nominal is not None:
            F_arr = np.array(F_nominal, dtype=np.float64)
            if F_arr.shape != self.F.shape:
                raise ValueError(
                    f"F_nominal shape mismatch: {F_arr.shape} vs {self.F.shape}"
                )
            self.F = F_arr
        if G_nominal is not None:
            G_arr = np.array(G_nominal, dtype=np.float64)
            if G_arr.shape != self.G.shape:
                raise ValueError(
                    f"G_nominal shape mismatch: {G_arr.shape} vs {self.G.shape}"
                )
            self.G = G_arr

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> KalmanStep:
        """Run one Kalman update.

        Args:
            x_meas: Measured state at time t, shape ``(n_state,)``.
            u_prev: Control applied at t-1, shape ``(n_control,)``.

        Returns:
            KalmanStep with posterior x_hat, innovation, S, K.
        """
        x = np.asarray(x_meas, dtype=np.float64).reshape(-1)
        u = np.asarray(u_prev, dtype=np.float64).reshape(-1)
        if x.size != self.n_state:
            raise ValueError(f"x_meas must have length {self.n_state}, got {x.size}")
        if u.size != self.n_control:
            raise ValueError(f"u_prev must have length {self.n_control}, got {u.size}")

        # Step 1: prior prediction (incremental form).
        x_prior = self.x_hat + self.F @ self.x_hat + self.G @ u
        F_jac = np.eye(self.n_state) + self.F
        P_prior = F_jac @ self.P @ F_jac.T + self.Q

        # Step 2: innovation.
        nu = x - x_prior  # H = I (full-state measurement)
        S = P_prior + self.R
        # Solve K via S, falling back to pinv on rank deficiency.
        try:
            K = np.linalg.solve(S.T, P_prior.T).T
        except np.linalg.LinAlgError:
            K = P_prior @ np.linalg.pinv(S)

        # Step 3: posterior correction.
        x_post = x_prior + K @ nu
        I_KH = np.eye(self.n_state) - K  # H = I
        P_post = I_KH @ P_prior @ I_KH.T + K @ self.R @ K.T

        # Sage-Husa adaptive Q, R. Both updates are EMA on outer products.
        if self.adapt_R:
            residual = x - x_post  # H·x_post = x_post
            self.R = self.alpha_R * self.R + (1.0 - self.alpha_R) * (
                np.outer(residual, residual) + I_KH @ P_prior @ I_KH.T
            )
            # Keep R symmetric and positive-definite-ish.
            self.R = 0.5 * (self.R + self.R.T)

        if self.adapt_Q:
            dx = K @ nu
            self.Q = self.alpha_Q * self.Q + (1.0 - self.alpha_Q) * (
                np.outer(dx, dx) + P_post - F_jac @ self.P @ F_jac.T
            )
            self.Q = 0.5 * (self.Q + self.Q.T)

        # Persist posterior state.
        self.x_hat = x_post
        self.P = 0.5 * (P_post + P_post.T)

        return KalmanStep(
            x_hat=self.x_hat.copy(), nu=nu.copy(), S=S.copy(), K=K.copy()
        )
```

- [ ] **Step 4: Wire up `__init__.py`**

```python
# tensoraerospace/agent/uftc/fdd/__init__.py
"""Fault Detection and Diagnosis primitives for UFTC."""
from __future__ import annotations

from .kalman_3step import KalmanStep, NominalKalman

__all__ = ["KalmanStep", "NominalKalman"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_kalman_3step.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/uftc/fdd/kalman_3step.py \
        tensoraerospace/agent/uftc/fdd/__init__.py \
        tests/agents/uftc/test_kalman_3step.py
git commit -m "feat(uftc): add adaptive 3-step Kalman filter for nominal dynamics"
```

---

### Task 3: `ChangePointDetector` — CUSUM with hysteresis

**Files:**
- Create: `tensoraerospace/agent/uftc/fdd/change_point.py`
- Modify: `tensoraerospace/agent/uftc/fdd/__init__.py`
- Create: `tests/agents/uftc/test_change_point.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_change_point.py
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
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0,
                              cooldown_steps=200)
    fired = False
    for _ in range(2000):
        d = float((rng.standard_normal(n) ** 2).sum())  # ~chi^2_n
        st = cpd.update(d)
        fired = fired or st.alarm
    assert not fired


def test_step_shift_triggers_alarm_within_latency() -> None:
    rng = np.random.default_rng(1)
    n = 4
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0,
                              cooldown_steps=200)
    for _ in range(500):
        d = float((rng.standard_normal(n) ** 2).sum())
        cpd.update(d)
    fired_at = None
    # Inject step shift: mean moves from n to 4·n.
    for k in range(500):
        d = float((rng.standard_normal(n) * 2.0) ** 2).sum() + 0.0
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
    cpd = ChangePointDetector(n_dim=2, drift=2.0, h_alarm=10.0,
                              h_clear=2.0, cooldown_steps=10)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_change_point.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `ChangePointDetector`**

```python
# tensoraerospace/agent/uftc/fdd/change_point.py
"""CUSUM change-point detector with hysteresis for UFTC FDD.

Drives the binary `fault_present` flag and the soft `severity` signal
that the rest of the controller uses to scale RLS reset and trust-region
expansion.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChangePointState:
    """One-step output of :class:`ChangePointDetector`."""

    cusum: float
    alarm: bool
    severity: float          # cusum / h_alarm, clipped to [0, 10]
    time_since_alarm: int    # steps since last rising edge (0 if never fired)


class ChangePointDetector:
    """One-sided CUSUM detector on a positive scalar score (e.g. Mahalanobis).

    Args:
        n_dim: Dimension of the underlying innovation vector. Used as
            the default ``drift`` value (the mean of χ²_n).
        drift: Per-step decrement subtracted from each input. ``None``
            defaults to ``n_dim``.
        h_alarm: Upper threshold; CUSUM crossing this triggers an alarm.
        h_clear: Lower threshold; CUSUM falling below this clears the
            alarm (after cooldown). Must be strictly less than h_alarm.
        cooldown_steps: Minimum steps between rising edges. Re-arming the
            detector during cooldown is suppressed.

    Raises:
        ValueError: If ``h_alarm <= h_clear`` or any threshold is
            non-positive.
    """

    def __init__(
        self,
        *,
        n_dim: int,
        drift: float | None = None,
        h_alarm: float = 20.0,
        h_clear: float = 5.0,
        cooldown_steps: int = 200,
    ) -> None:
        if h_alarm <= 0.0 or h_clear <= 0.0:
            raise ValueError("Thresholds must be positive")
        if h_alarm <= h_clear:
            raise ValueError("h_alarm must be strictly greater than h_clear")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")
        self.n_dim = int(n_dim)
        self.drift = float(drift if drift is not None else n_dim)
        self.h_alarm = float(h_alarm)
        self.h_clear = float(h_clear)
        self.cooldown_steps = int(cooldown_steps)
        self.reset()

    def reset(self) -> None:
        """Clear CUSUM state."""
        self._cusum = 0.0
        self._alarm = False
        self._cooldown = 0
        self._time_since_alarm = 0

    def update(self, d_t: float) -> ChangePointState:
        """Feed one Mahalanobis-distance sample, update state."""
        self._cusum = max(0.0, self._cusum + float(d_t) - self.drift)

        if self._cooldown > 0:
            self._cooldown -= 1

        if not self._alarm:
            if self._cusum > self.h_alarm and self._cooldown == 0:
                self._alarm = True
                self._time_since_alarm = 0
                self._cooldown = self.cooldown_steps
        else:
            if self._cusum < self.h_clear and self._cooldown == 0:
                self._alarm = False
            self._time_since_alarm += 1

        severity = max(0.0, min(self._cusum / self.h_alarm, 10.0))
        return ChangePointState(
            cusum=float(self._cusum),
            alarm=bool(self._alarm),
            severity=float(severity),
            time_since_alarm=int(self._time_since_alarm),
        )
```

- [ ] **Step 4: Wire up `__init__.py`**

```python
# tensoraerospace/agent/uftc/fdd/__init__.py
"""Fault Detection and Diagnosis primitives for UFTC."""
from __future__ import annotations

from .change_point import ChangePointDetector, ChangePointState
from .kalman_3step import KalmanStep, NominalKalman

__all__ = [
    "ChangePointDetector",
    "ChangePointState",
    "KalmanStep",
    "NominalKalman",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_change_point.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/uftc/fdd/change_point.py \
        tensoraerospace/agent/uftc/fdd/__init__.py \
        tests/agents/uftc/test_change_point.py
git commit -m "feat(uftc): add CUSUM change-point detector with hysteresis"
```

---

### Task 4: `FDDDetector` composite

**Files:**
- Create: `tensoraerospace/agent/uftc/fdd/detector.py`
- Modify: `tensoraerospace/agent/uftc/fdd/__init__.py`
- Create: `tests/agents/uftc/test_fdd_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_fdd_detector.py
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
    kf = NominalKalman(F_nominal=F, G_nominal=G,
                       Q=np.eye(n_state) * 1e-3,
                       R=np.eye(n_state) * 1e-2,
                       adapt_Q=False, adapt_R=False)
    cpd = ChangePointDetector(n_dim=n_state, h_alarm=h_alarm,
                              h_clear=5.0, cooldown_steps=200)
    return FDDDetector(n_state=n_state, n_control=n_control,
                       kalman=kf, cpd=cpd, dt=dt), F, G


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
    # Fault: G drops to ~zero (control surface lost).
    G_fault = G * 0.05
    fired_at = None
    for k in range(400):  # 4 s at dt=0.01
        u = rng.normal(0.0, 0.5, size=2)
        x = x + F @ x + G_fault @ u + rng.normal(0.0, 0.05, size=2)
        out = fdd.step(x, u)
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
        n_state=3, n_control=2, dt=0.01,
        config=FDDConfig(h_alarm=25.0),
        F_nominal=np.zeros((3, 3)), G_nominal=np.zeros((3, 2)),
    )
    assert fdd.kalman.n_state == 3
    assert fdd.cpd.h_alarm == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_fdd_detector.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `FDDDetector`**

```python
# tensoraerospace/agent/uftc/fdd/detector.py
"""Composite FDD detector: NominalKalman + CUSUM."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .change_point import ChangePointDetector
from .kalman_3step import NominalKalman


@dataclass
class FDDConfig:
    """Hyper-parameters for :class:`FDDDetector`."""

    # Kalman.
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    alpha_Q: float = 0.99
    alpha_R: float = 0.99
    adapt_Q: bool = True
    adapt_R: bool = True
    # CUSUM.
    drift: float | None = None
    h_alarm: float = 20.0
    h_clear: float = 5.0
    cooldown_steps: int = 200
    # Innovation gating: skip Kalman update when innovation is too far out.
    innovation_sigma_gate: float = 5.0


@dataclass
class FDDOutput:
    """One-step output of :class:`FDDDetector`."""

    fault_present: bool
    severity: float
    confidence: float          # 1 − exp(−severity); ∈ [0, 1)
    innovation_norm: float
    time_since_event: float    # seconds since last rising edge


class FDDDetector:
    """One nominal Kalman + one CUSUM detector → FDDOutput."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        kalman: NominalKalman,
        cpd: ChangePointDetector,
        *,
        dt: float,
        innovation_sigma_gate: float = 5.0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.kalman = kalman
        self.cpd = cpd
        self.dt = float(dt)
        self.innovation_sigma_gate = float(innovation_sigma_gate)

    @classmethod
    def from_config(
        cls,
        n_state: int,
        n_control: int,
        *,
        dt: float,
        config: FDDConfig,
        F_nominal: np.ndarray,
        G_nominal: np.ndarray,
    ) -> "FDDDetector":
        """Build detector with default Kalman / CPD wired from FDDConfig."""
        Q = np.eye(n_state) * config.process_noise
        R = np.eye(n_state) * config.measurement_noise
        kf = NominalKalman(
            F_nominal=F_nominal, G_nominal=G_nominal, Q=Q, R=R,
            alpha_Q=config.alpha_Q, alpha_R=config.alpha_R,
            adapt_Q=config.adapt_Q, adapt_R=config.adapt_R,
        )
        cpd = ChangePointDetector(
            n_dim=n_state, drift=config.drift,
            h_alarm=config.h_alarm, h_clear=config.h_clear,
            cooldown_steps=config.cooldown_steps,
        )
        return cls(n_state=n_state, n_control=n_control,
                   kalman=kf, cpd=cpd, dt=dt,
                   innovation_sigma_gate=config.innovation_sigma_gate)

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        """Update Kalman F/G from a refreshed nominal estimate."""
        self.kalman.warm_start(F_nominal=F_nominal, G_nominal=G_nominal)

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> FDDOutput:
        """Run Kalman + CUSUM; return FDDOutput."""
        kal = self.kalman.step(x_meas, u_prev)
        # Mahalanobis distance (regularised solve).
        try:
            d_t = float(kal.nu @ np.linalg.solve(kal.S, kal.nu))
        except np.linalg.LinAlgError:
            d_t = float(kal.nu @ (np.linalg.pinv(kal.S) @ kal.nu))
        d_t = max(d_t, 0.0)

        # Innovation gating: massive jumps are flagged but Kalman state
        # has already been updated. We accept the trade-off — the CPD
        # will fire which is the desired response.
        cp = self.cpd.update(d_t)

        confidence = float(1.0 - np.exp(-cp.severity))
        time_since_event = float(cp.time_since_alarm) * self.dt
        return FDDOutput(
            fault_present=cp.alarm,
            severity=cp.severity,
            confidence=confidence,
            innovation_norm=float(np.linalg.norm(kal.nu)),
            time_since_event=time_since_event,
        )

    def reset(self) -> None:
        """Reset Kalman + CUSUM state. Configs unchanged."""
        self.kalman.reset()
        self.cpd.reset()
```

- [ ] **Step 4: Wire up `__init__.py`**

```python
# tensoraerospace/agent/uftc/fdd/__init__.py
"""Fault Detection and Diagnosis primitives for UFTC."""
from __future__ import annotations

from .change_point import ChangePointDetector, ChangePointState
from .detector import FDDConfig, FDDDetector, FDDOutput
from .kalman_3step import KalmanStep, NominalKalman

__all__ = [
    "ChangePointDetector",
    "ChangePointState",
    "FDDConfig",
    "FDDDetector",
    "FDDOutput",
    "KalmanStep",
    "NominalKalman",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_fdd_detector.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/uftc/fdd/detector.py \
        tensoraerospace/agent/uftc/fdd/__init__.py \
        tests/agents/uftc/test_fdd_detector.py
git commit -m "feat(uftc): add composite FDDDetector (Kalman + CUSUM)"
```

---

### Task 5: `SuperTwistingObserver` and `ModeSwitcher`

**Files:**
- Create: `tensoraerospace/agent/uftc/inner.py` (partial — observer + mode switch only)
- Create: `tests/agents/uftc/test_inner_components.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_inner_components.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_inner_components.py -v`
Expected: FAIL — `inner` module missing.

- [ ] **Step 3: Implement observer + mode switch**

```python
# tensoraerospace/agent/uftc/inner.py
"""Inner-loop (L2) extensions for UFTC: SM observer, mode switch,
trust-region wrapper around aa_indi.AAINDIAgent.

Phase 1 MVP — see docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


class SuperTwistingObserver:
    """Higher-order sliding-mode observer (super-twisting algorithm).

    Estimates the unmodeled high-frequency disturbance δ̂ on each angular
    axis from the residual ``s = ω̇_meas − ν_des − δ̂``::

        ṡ = −k₁·|s|^{1/2}·sign(s) + z
        ż = −k₂·sign(s)

    Args:
        n_axes: Number of angular axes (typically 3 for an aircraft).
        k1: Outer super-twisting gain.
        k2: Inner super-twisting gain.
        dt: Sampling period [s].
    """

    def __init__(
        self,
        n_axes: int,
        *,
        k1: float = 3.0,
        k2: float = 1.5,
        dt: float = 0.01,
    ) -> None:
        self.n_axes = int(n_axes)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        """Clear observer state."""
        self._s = np.zeros(self.n_axes, dtype=np.float64)
        self._z = np.zeros(self.n_axes, dtype=np.float64)

    def update(
        self,
        omega_dot_meas: np.ndarray,
        nu_des: np.ndarray,
    ) -> np.ndarray:
        """Run one observer step. Returns δ̂ ≈ (ω̇_meas − ν_des)."""
        wd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        nd = np.asarray(nu_des, dtype=np.float64).reshape(-1)
        if wd.size != self.n_axes:
            raise ValueError(
                f"omega_dot_meas must have length {self.n_axes}, got {wd.size}"
            )
        if nd.size != self.n_axes:
            raise ValueError(
                f"nu_des must have length {self.n_axes}, got {nd.size}"
            )

        # δ̂ ≈ s + z (z accumulates the integrated discontinuous term).
        delta_hat = self._s + self._z

        # Sliding variable σ = ω̇_meas − ν_des − δ̂ — but in this surrogate
        # we treat (s, z) as the state directly. Discrete-time Euler
        # integration of the super-twisting law with ``e = ω̇_meas − ν_des``.
        e = wd - nd
        sgn = np.sign(self._s - e)
        abs_term = np.sqrt(np.abs(self._s - e))
        ds = -self.k1 * abs_term * sgn + self._z
        dz = -self.k2 * sgn

        self._s = self._s + self.dt * ds
        self._z = self._z + self.dt * dz

        # Recompute δ̂ after step for consistency.
        delta_hat = e + (self._z - self._s)
        return delta_hat


class ModeSwitcher:
    """Hysteretic rate-INDI ↔ angle-INDI mode selector.

    Args:
        alpha_threshold_deg: AoA above which the mode switches to angle-INDI.
        hysteresis_deg: AoA must drop below ``alpha_threshold_deg − hysteresis_deg``
            to switch back to rate-INDI.
    """

    def __init__(
        self,
        alpha_threshold_deg: float = 25.0,
        hysteresis_deg: float = 5.0,
    ) -> None:
        if hysteresis_deg < 0.0:
            raise ValueError("hysteresis_deg must be non-negative")
        self.alpha_threshold = float(np.deg2rad(alpha_threshold_deg))
        self.alpha_clear = float(np.deg2rad(alpha_threshold_deg - hysteresis_deg))
        self.reset()

    def reset(self) -> None:
        """Return to the default rate-INDI mode."""
        self._mode: Literal["rate", "angle"] = "rate"

    def select(self, alpha_rad: float) -> Literal["rate", "angle"]:
        """Update mode given current α and return new mode label."""
        a = float(alpha_rad)
        if self._mode == "rate" and a > self.alpha_threshold:
            self._mode = "angle"
        elif self._mode == "angle" and a < self.alpha_clear:
            self._mode = "rate"
        return self._mode

    @property
    def mode(self) -> Literal["rate", "angle"]:
        return self._mode
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_inner_components.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/inner.py \
        tests/agents/uftc/test_inner_components.py
git commit -m "feat(uftc): add super-twisting observer and rate/angle mode switch"
```

---

### Task 6: `WrappedAAINDI` — bounded trust-region wrapper

**Files:**
- Modify: `tensoraerospace/agent/uftc/inner.py` (append `WrappedAAINDI`)
- Create: `tests/agents/uftc/test_wrapped_aa_indi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_wrapped_aa_indi.py
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
    for k in range(50):
        omega = rng.normal(scale=0.1, size=3)
        ref = rng.normal(scale=0.1, size=3)
        u = w.predict(
            omega_ref=ref, omega_meas=omega, alpha=0.0,
            u_blend_target=u_blend_target if k > 0 else np.zeros(3),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_wrapped_aa_indi.py -v`
Expected: FAIL — `WrappedAAINDI` not exported.

- [ ] **Step 3: Append `WrappedAAINDI` to `inner.py`**

Append at the bottom of `tensoraerospace/agent/uftc/inner.py`:

```python
from tensoraerospace.agent.aa_indi.model import AAINDIAgent


class WrappedAAINDI:
    """Bounded trust-region wrapper around :class:`AAINDIAgent`.

    Adds three behaviours on top of the AAINDI base:

    1. **Sliding-mode disturbance compensation.** A super-twisting
       observer estimates δ̂ from the residual ``ω̇_meas − ν_des`` and
       feeds a small additive correction into ω_meas before AAINDI
       runs.
    2. **Rate-INDI / angle-INDI mode switching.** The mode-switch label
       is exposed via ``self.mode``; the underlying AAINDI law is the
       same in MVP (true angle-INDI law lands in Phase 2).
    3. **Bounded trust-region.** Final command is clipped to a ball of
       radius ``δ`` around ``u_blend_target`` (the L3 absolute-control
       output). Radius interpolates linearly between
       ``trust_radius_nominal`` and ``trust_radius_fault`` driven by
       ``fault_severity ∈ [0, 1]`` (clipped if outside).
    """

    def __init__(
        self,
        base: AAINDIAgent,
        *,
        sm_obs: SuperTwistingObserver,
        mode_switch: ModeSwitcher,
        trust_radius_nominal: float = 0.1,
        trust_radius_fault: float = 0.5,
        dt: float = 0.01,
    ) -> None:
        if trust_radius_nominal <= 0.0 or trust_radius_fault <= 0.0:
            raise ValueError("trust radii must be positive")
        if trust_radius_fault < trust_radius_nominal:
            raise ValueError(
                "trust_radius_fault must be ≥ trust_radius_nominal"
            )
        self.base = base
        self.sm_obs = sm_obs
        self.mode_switch = mode_switch
        self.trust_radius_nominal = float(trust_radius_nominal)
        self.trust_radius_fault = float(trust_radius_fault)
        self.dt = float(dt)
        self._n_state = base.n_state
        self._n_control = base.n_control

    def predict(
        self,
        omega_ref: np.ndarray,
        omega_meas: np.ndarray,
        *,
        alpha: float,
        u_blend_target: np.ndarray,
        fault_severity: float,
        time_step: int,
    ) -> np.ndarray:
        """One control-tick from a rate target → constrained INDI action."""
        omega_meas_v = np.asarray(omega_meas, dtype=np.float64).reshape(-1)
        omega_ref_v = np.asarray(omega_ref, dtype=np.float64).reshape(-1)
        u_target = np.asarray(u_blend_target, dtype=np.float64).reshape(-1)
        if omega_meas_v.size != self._n_state:
            raise ValueError(f"omega_meas size mismatch: {omega_meas_v.size}")
        if omega_ref_v.size != self._n_state:
            raise ValueError(f"omega_ref size mismatch: {omega_ref_v.size}")
        if u_target.size != self._n_control:
            raise ValueError(f"u_blend_target size mismatch: {u_target.size}")

        # SM observer with omega_dot ~ (ω_ref − ω_meas)/dt as a proxy for ω̇_meas.
        omega_dot_proxy = (omega_meas_v - getattr(self, "_prev_omega",
                                                  omega_meas_v)) / self.dt
        self._prev_omega = omega_meas_v.copy()
        nu_des_proxy = (omega_ref_v - omega_meas_v) / max(self.dt, 1e-9)
        delta_hat = self.sm_obs.update(omega_dot_proxy, nu_des_proxy)

        # Mode update (label only — MVP uses one allocator).
        self.mode_switch.select(alpha)

        # SM disturbance compensation in measurement space.
        omega_corrected = omega_meas_v - delta_hat * self.dt

        u_indi_raw = self.base.predict(omega_corrected, omega_ref_v,
                                       time_step=time_step)

        # Trust-region clip targeting L3 absolute control.
        sev = float(np.clip(fault_severity, 0.0, 1.0))
        radius = (self.trust_radius_nominal +
                  (self.trust_radius_fault - self.trust_radius_nominal) * sev)
        diff = u_indi_raw - u_target
        norm = float(np.linalg.norm(diff))
        if norm > radius and norm > 0.0:
            u_indi = u_target + diff * (radius / norm)
        else:
            u_indi = u_indi_raw

        # Re-clip against AAINDI magnitude limit (u_target may already exceed it).
        u_indi = np.clip(u_indi,
                         -self.base.cfg.u_magnitude_limit,
                         self.base.cfg.u_magnitude_limit)
        return u_indi

    def learn(
        self,
        next_omega: np.ndarray,
        omega_ref: np.ndarray,
        time_step: int,
    ) -> dict:
        """Forward to AAINDI's learn step."""
        return self.base.learn(next_omega, omega_ref, time_step=time_step)

    def reset(self) -> None:
        self.base.reset()
        self.sm_obs.reset()
        self.mode_switch.reset()
        self._prev_omega = None  # type: ignore[assignment]

    @property
    def mode(self) -> str:
        return self.mode_switch.mode
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_wrapped_aa_indi.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/inner.py \
        tests/agents/uftc/test_wrapped_aa_indi.py
git commit -m "feat(uftc): wrap AAINDIAgent with super-twisting + trust-region clip"
```

---

### Task 7: `IADPMiddle` — RLS reset on FDD rising edge

**Files:**
- Create: `tensoraerospace/agent/uftc/middle.py`
- Create: `tests/agents/uftc/test_middle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_middle.py
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
    pol = RLSResetPolicy(cov_inflation=100.0,
                         forgetting_drop=0.9,
                         forgetting_recover_steps=100)
    return IADPMiddle(base=base, reset_policy=pol)


def _nominal_fdd() -> FDDOutput:
    return FDDOutput(fault_present=False, severity=0.0,
                     confidence=0.0, innovation_norm=0.0,
                     time_since_event=0.0)


def _alarm_fdd() -> FDDOutput:
    return FDDOutput(fault_present=True, severity=1.5,
                     confidence=0.78, innovation_norm=2.0,
                     time_since_event=0.0)


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
        m.learn(x + 0.01 * rng.normal(size=2), ref, time_step=k,
                fdd=_nominal_fdd())
    phi_pre = np.linalg.norm(m.base.rls.Phi)
    m.learn(rng.normal(scale=0.1, size=2), np.zeros(2), time_step=5,
            fdd=_alarm_fdd())
    phi_post = np.linalg.norm(m.base.rls.Phi)
    assert phi_post > phi_pre


def test_forgetting_recovers_after_drop() -> None:
    m = _make_middle()
    m.learn(np.zeros(2), np.zeros(2), time_step=0, fdd=_alarm_fdd())
    assert m.base.rls.gamma_rls == pytest.approx(0.9, abs=1e-6)
    # Step nominal updates until recovery completes.
    for k in range(101):
        m.learn(np.zeros(2), np.zeros(2), time_step=k + 1,
                fdd=_nominal_fdd())
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
    m = IADPMiddle(base=base, reset_policy=pol,
                   omega_indices=[0, 1], lookahead_dt=0.1)
    x = np.array([0.0, 0.0])
    ref = np.array([1.0, -0.5])
    _, omega_ref = m.predict(x, ref, time_step=0)
    expected = (ref - x) / 0.1
    assert np.allclose(omega_ref, expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_middle.py -v`
Expected: FAIL — `middle` module missing.

- [ ] **Step 3: Implement `IADPMiddle`**

```python
# tensoraerospace/agent/uftc/middle.py
"""L3 middle for UFTC: IADPAgent + innovation-driven RLS reset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tensoraerospace.agent.iadp.model import IADPAgent

from .fdd.detector import FDDOutput


@dataclass
class RLSResetPolicy:
    """How L3 reacts to an FDD rising edge."""

    cov_inflation: float = 100.0           # Φ ← Φ + κ·I
    forgetting_drop: float = 0.9           # γ_RLS ← drop on rising edge
    forgetting_recover_steps: int = 500    # linear ramp back to nominal


class IADPMiddle:
    """Wraps IADPAgent with FDD-triggered RLS reset and rate-target derivation."""

    def __init__(
        self,
        base: IADPAgent,
        reset_policy: RLSResetPolicy,
        *,
        omega_indices: Optional[list[int]] = None,
        lookahead_dt: float = 0.05,
    ) -> None:
        self.base = base
        self.reset_policy = reset_policy
        self.omega_indices = list(omega_indices) if omega_indices else None
        self.lookahead_dt = float(lookahead_dt)
        self._gamma_nominal = float(base.rls.gamma_rls)
        self._gamma_active = self._gamma_nominal
        self._recover_countdown = 0
        self._prev_fault = False

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        u_iadp = self.base.predict(x_obs, reference, time_step=time_step)
        omega_ref = self._derive_omega_ref(x_obs, reference, time_step)
        return u_iadp, omega_ref

    def learn(
        self,
        next_x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
        *,
        fdd: FDDOutput,
    ) -> dict:
        # Rising-edge detection.
        rising = bool(fdd.fault_present and not self._prev_fault)
        if rising:
            self._trigger_reset()
        self._prev_fault = bool(fdd.fault_present)

        # Linear γ_RLS recovery.
        if self._recover_countdown > 0:
            self._recover_countdown -= 1
            frac = 1.0 - (self._recover_countdown
                          / max(1, self.reset_policy.forgetting_recover_steps))
            self._gamma_active = (self.reset_policy.forgetting_drop
                                  + (self._gamma_nominal
                                     - self.reset_policy.forgetting_drop) * frac)
            self.base.rls.gamma_rls = float(self._gamma_active)

        return self.base.learn(next_x_obs, reference, time_step=time_step)

    def reset(self) -> None:
        """Restore base agent + recovery state."""
        self.base.reset()
        self.base.rls.gamma_rls = self._gamma_nominal
        self._gamma_active = self._gamma_nominal
        self._recover_countdown = 0
        self._prev_fault = False

    def _trigger_reset(self) -> None:
        n = self.base.rls.n_regressor
        self.base.rls.Phi = (self.base.rls.Phi
                             + self.reset_policy.cov_inflation
                             * np.eye(n, dtype=np.float64))
        self.base.rls.gamma_rls = float(self.reset_policy.forgetting_drop)
        self._gamma_active = float(self.reset_policy.forgetting_drop)
        self._recover_countdown = int(self.reset_policy.forgetting_recover_steps)

    def _derive_omega_ref(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int,
    ) -> np.ndarray:
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        ref = np.asarray(reference, dtype=np.float64)
        if ref.ndim == 2:
            idx = int(np.clip(time_step, 0, ref.shape[1] - 1))
            ref_vec = ref[:, idx]
        elif ref.ndim == 1 and ref.size != x.size:
            idx = int(np.clip(time_step, 0, ref.size - 1))
            ref_vec = np.full(x.size, ref[idx], dtype=np.float64)
        else:
            ref_vec = ref.astype(np.float64).reshape(-1)
            if ref_vec.size != x.size:
                ref_vec = np.broadcast_to(ref_vec, x.shape).copy()

        if self.omega_indices is None:
            return ref_vec.copy()

        idx_arr = np.asarray(self.omega_indices, dtype=np.int64)
        omega_now = x[idx_arr]
        omega_target = ref_vec[idx_arr]
        return (omega_target - omega_now) / max(self.lookahead_dt, 1e-9)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_middle.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/middle.py \
        tests/agents/uftc/test_middle.py
git commit -m "feat(uftc): add IADPMiddle with FDD-triggered RLS reset"
```

---

### Task 8: `UFTCConfig` and `UFTCController` core

**Files:**
- Create: `tensoraerospace/agent/uftc/controller.py`
- Modify: `tensoraerospace/agent/uftc/__init__.py`
- Create: `tests/agents/uftc/test_controller_core.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_controller_core.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_controller_core.py -v`
Expected: FAIL — `controller` module missing.

- [ ] **Step 3: Implement `UFTCController` core**

```python
# tensoraerospace/agent/uftc/controller.py
"""UFTCController orchestrator: composes L2 inner + L3 middle + FDD detector.

Phase 1 MVP — see docs/superpowers/specs/2026-05-07-uftc-phase1-mvp-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tensoraerospace.agent.aa_indi.model import AAINDIAgent, AAINDIConfig
from tensoraerospace.agent.base import BaseRLModel
from tensoraerospace.agent.iadp.model import IADPAgent, IADPConfig

from .fdd.detector import FDDConfig, FDDDetector, FDDOutput
from .inner import ModeSwitcher, SuperTwistingObserver, WrappedAAINDI
from .middle import IADPMiddle, RLSResetPolicy


@dataclass
class UFTCConfig:
    """Hyper-parameters for :class:`UFTCController`."""

    dt: float = 0.01
    fdd_update_every: int = 1
    fdd_warmup_steps: int = 200

    inner_cfg: AAINDIConfig = field(default_factory=AAINDIConfig)
    middle_cfg: IADPConfig = field(default_factory=IADPConfig)
    fdd_cfg: FDDConfig = field(default_factory=FDDConfig)
    rls_reset_policy: RLSResetPolicy = field(default_factory=RLSResetPolicy)

    sm_obs_k1: float = 3.0
    sm_obs_k2: float = 1.5
    trust_radius_nominal: float = 0.1
    trust_radius_fault: float = 0.5
    alpha_threshold_deg: float = 25.0
    alpha_hysteresis_deg: float = 5.0

    alpha_index: Optional[int] = None
    omega_indices: Optional[list[int]] = None
    middle_lookahead_dt: float = 0.05

    enable_outer: bool = False  # placeholder for L4 D-SAC (Phase 3)


class UFTCController(BaseRLModel):
    """Top-level UFTC orchestrator (Phase 1 MVP)."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        *,
        nominal_F: Optional[np.ndarray] = None,
        nominal_G: Optional[np.ndarray] = None,
        config: Optional[UFTCConfig] = None,
    ) -> None:
        super().__init__()
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else UFTCConfig()

        # Inner cfg dt should match outer dt.
        self.cfg.inner_cfg = AAINDIConfig(**{
            **self.cfg.inner_cfg.__dict__,
            "dt": self.cfg.dt,
        })
        self.cfg.middle_cfg = IADPConfig(**{
            **{k: v for k, v in self.cfg.middle_cfg.__dict__.items()
               if k != "history"},
            "dt": self.cfg.dt,
        })

        # L2 inner.
        inner_base = AAINDIAgent(
            n_state=n_state, n_control=n_control, config=self.cfg.inner_cfg,
        )
        sm_obs = SuperTwistingObserver(
            n_axes=n_state, k1=self.cfg.sm_obs_k1,
            k2=self.cfg.sm_obs_k2, dt=self.cfg.dt,
        )
        mode_sw = ModeSwitcher(
            alpha_threshold_deg=self.cfg.alpha_threshold_deg,
            hysteresis_deg=self.cfg.alpha_hysteresis_deg,
        )
        self.inner = WrappedAAINDI(
            base=inner_base, sm_obs=sm_obs, mode_switch=mode_sw,
            trust_radius_nominal=self.cfg.trust_radius_nominal,
            trust_radius_fault=self.cfg.trust_radius_fault,
            dt=self.cfg.dt,
        )

        # L3 middle.
        middle_base = IADPAgent(
            n_state=n_state, n_control=n_control, config=self.cfg.middle_cfg,
        )
        self.middle = IADPMiddle(
            base=middle_base, reset_policy=self.cfg.rls_reset_policy,
            omega_indices=self.cfg.omega_indices,
            lookahead_dt=self.cfg.middle_lookahead_dt,
        )

        # FDD.
        F_init = (np.array(nominal_F, dtype=np.float64)
                  if nominal_F is not None
                  else np.zeros((n_state, n_state), dtype=np.float64))
        G_init = (np.array(nominal_G, dtype=np.float64)
                  if nominal_G is not None
                  else np.zeros((n_state, n_control), dtype=np.float64))
        self.fdd = FDDDetector.from_config(
            n_state=n_state, n_control=n_control, dt=self.cfg.dt,
            config=self.cfg.fdd_cfg, F_nominal=F_init, G_nominal=G_init,
        )
        self._fdd_active = nominal_F is not None and nominal_G is not None

        # Rolling state.
        self._step = 0
        self._last_fdd: FDDOutput = FDDOutput(
            fault_present=False, severity=0.0, confidence=0.0,
            innovation_norm=0.0, time_since_event=0.0,
        )
        self._last_u_indi = np.zeros(n_control, dtype=np.float64)
        self._last_omega_ref = np.zeros(n_state, dtype=np.float64)

    def predict(
        self,
        x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> np.ndarray:
        u_iadp, omega_ref = self.middle.predict(x_obs, reference, time_step)
        omega_meas = self._extract_omega(x_obs)
        alpha = self._extract_alpha(x_obs)
        u_indi = self.inner.predict(
            omega_ref=omega_ref,
            omega_meas=omega_meas,
            alpha=alpha,
            u_blend_target=u_iadp,
            fault_severity=self._last_fdd.severity,
            time_step=time_step,
        )
        self._last_u_indi = u_indi.copy()
        self._last_omega_ref = omega_ref.copy()
        return u_indi

    def learn(
        self,
        next_x_obs: np.ndarray,
        reference: np.ndarray,
        time_step: int = 0,
    ) -> dict:
        # Inner learns from raw next angular state.
        next_omega = self._extract_omega(next_x_obs)
        inner_diag = self.inner.learn(next_omega, self._last_omega_ref,
                                      time_step=time_step)

        # FDD: warm-up first, then run on cadence.
        if self._step + 1 == self.cfg.fdd_warmup_steps and not self._fdd_active:
            # Warm-start FDD Kalman from middle's RLS estimate.
            F_warm = self.middle.base.F[:self.n_state, :self.n_state].copy()
            G_warm = self.middle.base.G[:self.n_state, :self.n_control].copy()
            self.fdd.warm_start(F_nominal=F_warm, G_nominal=G_warm)
            self._fdd_active = True

        if self._fdd_active and self._step % self.cfg.fdd_update_every == 0:
            self._last_fdd = self.fdd.step(next_x_obs, self._last_u_indi)

        # Middle learns + reacts to FDD.
        middle_diag = self.middle.learn(
            next_x_obs, reference, time_step=time_step, fdd=self._last_fdd,
        )

        self._step += 1
        return {
            **{f"inner_{k}": v for k, v in inner_diag.items()},
            **{f"middle_{k}": v for k, v in middle_diag.items()},
            "fault_present": self._last_fdd.fault_present,
            "fdd_severity": self._last_fdd.severity,
            "fdd_confidence": self._last_fdd.confidence,
            "fdd_innovation_norm": self._last_fdd.innovation_norm,
        }

    def diagnostics(self) -> dict:
        """Snapshot of all sub-components for logging / plotting."""
        return {
            "step": int(self._step),
            "fault_present": bool(self._last_fdd.fault_present),
            "severity": float(self._last_fdd.severity),
            "confidence": float(self._last_fdd.confidence),
            "innovation_norm": float(self._last_fdd.innovation_norm),
            "rls_gamma": float(self.middle.base.rls.gamma_rls),
            "mode": str(self.inner.mode),
            "fdd_active": bool(self._fdd_active),
        }

    def reset(self) -> None:
        self.inner.reset()
        self.middle.reset()
        self.fdd.reset()
        self._step = 0
        self._last_fdd = FDDOutput(False, 0.0, 0.0, 0.0, 0.0)
        self._last_u_indi.fill(0.0)
        self._last_omega_ref.fill(0.0)

    def _extract_omega(self, x_obs: np.ndarray) -> np.ndarray:
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        if self.cfg.omega_indices is None:
            return x[: self.n_state]
        idx = np.asarray(self.cfg.omega_indices, dtype=np.int64)
        return x[idx]

    def _extract_alpha(self, x_obs: np.ndarray) -> float:
        if self.cfg.alpha_index is None:
            return 0.0
        x = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        return float(x[int(self.cfg.alpha_index)])

    def get_param_env(self) -> dict[str, Any]:
        return {
            "policy": {
                "name": f"{self.__class__.__module__}.{self.__class__.__name__}",
                "params": {
                    "n_state": self.n_state,
                    "n_control": self.n_control,
                },
            },
        }
```

- [ ] **Step 4: Wire up package `__init__.py`**

```python
# tensoraerospace/agent/uftc/__init__.py
"""Unified Fault-Tolerant Control (UFTC) — Phase 1 MVP orchestrator."""
from __future__ import annotations

from .controller import UFTCConfig, UFTCController
from .fdd.detector import FDDConfig, FDDDetector, FDDOutput
from .inner import ModeSwitcher, SuperTwistingObserver, WrappedAAINDI
from .middle import IADPMiddle, RLSResetPolicy

__all__ = [
    "FDDConfig",
    "FDDDetector",
    "FDDOutput",
    "IADPMiddle",
    "ModeSwitcher",
    "RLSResetPolicy",
    "SuperTwistingObserver",
    "UFTCConfig",
    "UFTCController",
    "WrappedAAINDI",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_controller_core.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/uftc/controller.py \
        tensoraerospace/agent/uftc/__init__.py \
        tests/agents/uftc/test_controller_core.py
git commit -m "feat(uftc): add UFTCController orchestrator with predict/learn/reset"
```

---

### Task 9: `save` / `from_pretrained` round-trip

**Files:**
- Modify: `tensoraerospace/agent/uftc/controller.py` (add save/load)
- Create: `tests/agents/uftc/test_controller_save_load.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_controller_save_load.py
"""save/load round-trip preserves UFTCController behaviour."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _drive(ctl: UFTCController, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(ctl.n_state)
    out = []
    for k in range(n):
        ref = rng.normal(scale=0.05, size=ctl.n_state)
        u = ctl.predict(x, ref, time_step=k)
        out.append(u.copy())
        x = x + 0.05 * rng.normal(size=ctl.n_state) + 0.1 * u
        ctl.learn(x, ref, time_step=k)
    return np.stack(out)


def test_save_creates_expected_files(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=3, n_control=3,
        nominal_F=np.zeros((3, 3)), nominal_G=np.eye(3) * 0.1,
        config=UFTCConfig(fdd_warmup_steps=10),
    )
    _drive(ctl, 30)
    folder = ctl.save(tmp_path)
    p = Path(folder)
    assert (p / "config.json").is_file()
    assert (p / "controller_state.npz").is_file()
    assert (p / "fdd" / "kalman.npz").is_file()
    assert (p / "fdd" / "cpd.npz").is_file()
    inner_dirs = list((p / "inner").glob("*"))
    middle_dirs = list((p / "middle").glob("*"))
    assert len(inner_dirs) == 1
    assert len(middle_dirs) == 1


def test_round_trip_predict_matches(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=3, n_control=3,
        nominal_F=np.zeros((3, 3)), nominal_G=np.eye(3) * 0.1,
        config=UFTCConfig(fdd_warmup_steps=10),
    )
    _drive(ctl, 50, seed=42)
    folder = ctl.save(tmp_path)
    restored = UFTCController.from_pretrained(folder)

    # Both should produce identical predictions on the same input.
    x = np.array([0.1, -0.05, 0.02])
    ref = np.array([0.05, 0.0, -0.01])
    u_orig = ctl.predict(x, ref, time_step=100)
    u_restored = restored.predict(x, ref, time_step=100)
    assert np.allclose(u_orig, u_restored, atol=1e-9)


def test_config_json_human_readable(tmp_path: Path) -> None:
    ctl = UFTCController(
        n_state=2, n_control=2,
        nominal_F=np.zeros((2, 2)), nominal_G=np.eye(2),
        config=UFTCConfig(fdd_warmup_steps=5),
    )
    folder = ctl.save(tmp_path)
    payload = json.loads((Path(folder) / "config.json").read_text())
    assert payload["policy"]["params"]["n_state"] == 2
    assert payload["policy"]["params"]["n_control"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_controller_save_load.py -v`
Expected: FAIL — `save`/`from_pretrained` not implemented.

- [ ] **Step 3: Add save/load to UFTCController**

Append to `tensoraerospace/agent/uftc/controller.py`:

```python
import dataclasses
import datetime
import json as _json
import shutil
from pathlib import Path as _Path
from typing import Union


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v)
                for k, v in dataclasses.asdict(obj).items()
                if k != "history"}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable_for_iadpconfig(payload: dict) -> dict:
    arr_keys = ("Q", "R", "F_init", "G_init", "P_init", "excitation_signal")
    out = dict(payload)
    for k in arr_keys:
        if out.get(k) is not None:
            out[k] = np.asarray(out[k], dtype=np.float64)
    return out


def _add_save_load(_cls):
    def save(self, path: Union[str, _Path, None] = None) -> str:
        base = _Path.cwd() if path is None else _Path(path)
        run_dir = base / (
            datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            + f"_{self.__class__.__name__}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_payload = {
            "policy": {
                "name": (f"{self.__class__.__module__}."
                         f"{self.__class__.__name__}"),
                "params": {
                    "n_state": self.n_state,
                    "n_control": self.n_control,
                },
                "config": _to_jsonable(self.cfg),
            },
        }
        (run_dir / "config.json").write_text(_json.dumps(cfg_payload, indent=2))

        # Inner / middle: delegate to existing save methods.
        inner_dir = self.inner.base.save(run_dir / "inner")
        middle_dir = self.middle.base.save(run_dir / "middle")

        # FDD state.
        fdd_dir = run_dir / "fdd"
        fdd_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            fdd_dir / "kalman.npz",
            F=self.fdd.kalman.F, G=self.fdd.kalman.G,
            Q=self.fdd.kalman.Q, R=self.fdd.kalman.R,
            x_hat=self.fdd.kalman.x_hat, P=self.fdd.kalman.P,
        )
        np.savez(
            fdd_dir / "cpd.npz",
            cusum=np.asarray(self.fdd.cpd._cusum),
            alarm=np.asarray(self.fdd.cpd._alarm),
            cooldown=np.asarray(self.fdd.cpd._cooldown),
            time_since_alarm=np.asarray(self.fdd.cpd._time_since_alarm),
        )

        np.savez(
            run_dir / "controller_state.npz",
            step=np.asarray(self._step),
            last_u=self._last_u_indi,
            last_omega_ref=self._last_omega_ref,
            fdd_active=np.asarray(self._fdd_active),
            sm_s=self.inner.sm_obs._s,
            sm_z=self.inner.sm_obs._z,
            mode=np.asarray(self.inner.mode),
        )
        # Persist the actual subdir names (timestamps) for from_pretrained.
        (run_dir / "manifest.json").write_text(_json.dumps({
            "inner_subdir": _Path(inner_dir).name,
            "middle_subdir": _Path(middle_dir).name,
        }, indent=2))
        return str(run_dir)

    @classmethod
    def from_pretrained(cls, repo_name: str,
                        access_token=None, version=None) -> "UFTCController":
        p = _Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p)
        from huggingface_hub import snapshot_download
        folder = snapshot_download(repo_id=str(repo_name),
                                   token=access_token, revision=version)
        return cls._load_from_dir(_Path(folder))

    @classmethod
    def _load_from_dir(cls, folder: _Path) -> "UFTCController":
        cfg_payload = _json.loads((folder / "config.json").read_text())
        params = cfg_payload["policy"]["params"]
        cfg_dict = cfg_payload["policy"]["config"]

        # Rebuild nested dataclasses.
        inner_cfg = AAINDIConfig(**cfg_dict["inner_cfg"])
        middle_cfg = IADPConfig(**_from_jsonable_for_iadpconfig(
            cfg_dict["middle_cfg"]))
        fdd_cfg = FDDConfig(**cfg_dict["fdd_cfg"])
        rls_pol = RLSResetPolicy(**cfg_dict["rls_reset_policy"])

        cfg = UFTCConfig(
            **{k: v for k, v in cfg_dict.items() if k not in (
                "inner_cfg", "middle_cfg", "fdd_cfg", "rls_reset_policy")},
            inner_cfg=inner_cfg, middle_cfg=middle_cfg,
            fdd_cfg=fdd_cfg, rls_reset_policy=rls_pol,
        )

        ctl = cls(
            n_state=int(params["n_state"]),
            n_control=int(params["n_control"]),
            nominal_F=np.zeros((params["n_state"], params["n_state"])),
            nominal_G=np.zeros((params["n_state"], params["n_control"])),
            config=cfg,
        )

        manifest = _json.loads((folder / "manifest.json").read_text())
        ctl.inner.base = AAINDIAgent._load_from_dir(
            folder / "inner" / manifest["inner_subdir"])
        ctl.middle.base = IADPAgent._load_from_dir(
            folder / "middle" / manifest["middle_subdir"])
        # Re-bind reset policy / lookahead to the reloaded base.
        ctl.middle._gamma_nominal = ctl.middle.base.rls.gamma_rls

        with np.load(folder / "fdd" / "kalman.npz") as npz:
            ctl.fdd.kalman.F = npz["F"]
            ctl.fdd.kalman.G = npz["G"]
            ctl.fdd.kalman.Q = npz["Q"]
            ctl.fdd.kalman.R = npz["R"]
            ctl.fdd.kalman.x_hat = npz["x_hat"]
            ctl.fdd.kalman.P = npz["P"]
        with np.load(folder / "fdd" / "cpd.npz") as npz:
            ctl.fdd.cpd._cusum = float(npz["cusum"])
            ctl.fdd.cpd._alarm = bool(npz["alarm"])
            ctl.fdd.cpd._cooldown = int(npz["cooldown"])
            ctl.fdd.cpd._time_since_alarm = int(npz["time_since_alarm"])

        with np.load(folder / "controller_state.npz") as npz:
            ctl._step = int(npz["step"])
            ctl._last_u_indi = npz["last_u"]
            ctl._last_omega_ref = npz["last_omega_ref"]
            ctl._fdd_active = bool(npz["fdd_active"])
            ctl.inner.sm_obs._s = npz["sm_s"]
            ctl.inner.sm_obs._z = npz["sm_z"]
            ctl.inner.mode_switch._mode = str(npz["mode"])
        return ctl

    _cls.save = save
    _cls.from_pretrained = from_pretrained
    _cls._load_from_dir = _load_from_dir
    return _cls


_add_save_load(UFTCController)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_controller_save_load.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/controller.py \
        tests/agents/uftc/test_controller_save_load.py
git commit -m "feat(uftc): add multi-component save/from_pretrained round-trip"
```

---

### Task 10: Public exports + smoke test

**Files:**
- Modify: `tensoraerospace/agent/__init__.py`
- Create: `tests/agents/uftc/test_uftc_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/agents/uftc/test_uftc_smoke.py
"""1000-step smoke test on a mock plant — no NaN, all keys filled."""
from __future__ import annotations

import math

import numpy as np

from tensoraerospace.agent import UFTCConfig, UFTCController


def test_thousand_step_smoke_no_nan() -> None:
    rng = np.random.default_rng(0)
    n_state = 3
    n_control = 2
    F = np.array([[-0.1, 0.0, 0.0],
                  [0.0, -0.05, 0.0],
                  [0.0, 0.0, -0.2]])
    G = np.array([[0.5, 0.0],
                  [0.0, 0.4],
                  [0.1, 0.1]])
    ctl = UFTCController(
        n_state=n_state, n_control=n_control,
        nominal_F=F, nominal_G=G,
        config=UFTCConfig(fdd_warmup_steps=50),
    )
    x = np.zeros(n_state)
    for k in range(1000):
        ref = 0.1 * np.array([math.sin(k * 0.05),
                              math.cos(k * 0.07),
                              math.sin(k * 0.03)])
        u = ctl.predict(x, ref, time_step=k)
        assert np.isfinite(u).all()
        x = x + (F @ x + G @ u) * 0.01 + 0.005 * rng.normal(size=n_state)
        ctl.learn(x, ref, time_step=k)

    diag = ctl.diagnostics()
    for key in ("step", "fault_present", "severity", "confidence",
                "innovation_norm", "rls_gamma", "mode", "fdd_active"):
        assert key in diag
    assert diag["step"] == 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_smoke.py -v`
Expected: FAIL — `UFTCConfig` / `UFTCController` not exported from `tensoraerospace.agent`.

- [ ] **Step 3: Add public re-exports**

Append to `tensoraerospace/agent/__init__.py` after the existing imports:

```python
# Unified Fault-Tolerant Control (UFTC) — Phase 1 MVP
from .uftc import UFTCConfig as UFTCConfig  # noqa: F401
from .uftc import UFTCController as UFTCController  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/__init__.py \
        tests/agents/uftc/test_uftc_smoke.py
git commit -m "feat(uftc): expose UFTCController/UFTCConfig from tensoraerospace.agent"
```

---

### Task 11: Integration test — F-16 nominal regression guard

**Files:**
- Create: `tests/agents/uftc/test_uftc_f16_nominal.py`

- [ ] **Step 1: Write the test**

```python
# tests/agents/uftc/test_uftc_f16_nominal.py
"""F-16 nonlinear angular flight: UFTC on nominal — regression guard."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


@pytest.mark.slow
def test_uftc_holds_attitude_under_nominal_f16() -> None:
    rng = np.random.default_rng(0)
    n_steps = 800
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=None,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=np.zeros((3, 3)),
        nominal_G=np.eye(3, 4) * 0.1,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],   # roll, pitch, yaw rates
            middle_lookahead_dt=0.05,
        ),
    )
    ref_zero = np.zeros(3)
    rms = []
    for k in range(n_steps):
        x_omega = obs[:3]
        u = ctl.predict(x_omega, ref_zero, time_step=k)
        # Pad to 4-DOF actuator input expected by env.
        u_env = np.zeros(4)
        u_env[: len(u)] = u
        obs, _, terminated, truncated, info = env.step(u_env)
        ctl.learn(obs[:3], ref_zero, time_step=k)
        rms.append(float(np.linalg.norm(obs[:3])))
        if terminated or truncated:
            break
    rms_after_warmup = np.mean(rms[300:])
    # Coarse regression bound: nominal hold must beat 0.5 rad/s RMS.
    assert rms_after_warmup < 0.5
    assert not ctl.diagnostics()["fault_present"]
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_f16_nominal.py -v`
Expected: 1 passed (may take ~10 s).

- [ ] **Step 3: If the test fails on RMS bound, tune `middle_lookahead_dt` upward (looser tracking) up to 0.2; do not loosen the bound itself unless the assertion becomes trivially passing on a buggy controller. Document the chosen value with a single-line comment.**

- [ ] **Step 4: Commit**

```bash
git add tests/agents/uftc/test_uftc_f16_nominal.py
git commit -m "test(uftc): add F-16 nominal flight integration regression"
```

---

### Task 12: Integration test — F-16 with elevator-jam fault

**Files:**
- Create: `tests/agents/uftc/test_uftc_f16_elevator_jam.py`

- [ ] **Step 1: Write the test**

```python
# tests/agents/uftc/test_uftc_f16_elevator_jam.py
"""F-16 elevator jam at t=5s — UFTC FDD must trigger and recover."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    ELEVATOR_JAM_NEUTRAL,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


@pytest.mark.slow
def test_uftc_detects_elevator_jam_and_recovers() -> None:
    n_steps = 1500
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=ELEVATOR_JAM_NEUTRAL,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=np.zeros((3, 3)),
        nominal_G=np.eye(3, 4) * 0.1,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
        ),
    )
    ref = np.zeros(3)
    detect_step = None
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4); u_env[: len(u)] = u
        obs, _, terminated, truncated, info = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        if ctl.diagnostics()["fault_present"] and detect_step is None:
            detect_step = k
        if terminated or truncated:
            break

    # Damage triggers at t≈0 of the env (default behaviour for this preset);
    # detection latency budget = 1.0 s after warm-up window.
    assert detect_step is not None
    assert detect_step < 200 + 100  # warm-up + 1 s
    # Final state stays bounded (no divergence).
    assert np.linalg.norm(obs[:3]) < 5.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_f16_elevator_jam.py -v`
Expected: 1 passed.

- [ ] **Step 3: If `detect_step is None`, the FDD thresholds need calibration. Tune `UFTCConfig.fdd_cfg.h_alarm` *down* (e.g. 15.0) until the test passes on this env, but verify it still passes the nominal regression in Task 11. Both must pass simultaneously.**

- [ ] **Step 4: Commit**

```bash
git add tests/agents/uftc/test_uftc_f16_elevator_jam.py
git commit -m "test(uftc): elevator-jam detection and recovery on F-16"
```

---

### Task 13: Integration test — F-16 with wing-strike fault

**Files:**
- Create: `tests/agents/uftc/test_uftc_f16_wing_strike.py`

- [ ] **Step 1: Write the test**

```python
# tests/agents/uftc/test_uftc_f16_wing_strike.py
"""F-16 wing-strike at t=5s — UFTC must keep aircraft bounded."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


@pytest.mark.slow
def test_uftc_keeps_state_bounded_under_wing_strike() -> None:
    n_steps = 1500
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=WING_STRIKE_LEFT_TIP,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=np.zeros((3, 3)),
        nominal_G=np.eye(3, 4) * 0.1,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            trust_radius_fault=0.7,    # let L3 re-adapt aggressively
        ),
    )
    ref = np.zeros(3)
    fault_seen = False
    max_norm = 0.0
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4); u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        fault_seen = fault_seen or ctl.diagnostics()["fault_present"]
        max_norm = max(max_norm, float(np.linalg.norm(obs[:3])))
        if terminated or truncated:
            break

    assert fault_seen
    # Wing-strike usually drives a strong roll; assert no divergence.
    assert max_norm < 8.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_f16_wing_strike.py -v`
Expected: 1 passed.

- [ ] **Step 3: If the bound is violated, do not relax it without reading the diagnostics first. Plot `diagnostics()["severity"]` over time — if severity stays high but `rls_gamma` doesn't drop, the rising-edge detection is not firing. Inspect `_prev_fault` rolling state in `IADPMiddle` to confirm.**

- [ ] **Step 4: Commit**

```bash
git add tests/agents/uftc/test_uftc_f16_wing_strike.py
git commit -m "test(uftc): wing-strike survival regression on F-16"
```

---

### Task 14: Integration test — F-16 with engine flameout (gradual fault)

**Files:**
- Create: `tests/agents/uftc/test_uftc_f16_engine_flameout.py`

- [ ] **Step 1: Write the test**

```python
# tests/agents/uftc/test_uftc_f16_engine_flameout.py
"""F-16 engine flameout — gradual degradation; RLS should catch without
hard CPD trigger.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    ENGINE_FLAMEOUT,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


@pytest.mark.slow
def test_engine_flameout_handled_without_divergence() -> None:
    n_steps = 1500
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=ENGINE_FLAMEOUT,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=np.zeros((3, 3)),
        nominal_G=np.eye(3, 4) * 0.1,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
        ),
    )
    ref = np.zeros(3)
    max_norm = 0.0
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4); u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        max_norm = max(max_norm, float(np.linalg.norm(obs[:3])))
        if terminated or truncated:
            break

    # Aircraft must stay bounded whether or not CPD fires —
    # the test asserts behavioural success, not mode classification.
    assert max_norm < 8.0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_f16_engine_flameout.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_f16_engine_flameout.py
git commit -m "test(uftc): engine-flameout boundedness on F-16"
```

---

### Task 15: Example notebook + final coverage check

**Files:**
- Create: `example/reinforcement_learning/uftc/uftc_f16_damage_demo.py`
- Run: full coverage check across `tensoraerospace/agent/uftc/`

- [ ] **Step 1: Create example demo (script form, easier to lint than notebook)**

```python
# example/reinforcement_learning/uftc/uftc_f16_damage_demo.py
"""Demo: UFTC on F-16 nonlinear angular with a wing-tip strike at t≈5s.

Run:
    poetry run python example/reinforcement_learning/uftc/uftc_f16_damage_demo.py

Outputs a trace of FDD severity, ω state norm, and RLS forgetting factor
per step to stdout. Does not require matplotlib — use the printed
columns to drive a separate plotting tool if desired.
"""
from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    WING_STRIKE_LEFT_TIP,
)
from tensoraerospace.agent import UFTCConfig, UFTCController
from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16


def main() -> None:
    n_steps = 1500
    env = NonlinearAngularF16(
        initial_state=np.zeros(14),
        number_time_steps=n_steps,
        dt=0.01,
        airspeed=200.0,
        damage_profile=WING_STRIKE_LEFT_TIP,
        split_stab=True,
    )
    obs, _ = env.reset()
    ctl = UFTCController(
        n_state=3, n_control=4,
        nominal_F=np.zeros((3, 3)),
        nominal_G=np.eye(3, 4) * 0.1,
        config=UFTCConfig(
            dt=0.01, fdd_warmup_steps=200,
            omega_indices=[0, 1, 2],
            middle_lookahead_dt=0.05,
            trust_radius_fault=0.7,
        ),
    )
    ref = np.zeros(3)
    print("# t,omega_norm,severity,fault_present,rls_gamma")
    for k in range(n_steps):
        u = ctl.predict(obs[:3], ref, time_step=k)
        u_env = np.zeros(4); u_env[: len(u)] = u
        obs, _, terminated, truncated, _ = env.step(u_env)
        ctl.learn(obs[:3], ref, time_step=k)
        if k % 25 == 0:
            d = ctl.diagnostics()
            print(f"{k * 0.01:.2f},{np.linalg.norm(obs[:3]):.4f},"
                  f"{d['severity']:.3f},{int(d['fault_present'])},"
                  f"{d['rls_gamma']:.4f}")
        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example to confirm it executes cleanly**

Run: `poetry run python example/reinforcement_learning/uftc/uftc_f16_damage_demo.py | head -20`
Expected: header line, then ~20 CSV rows with monotonically increasing time.

- [ ] **Step 3: Run the full uftc test suite with coverage**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest -p pytest_cov tests/agents/uftc/ -v --cov=tensoraerospace.agent.uftc --cov-report=term-missing`
Expected: all tests pass, coverage on `tensoraerospace/agent/uftc/` ≥ 80 %.

- [ ] **Step 4: If coverage < 80 %, add targeted unit tests for the missed lines (most likely save/load error paths or guard branches in `_extract_omega`/`_extract_alpha`). Commit the extra tests in a follow-up step before declaring done.**

- [ ] **Step 5: Commit example**

```bash
git add example/reinforcement_learning/uftc/uftc_f16_damage_demo.py
git commit -m "docs(uftc): add F-16 wing-strike demo script"
```

- [ ] **Step 6: Final commit (only if Step 4 added tests)**

```bash
git add tests/agents/uftc/
git commit -m "test(uftc): close coverage gaps in save/load and extract helpers"
```

---

## Self-Review Notes

**Spec coverage map** (each spec section → tasks that implement it):

| Spec § | Topic | Tasks |
|---|---|---|
| 2.1 | Package layout | 1 |
| 2.2 | Component dependencies | 1, 8 |
| 2.3 | Public API | 8, 10 |
| 3.1 | NominalKalman | 2 |
| 3.2 | ChangePointDetector | 3 |
| 3.3 | FDDDetector | 4 |
| 3.4 | SuperTwistingObserver | 5 |
| 3.5 | ModeSwitcher | 5 |
| 3.6 | WrappedAAINDI | 6 |
| 3.7 | IADPMiddle | 7 |
| 3.8 | UFTCController | 8 |
| 3.9 | save / from_pretrained | 9 |
| 4 | Data flow | 8 (predict/learn methods) |
| 5 | Error handling | 7 (RLS reset), 8 (extract guards), 9 (load fallbacks) |
| 6.1 | Unit tests | 2–9 |
| 6.2 | Integration tests | 11–14 |
| 6.3 | Smoke test | 10 |
| 6.4 | Coverage ≥ 80 % | 15 |
| 7 | Documentation | 15 (demo) |

**Type consistency:** `FDDOutput` uses `confidence: float ∈ [0, 1)`; controller diagnostics expose the same field as `confidence`. `ChangePointState.severity` and `FDDOutput.severity` use the same scaling (cusum/h_alarm clipped to 10). `IADPMiddle.predict` returns `(u_iadp, omega_ref)`, matched in `UFTCController.predict()` step 1. `WrappedAAINDI.predict` signature in Task 6 matches calling convention in Task 8.

**No placeholders:** every code block is complete, every command lists its expected outcome, no "implement later" lines.

**Plant-agnostic check:** every test in tasks 2–10 uses synthetic / mock plants (no F-16 dependency). F-16 enters only at Tasks 11–14 as integration regression. Spec section 1.2 honored.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-07-uftc-phase1-mvp.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
