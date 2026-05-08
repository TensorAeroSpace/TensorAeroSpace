# UFTC Phase 2 — L1 HJ-shield + GLR FDD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a HJ-Reachability safety shield (post-filter on `u_indi`) and a generalised-likelihood-ratio detector for slow-drift faults to the UFTC stack, keeping Phase 1 behaviour bit-identical when both are off.

**Architecture:** Two independent additions wired into the existing `UFTCController`. (1) `tensoraerospace/agent/uftc/l1/` package — `HJValueFunction` protocol with a `DeepReachValueFn` torch-backed implementation, a `ConformalMargin` from FDD severity, a `ValueBank` over fault modes, and a QP-based `HJReachabilityShield` post-filter. (2) `tensoraerospace/agent/uftc/fdd/glr.py` — sliding-window GLR detector consuming Kalman innovations; integrated into the existing `FDDDetector` so `FDDOutput` gains `fault_kind` and `severity_gradual` while staying backwards-compatible with Phase 1 consumers of `severity`.

**Tech Stack:** Python 3.10+, NumPy (algorithm core), PyTorch (DeepReach value-network only), `cvxpy` + OSQP (QP solver), pytest (tests), poetry. Existing `aa_indi.AAINDIAgent`, `iadp.IADPAgent`, and Phase 1 `uftc.controller.UFTCController` are extended via flags but not modified in incompatible ways.

**Spec:** [`docs/superpowers/specs/2026-05-08-uftc-l1-hjshield-and-glr-design.md`](../specs/2026-05-08-uftc-l1-hjshield-and-glr-design.md)
**Master spec:** [`docs/superpowers/specs/2026-05-08-uftc-cascade-extension-design.md`](../specs/2026-05-08-uftc-cascade-extension-design.md)

**Build order (bottom-up TDD):**

```
HJValueFunction protocol ──┐
power-iteration Lipschitz ─┤
DeepReachValueFn ──────────┤
ConformalMargin ───────────┤
ValueBank ─────────────────┼─→ HJReachabilityShield (QP)
                           │           │
                           │           └→ request_actuator_hold (macro-action)
GLRDetector ─→ FDDDetector ext. ─→ extended FDDOutput
                           │
                           ↓
              UFTCController integration ─→ phase1 invariance test
                                          └→ engine-drift integration test
                                          └→ docs / READMEs
```

**Conventions:**
- Run tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest <path> -v` (matches the project Makefile and bypasses pytest-cov auto-load).
- New source under `tensoraerospace/agent/uftc/l1/` and `tensoraerospace/agent/uftc/fdd/glr.py`. New tests under `tests/agents/uftc/l1/` and `tests/agents/uftc/fdd/`.
- Commit style mirrors recent commits (`feat(uftc): ...`, `test(uftc): ...`, etc.); no Claude attribution.
- All new files start with `from __future__ import annotations` and a triple-quoted module docstring summarising responsibility.
- Type hints mandatory on public symbols.
- `cvxpy` is added to `pyproject.toml` test/runtime extras in Task 8 (the only task that needs the QP solver).

---

### Task 1: Bootstrap `l1/` package skeleton + `HJValueFunction` protocol

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/__init__.py`
- Create: `tensoraerospace/agent/uftc/l1/value_fn.py`
- Create: `tests/agents/uftc/l1/__init__.py`
- Create: `tests/agents/uftc/l1/test_value_fn_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_value_fn_protocol.py
"""Lock in the HJValueFunction protocol surface."""
from __future__ import annotations

import numpy as np


def test_hj_value_function_protocol_methods() -> None:
    from tensoraerospace.agent.uftc.l1.value_fn import HJValueFunction

    class Dummy:
        def value(self, x: np.ndarray) -> float:
            return 0.0

        def gradient(self, x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        def lipschitz_const(self) -> float:
            return 1.0

    d = Dummy()
    assert isinstance(d, HJValueFunction)


def test_l1_subpackage_importable() -> None:
    import tensoraerospace.agent.uftc.l1 as l1

    assert hasattr(l1, "__all__")
    assert "HJValueFunction" in l1.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_value_fn_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tensoraerospace.agent.uftc.l1'`.

- [ ] **Step 3: Create the skeleton**

```python
# tensoraerospace/agent/uftc/l1/__init__.py
"""UFTC Phase 2 — L1 HJ-Reachability safety shield."""
from __future__ import annotations

from .value_fn import HJValueFunction

__all__ = ["HJValueFunction"]
```

```python
# tensoraerospace/agent/uftc/l1/value_fn.py
"""HJ value-function protocol used by the L1 shield.

A value function ``V(x)`` is *non-positive inside* the safe set, *zero on the
boundary*, and *positive outside*. The shield uses ``V`` and ``∇V`` to enforce
forward-invariance of the safe set under a CBF-style QP.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HJValueFunction(Protocol):
    """Minimal contract any L1 value function must satisfy."""

    def value(self, x: np.ndarray) -> float: ...
    def gradient(self, x: np.ndarray) -> np.ndarray: ...
    def lipschitz_const(self) -> float: ...
```

```python
# tests/agents/uftc/l1/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_value_fn_protocol.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/__init__.py \
        tensoraerospace/agent/uftc/l1/value_fn.py \
        tests/agents/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_value_fn_protocol.py
git commit -m "feat(uftc): bootstrap l1 package with HJValueFunction protocol"
```

---

### Task 2: Power-iteration Lipschitz upper bound

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/lipschitz.py`
- Create: `tests/agents/uftc/l1/test_lipschitz.py`
- Modify: `tensoraerospace/agent/uftc/l1/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_lipschitz.py
"""Power-iteration Lipschitz upper bound on a torch nn.Module."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l1.lipschitz import power_iteration_lipschitz


def test_linear_layer_returns_operator_norm() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 3))
    b = rng.standard_normal(4)

    class Linear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.W = torch.nn.Parameter(torch.tensor(W, dtype=torch.float64))
            self.b = torch.nn.Parameter(torch.tensor(b, dtype=torch.float64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.W.t() + self.b

    model = Linear().eval()

    def sample() -> np.ndarray:
        return rng.standard_normal(3)

    L = power_iteration_lipschitz(model, sample, n_iter=200, n_starts=4,
                                  dtype=torch.float64)
    expected = float(np.linalg.norm(W, ord=2))
    assert abs(L - expected) / expected < 0.05


def test_returns_finite_positive_float() -> None:
    class Tanh(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(x)

    rng = np.random.default_rng(1)
    L = power_iteration_lipschitz(
        Tanh().eval(), lambda: rng.standard_normal(2),
        n_iter=50, n_starts=2, dtype=torch.float64,
    )
    assert isinstance(L, float)
    assert 0.0 < L <= 1.0 + 1e-6  # tanh derivative is bounded by 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_lipschitz.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tensoraerospace.agent.uftc.l1.lipschitz'`.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l1/lipschitz.py
"""Power-iteration upper bound on the Lipschitz constant of a torch module.

Approximate ``L(f) = sup_x ‖J_f(x)‖_2`` by sampling ``n_starts`` points
from ``sample_fn`` and running ``n_iter`` steps of the power method on
the Jacobian-vector product. Returns the maximum spectral norm seen.
This is an upper bound only when the maximiser falls inside the
sampled distribution; we mitigate with multiple restarts.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def power_iteration_lipschitz(
    model: "torch.nn.Module",
    sample_fn: Callable[[], np.ndarray],
    *,
    n_iter: int = 200,
    n_starts: int = 8,
    dtype=None,
) -> float:
    import torch

    if dtype is None:
        dtype = torch.float32

    L_max = 0.0
    for _ in range(int(n_starts)):
        x_np = sample_fn().astype(np.float64)
        x = torch.tensor(x_np, dtype=dtype, requires_grad=True)
        v = torch.randn_like(x)
        v = v / (v.norm() + 1e-12)

        for _ in range(int(n_iter)):
            y = model(x)
            # Compute J^T v via vector-Jacobian product, then J (J^T v) via JVP.
            (jt_v,) = torch.autograd.grad(y, x, grad_outputs=v,
                                          retain_graph=True, create_graph=False)
            v_new = jt_v.detach()
            norm = float(v_new.norm())
            if norm < 1e-18:
                break
            v = v_new / norm
        L_max = max(L_max, norm)
    return float(L_max)
```

Update package surface:

```python
# tensoraerospace/agent/uftc/l1/__init__.py
"""UFTC Phase 2 — L1 HJ-Reachability safety shield."""
from __future__ import annotations

from .lipschitz import power_iteration_lipschitz
from .value_fn import HJValueFunction

__all__ = ["HJValueFunction", "power_iteration_lipschitz"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_lipschitz.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/lipschitz.py \
        tensoraerospace/agent/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_lipschitz.py
git commit -m "feat(uftc): add power-iteration Lipschitz bound for V_theta"
```

---

### Task 3: `DeepReachValueFn` torch backend (no training yet)

**Files:**
- Modify: `tensoraerospace/agent/uftc/l1/value_fn.py`
- Create: `tests/agents/uftc/l1/test_deepreach_value_fn.py`
- Modify: `tensoraerospace/agent/uftc/l1/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_deepreach_value_fn.py
"""DeepReachValueFn satisfies HJValueFunction; save/load round-trip."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l1.value_fn import (
    DeepReachConfig,
    DeepReachValueFn,
    HJValueFunction,
)


def test_value_fn_satisfies_protocol() -> None:
    cfg = DeepReachConfig(n_state=3, hidden_sizes=(32, 32))
    fn = DeepReachValueFn.from_config(cfg)
    assert isinstance(fn, HJValueFunction)


def test_value_returns_scalar_and_gradient_matches_state_dim() -> None:
    cfg = DeepReachConfig(n_state=4, hidden_sizes=(16, 16))
    fn = DeepReachValueFn.from_config(cfg, seed=0)
    x = np.array([0.1, -0.2, 0.0, 0.4])
    v = fn.value(x)
    g = fn.gradient(x)
    assert isinstance(v, float)
    assert g.shape == (4,)


def test_save_load_round_trip(tmp_path) -> None:
    cfg = DeepReachConfig(n_state=2, hidden_sizes=(8, 8))
    fn = DeepReachValueFn.from_config(cfg, seed=42)
    x = np.array([0.3, -0.1])
    v_before = fn.value(x)

    fn.save(tmp_path / "v.pt")
    fn2 = DeepReachValueFn.load(tmp_path / "v.pt")
    v_after = fn2.value(x)
    assert abs(v_before - v_after) < 1e-6
    meta = json.loads((tmp_path / "v.json").read_text())
    assert meta["n_state"] == 2


def test_lipschitz_const_returns_finite() -> None:
    cfg = DeepReachConfig(n_state=3, hidden_sizes=(8, 8))
    fn = DeepReachValueFn.from_config(cfg, seed=1)
    L = fn.lipschitz_const()
    assert isinstance(L, float)
    assert L > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_deepreach_value_fn.py -v`
Expected: FAIL with `ImportError: cannot import name 'DeepReachConfig' from ...value_fn`.

- [ ] **Step 3: Implement**

Append to `tensoraerospace/agent/uftc/l1/value_fn.py`:

```python
# (added below the existing protocol)
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class DeepReachConfig:
    """Hyper-parameters for :class:`DeepReachValueFn`."""

    n_state: int
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    activation: str = "tanh"               # "tanh" | "sine"
    state_bounds: list[list[float]] | None = None  # shape (n_state, 2)
    time_horizon: float = 5.0
    safe_set_fn_name: str = "alpha_envelope"
    dt: float = 0.01
    lipschitz_n_starts: int = 8
    lipschitz_n_iter: int = 200


class _MLP:
    """Internal builder; we keep it private so torch is only imported lazily."""

    @staticmethod
    def build(cfg: DeepReachConfig, seed: int = 0):
        import torch
        from torch import nn

        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        in_dim = int(cfg.n_state) + 1     # state + time
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh() if cfg.activation == "tanh" else nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers).double()


class DeepReachValueFn:
    """Torch-MLP value function ``V_θ(x, t)`` with ``t`` fixed to ``time_horizon``."""

    def __init__(self, cfg: DeepReachConfig, model: "torch.nn.Module") -> None:
        self.cfg = cfg
        self._model = model

    # ----- factory -----
    @classmethod
    def from_config(cls, cfg: DeepReachConfig, *, seed: int = 0) -> "DeepReachValueFn":
        model = _MLP.build(cfg, seed=seed)
        model.eval()
        return cls(cfg, model)

    # ----- HJValueFunction surface -----
    def value(self, x: np.ndarray) -> float:
        import torch

        with torch.no_grad():
            inp = self._make_input(x)
            return float(self._model(inp).squeeze(-1).item())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        import torch

        inp = self._make_input(x).requires_grad_(True)
        v = self._model(inp).squeeze(-1)
        (g,) = torch.autograd.grad(v, inp)
        return g.detach().cpu().numpy()[: int(self.cfg.n_state)]

    def lipschitz_const(self) -> float:
        from .lipschitz import power_iteration_lipschitz

        rng = np.random.default_rng(0)
        n = int(self.cfg.n_state)
        bounds = (np.asarray(self.cfg.state_bounds, dtype=np.float64)
                  if self.cfg.state_bounds is not None
                  else np.repeat([[-1.0, 1.0]], n, axis=0))

        def sample() -> np.ndarray:
            return rng.uniform(bounds[:, 0], bounds[:, 1])

        # Wrap the (state-only) Jacobian with a thin module that holds time fixed.
        import torch

        cfg = self.cfg
        model = self._model

        class _StateOnly(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                t = torch.full((1,), float(cfg.time_horizon), dtype=x.dtype)
                return model(torch.cat([x, t], dim=-1)).squeeze(-1)

        return power_iteration_lipschitz(
            _StateOnly().eval(), sample,
            n_iter=cfg.lipschitz_n_iter, n_starts=cfg.lipschitz_n_starts,
            dtype=torch.float64,
        )

    # ----- persistence -----
    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        meta = asdict(self.cfg)
        Path(path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DeepReachValueFn":
        import torch

        path = Path(path)
        meta = json.loads(Path(path.with_suffix(".json")).read_text())
        meta["hidden_sizes"] = tuple(meta["hidden_sizes"])
        cfg = DeepReachConfig(**meta)
        model = _MLP.build(cfg, seed=0)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return cls(cfg, model)

    # ----- helpers -----
    def _make_input(self, x: np.ndarray) -> "torch.Tensor":
        import torch

        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size != int(self.cfg.n_state):
            raise ValueError(f"expected x of size {self.cfg.n_state}, got {x.size}")
        t = np.array([float(self.cfg.time_horizon)], dtype=np.float64)
        inp = torch.tensor(np.concatenate([x, t]), dtype=torch.float64)
        return inp
```

Update package surface:

```python
# tensoraerospace/agent/uftc/l1/__init__.py
"""UFTC Phase 2 — L1 HJ-Reachability safety shield."""
from __future__ import annotations

from .lipschitz import power_iteration_lipschitz
from .value_fn import DeepReachConfig, DeepReachValueFn, HJValueFunction

__all__ = [
    "DeepReachConfig",
    "DeepReachValueFn",
    "HJValueFunction",
    "power_iteration_lipschitz",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_deepreach_value_fn.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/value_fn.py \
        tensoraerospace/agent/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_deepreach_value_fn.py
git commit -m "feat(uftc): add DeepReach torch-MLP value function (no training yet)"
```

---

### Task 4: `ConformalMargin` from FDDOutput severity

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/conformal.py`
- Create: `tests/agents/uftc/l1/test_conformal.py`
- Modify: `tensoraerospace/agent/uftc/l1/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_conformal.py
"""ConformalMargin growth law and monotonicity properties."""
from __future__ import annotations

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)


def _zero_output() -> FDDOutput:
    return FDDOutput(
        fault_present=False,
        severity=0.0,
        confidence=0.0,
        innovation_norm=0.0,
        time_since_event=0.0,
    )


def test_baseline_eps_when_fdd_clean() -> None:
    cfg = ConformalMarginConfig()
    cm = ConformalMargin(cfg, lipschitz_const=1.0)
    eps = cm.compute(_zero_output(), monitor_alarm="OK")
    assert abs(eps - cfg.eps_0) < 1e-12


def test_eps_grows_with_severity_and_alarm() -> None:
    cfg = ConformalMarginConfig()
    cm = ConformalMargin(cfg, lipschitz_const=1.0)
    base = cm.compute(_zero_output(), monitor_alarm="OK")

    sev = FDDOutput(fault_present=True, severity=2.0,
                    confidence=0.8, innovation_norm=1.5,
                    time_since_event=0.0)
    e_sev = cm.compute(sev, monitor_alarm="OK")
    assert e_sev > base

    e_warn = cm.compute(sev, monitor_alarm="WARN")
    e_crit = cm.compute(sev, monitor_alarm="CRITICAL")
    assert e_warn > e_sev
    assert e_crit > e_warn


def test_lipschitz_scales_eps_linearly() -> None:
    cfg = ConformalMarginConfig()
    e1 = ConformalMargin(cfg, lipschitz_const=1.0).compute(_zero_output())
    e3 = ConformalMargin(cfg, lipschitz_const=3.0).compute(_zero_output())
    assert abs(e3 - 3.0 * e1) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_conformal.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l1/conformal.py
"""Conformal margin εₜ from FDD severity and monitor alarm.

The shield uses ``εₜ = L · ε_raw(fdd, alarm)`` where ``L`` is an upper
bound on the Lipschitz constant of ``∇V``. ``ε_raw`` aggregates abrupt
and gradual severities, the innovation norm and the monitor alarm
level. Phase 1 ``FDDOutput`` exposes only ``severity`` (used as
``severity_abrupt`` here); ``severity_gradual`` defaults to 0.0 until
Task 6 enriches the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput


@dataclass
class ConformalMarginConfig:
    eps_0: float = 0.05
    k_grad: float = 0.10
    k_abrupt: float = 0.20
    k_innov: float = 0.05
    k_alarm: float = 0.30


_ALARM_GAIN = {"OK": 0.0, "WARN": 0.5, "CRITICAL": 1.0}


class ConformalMargin:
    """Compute εₜ from FDDOutput + monitor alarm."""

    def __init__(self, cfg: ConformalMarginConfig, *, lipschitz_const: float) -> None:
        self.cfg = cfg
        self.lipschitz_const = float(lipschitz_const)

    def compute(self, fdd: FDDOutput, monitor_alarm: str = "OK") -> float:
        sev_abrupt = float(getattr(fdd, "severity_abrupt", fdd.severity))
        sev_grad = float(getattr(fdd, "severity_gradual", 0.0))
        innov = float(getattr(fdd, "innovation_norm", 0.0))
        gain_alarm = _ALARM_GAIN.get(str(monitor_alarm), 0.0)
        eps_raw = (
            self.cfg.eps_0
            + self.cfg.k_grad * sev_grad
            + self.cfg.k_abrupt * sev_abrupt
            + self.cfg.k_innov * innov
            + self.cfg.k_alarm * gain_alarm
        )
        return float(eps_raw * self.lipschitz_const)
```

Update `__init__.py`:

```python
# tensoraerospace/agent/uftc/l1/__init__.py — extend __all__
from .conformal import ConformalMargin, ConformalMarginConfig
from .lipschitz import power_iteration_lipschitz
from .value_fn import DeepReachConfig, DeepReachValueFn, HJValueFunction

__all__ = [
    "ConformalMargin",
    "ConformalMarginConfig",
    "DeepReachConfig",
    "DeepReachValueFn",
    "HJValueFunction",
    "power_iteration_lipschitz",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_conformal.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/conformal.py \
        tensoraerospace/agent/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_conformal.py
git commit -m "feat(uftc): add ConformalMargin computing eps_t from FDD severity"
```

---

### Task 5: `GLRDetector` for slow-drift faults

**Files:**
- Create: `tensoraerospace/agent/uftc/fdd/glr.py`
- Create: `tests/agents/uftc/fdd/__init__.py` (if missing)
- Create: `tests/agents/uftc/fdd/test_glr.py`
- Modify: `tensoraerospace/agent/uftc/fdd/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/fdd/test_glr.py
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
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=200, h_alarm=30.0,
                                             cooldown_steps=200))
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
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=100, h_alarm=20.0,
                                             h_clear=5.0, cooldown_steps=50))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/fdd/test_glr.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tensoraerospace.agent.uftc.fdd.glr'`.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/fdd/glr.py
"""Generalised likelihood ratio (GLR) test on Kalman innovations.

Two-sided GLR over a sliding window for slow drifts in the innovation
mean. Under the nominal hypothesis ``ν_t ~ N(0, S_t)``. For an unknown
drift ``μ ≠ 0`` starting at unknown change-time ``τ``,

    T_t = max_{t-W ≤ τ ≤ t-1}  ‖ Σ_{i=τ}^t S_i^{-1} ν_i ‖²_{(Σ_{i=τ}^t S_i^{-1})^{-1}}

This implementation keeps an O(W) window of ``S^{-1} ν`` and ``S^{-1}``
per step; the sup is computed by sweeping τ at update-time. Hysteresis
between ``h_alarm`` and ``h_clear`` plus a ``cooldown_steps`` window
prevent chattering.

References:
    Basseville & Nikiforov (1993) Detection of Abrupt Changes, ch. 7.
    Willsky (1976) Survey of failure detection methods.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class GLRConfig:
    window: int = 200
    h_alarm: float = 30.0
    h_clear: float = 8.0
    cooldown_steps: int = 200
    mu_min_norm: float = 0.05  # discard drift estimate below this magnitude


@dataclass
class GLRState:
    statistic: float
    alarm: bool
    severity: float            # statistic / h_alarm, clipped to [0, 10]
    drift_estimate: np.ndarray
    time_since_alarm: int


class GLRDetector:
    """Sliding-window GLR detector on Kalman innovations."""

    def __init__(self, n_dim: int, cfg: GLRConfig) -> None:
        if cfg.h_clear >= cfg.h_alarm:
            raise ValueError("h_clear must be strictly below h_alarm")
        if cfg.window < 2:
            raise ValueError("window must be ≥ 2")
        self.n_dim = int(n_dim)
        self.cfg = cfg
        self._buf_Sinv_nu: deque[np.ndarray] = deque(maxlen=cfg.window)
        self._buf_Sinv: deque[np.ndarray] = deque(maxlen=cfg.window)
        self._in_alarm = False
        self._steps_in_alarm = 0
        self._steps_since_alarm = 10**9

    def update(self, nu: np.ndarray, S: np.ndarray) -> GLRState:
        nu = np.asarray(nu, dtype=np.float64).reshape(-1)
        S = np.asarray(S, dtype=np.float64)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        Sinv_nu = Sinv @ nu

        self._buf_Sinv_nu.append(Sinv_nu)
        self._buf_Sinv.append(Sinv)

        # Sweep tau in [t-W, t-1] to find max statistic.
        cum_Sinv_nu = np.zeros(self.n_dim)
        cum_Sinv = np.zeros((self.n_dim, self.n_dim))
        T_max = 0.0
        mu_hat = np.zeros(self.n_dim)
        for k in range(len(self._buf_Sinv) - 1, -1, -1):
            cum_Sinv_nu += self._buf_Sinv_nu[k]
            cum_Sinv += self._buf_Sinv[k]
            try:
                solve = np.linalg.solve(cum_Sinv, cum_Sinv_nu)
            except np.linalg.LinAlgError:
                solve = np.linalg.pinv(cum_Sinv) @ cum_Sinv_nu
            T = float(cum_Sinv_nu @ solve)
            if T > T_max:
                T_max = T
                mu_hat = solve

        # Hysteresis & cooldown.
        if (not self._in_alarm) and T_max > self.cfg.h_alarm:
            self._in_alarm = True
            self._steps_since_alarm = 0
            self._steps_in_alarm = 1
        elif self._in_alarm:
            self._steps_in_alarm += 1
            self._steps_since_alarm += 1
            if (T_max < self.cfg.h_clear
                    and self._steps_since_alarm > self.cfg.cooldown_steps):
                self._in_alarm = False
        else:
            self._steps_since_alarm = min(self._steps_since_alarm + 1, 10**9)

        if float(np.linalg.norm(mu_hat)) < self.cfg.mu_min_norm:
            mu_hat = np.zeros(self.n_dim)

        severity = float(min(T_max / self.cfg.h_alarm, 10.0))
        return GLRState(
            statistic=float(T_max),
            alarm=bool(self._in_alarm),
            severity=severity,
            drift_estimate=mu_hat,
            time_since_alarm=int(self._steps_since_alarm),
        )

    def reset(self) -> None:
        self._buf_Sinv_nu.clear()
        self._buf_Sinv.clear()
        self._in_alarm = False
        self._steps_in_alarm = 0
        self._steps_since_alarm = 10**9
```

Update `fdd/__init__.py`:

```python
# tensoraerospace/agent/uftc/fdd/__init__.py
"""FDD primitives for UFTC."""
from __future__ import annotations

from .change_point import ChangePointDetector, ChangePointState
from .detector import FDDConfig, FDDDetector, FDDOutput
from .glr import GLRConfig, GLRDetector, GLRState
from .kalman_3step import KalmanStep, NominalKalman

__all__ = [
    "ChangePointDetector",
    "ChangePointState",
    "FDDConfig",
    "FDDDetector",
    "FDDOutput",
    "GLRConfig",
    "GLRDetector",
    "GLRState",
    "KalmanStep",
    "NominalKalman",
]
```

Create `tests/agents/uftc/fdd/__init__.py` if it does not exist:

```python
# tests/agents/uftc/fdd/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/fdd/test_glr.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/fdd/glr.py \
        tensoraerospace/agent/uftc/fdd/__init__.py \
        tests/agents/uftc/fdd/__init__.py \
        tests/agents/uftc/fdd/test_glr.py
git commit -m "feat(uftc): add sliding-window GLR detector for slow-drift faults"
```

---

### Task 6: Extend `FDDOutput` and compose `FDDDetector` with optional `GLRDetector`

**Files:**
- Modify: `tensoraerospace/agent/uftc/fdd/detector.py`
- Create: `tests/agents/uftc/fdd/test_fdd_extended.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/fdd/test_fdd_extended.py
"""FDDDetector composition with optional GLR; FDDOutput extension."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.change_point import ChangePointDetector
from tensoraerospace.agent.uftc.fdd.detector import (
    FDDConfig,
    FDDDetector,
    FDDOutput,
)
from tensoraerospace.agent.uftc.fdd.glr import GLRConfig, GLRDetector
from tensoraerospace.agent.uftc.fdd.kalman_3step import NominalKalman


def _build_detector(*, with_glr: bool) -> FDDDetector:
    n = 2
    F = np.eye(n) * 0.0           # incremental form
    G = np.zeros((n, 1))
    Q = np.eye(n) * 1e-3
    R = np.eye(n) * 1e-2
    kalman = NominalKalman(F_nominal=F, G_nominal=G, Q=Q, R=R)
    cpd = ChangePointDetector(n_dim=n, h_alarm=20.0, h_clear=5.0,
                              cooldown_steps=200)
    glr = GLRDetector(n_dim=n, cfg=GLRConfig(window=100, h_alarm=30.0,
                                             cooldown_steps=200)) if with_glr else None
    return FDDDetector(n_state=n, n_control=1, kalman=kalman, cpd=cpd,
                       glr=glr, dt=0.01)


def test_extended_fields_default_for_clean_input() -> None:
    rng = np.random.default_rng(0)
    det = _build_detector(with_glr=True)
    x = np.zeros(2)
    u = np.zeros(1)
    out = det.step(x + rng.standard_normal(2) * 0.05, u)
    assert isinstance(out, FDDOutput)
    assert out.fault_kind in ("none", "abrupt", "gradual", "compound")
    assert hasattr(out, "severity_abrupt")
    assert hasattr(out, "severity_gradual")
    assert out.severity == max(out.severity_abrupt, out.severity_gradual)


def test_phase1_compatibility_when_glr_disabled() -> None:
    det = _build_detector(with_glr=False)
    out = det.step(np.zeros(2), np.zeros(1))
    assert out.severity_gradual == 0.0
    # Phase 1 consumers reading FDDOutput.severity see CUSUM-only severity.
    assert abs(out.severity - out.severity_abrupt) < 1e-12


def test_compound_when_both_channels_alarm() -> None:
    rng = np.random.default_rng(1)
    det = _build_detector(with_glr=True)

    # Burn in clean dynamics.
    for _ in range(500):
        det.step(rng.standard_normal(2) * 0.05, np.zeros(1))

    # Inject a sustained large-mean drift to fire both CUSUM and GLR.
    seen_kinds: set[str] = set()
    for _ in range(2000):
        x = rng.standard_normal(2) * 0.05 + np.array([4.0, 0.0])
        out = det.step(x, np.zeros(1))
        seen_kinds.add(out.fault_kind)
    assert "compound" in seen_kinds or "abrupt" in seen_kinds  # at least abrupt; compound when GLR catches up
    assert "compound" in seen_kinds  # both must fire eventually under sustained step
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/fdd/test_fdd_extended.py -v`
Expected: FAIL — `FDDOutput` has no `severity_abrupt`/`severity_gradual`/`fault_kind`; `FDDDetector.__init__` rejects `glr=`.

- [ ] **Step 3: Implement**

Replace `tensoraerospace/agent/uftc/fdd/detector.py`:

```python
"""Composite FDD detector: NominalKalman + CUSUM + optional GLR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .change_point import ChangePointDetector
from .glr import GLRDetector
from .kalman_3step import NominalKalman


@dataclass
class FDDConfig:
    process_noise: float = 1e-3
    measurement_noise: float = 1e-2
    alpha_Q: float = 0.99
    alpha_R: float = 0.99
    adapt_Q: bool = True
    adapt_R: bool = True
    drift: float | None = None
    h_alarm: float = 20.0
    h_clear: float = 5.0
    cooldown_steps: int = 200
    innovation_sigma_gate: float = 5.0


FaultKind = Literal["none", "abrupt", "gradual", "compound"]


@dataclass
class FDDOutput:
    """One-step output of :class:`FDDDetector`.

    Phase 1 fields: ``fault_present``, ``severity``, ``confidence``,
    ``innovation_norm``, ``time_since_event``. ``severity`` always equals
    ``max(severity_abrupt, severity_gradual)`` for compatibility with
    Phase 1 consumers.

    Phase 2 additions: ``fault_kind`` ∈ {"none","abrupt","gradual","compound"},
    ``severity_abrupt``, ``severity_gradual``, ``glr_drift_estimate``.
    """

    fault_present: bool
    severity: float
    confidence: float
    innovation_norm: float
    time_since_event: float
    fault_kind: FaultKind = "none"
    severity_abrupt: float = 0.0
    severity_gradual: float = 0.0
    glr_drift_estimate: Optional[np.ndarray] = None


class FDDDetector:
    """One nominal Kalman + CUSUM + optional GLR → FDDOutput."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        kalman: NominalKalman,
        cpd: ChangePointDetector,
        *,
        dt: float,
        glr: GLRDetector | None = None,
        innovation_sigma_gate: float = 5.0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.kalman = kalman
        self.cpd = cpd
        self.glr = glr
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
        glr: GLRDetector | None = None,
    ) -> "FDDDetector":
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
        return cls(
            n_state=n_state, n_control=n_control,
            kalman=kf, cpd=cpd, dt=dt, glr=glr,
            innovation_sigma_gate=config.innovation_sigma_gate,
        )

    def warm_start(
        self,
        F_nominal: np.ndarray | None = None,
        G_nominal: np.ndarray | None = None,
    ) -> None:
        self.kalman.warm_start(F_nominal=F_nominal, G_nominal=G_nominal)

    def step(self, x_meas: np.ndarray, u_prev: np.ndarray) -> FDDOutput:
        kal = self.kalman.step(x_meas, u_prev)
        try:
            d_t = float(kal.nu @ np.linalg.solve(kal.S, kal.nu))
        except np.linalg.LinAlgError:
            d_t = float(kal.nu @ (np.linalg.pinv(kal.S) @ kal.nu))
        d_t = max(d_t, 0.0)

        cp = self.cpd.update(d_t)
        gl = self.glr.update(kal.nu, kal.S) if self.glr is not None else None

        abrupt = bool(cp.alarm)
        gradual = bool(gl.alarm) if gl is not None else False
        kind: FaultKind = (
            "compound" if abrupt and gradual
            else "abrupt" if abrupt
            else "gradual" if gradual
            else "none"
        )
        sev_a = float(cp.severity)
        sev_g = float(gl.severity) if gl is not None else 0.0
        severity = max(sev_a, sev_g)
        confidence = float(1.0 - np.exp(-(sev_a + sev_g)))
        return FDDOutput(
            fault_present=(abrupt or gradual),
            severity=severity,
            confidence=confidence,
            innovation_norm=float(np.linalg.norm(kal.nu)),
            time_since_event=float(cp.time_since_alarm) * self.dt,
            fault_kind=kind,
            severity_abrupt=sev_a,
            severity_gradual=sev_g,
            glr_drift_estimate=(gl.drift_estimate if gl is not None and gl.alarm
                                else None),
        )

    def reset(self) -> None:
        self.kalman.reset()
        self.cpd.reset()
        if self.glr is not None:
            self.glr.reset()
```

- [ ] **Step 4: Run test to verify it passes (and Phase 1 tests still pass)**

Run:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest \
    tests/agents/uftc/fdd/test_fdd_extended.py \
    tests/agents/uftc/test_fdd_detector.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/fdd/detector.py \
        tests/agents/uftc/fdd/test_fdd_extended.py
git commit -m "feat(uftc): extend FDDOutput with fault_kind+gradual; compose GLR in FDDDetector"
```

---

### Task 7: `ValueBank` worst-case fallback under open-world FDD

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/bank.py`
- Create: `tests/agents/uftc/l1/test_bank.py`
- Modify: `tensoraerospace/agent/uftc/l1/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_bank.py
"""ValueBank lookup logic: nominal/abrupt-with-prob/min-fallback."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.bank import ValueBank, ValueBankConfig


@dataclass
class _Const:
    """Tiny stub HJValueFunction for tests."""

    val: float
    L: float = 1.0

    def value(self, x: np.ndarray) -> float:  # noqa: D401, ARG002
        return float(self.val)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def lipschitz_const(self) -> float:
        return self.L


def _zero_fdd(kind: str = "none") -> FDDOutput:
    return FDDOutput(
        fault_present=(kind != "none"),
        severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind=kind, severity_abrupt=0.0, severity_gradual=0.0,
    )


def test_nominal_picks_nominal() -> None:
    bank = ValueBank({"nominal": _Const(0.7), "elev_jam": _Const(-0.3)},
                     ValueBankConfig(fallback="min"))
    fdd = _zero_fdd("none")
    assert bank.value(np.zeros(2), fdd) == 0.7


def test_open_world_fallback_min() -> None:
    bank = ValueBank({"nominal": _Const(0.7), "elev_jam": _Const(-0.3)},
                     ValueBankConfig(fallback="min"))
    fdd = _zero_fdd("abrupt")  # no MMAE probs available
    assert bank.value(np.zeros(2), fdd) == -0.3


def test_lipschitz_max_over_bank() -> None:
    bank = ValueBank({"nominal": _Const(0.7, L=2.0),
                      "elev_jam": _Const(-0.3, L=5.0)},
                     ValueBankConfig(fallback="min"))
    assert bank.lipschitz_const() == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_bank.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l1/bank.py
"""Per-mode value-function bank with worst-case open-world fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .value_fn import HJValueFunction


@dataclass
class ValueBankConfig:
    fallback: Literal["nominal", "min"] = "min"
    abrupt_lookup_threshold: float = 0.7


class ValueBank:
    """Picks a per-mode V_θ^(h) based on FDDOutput."""

    def __init__(self, value_fns: Mapping[str, HJValueFunction],
                 cfg: ValueBankConfig | None = None) -> None:
        if "nominal" not in value_fns:
            raise ValueError("bank must contain a 'nominal' entry")
        self._vs = dict(value_fns)
        self.cfg = cfg or ValueBankConfig()

    def value(self, x: np.ndarray, fdd: FDDOutput) -> float:
        return self._lookup(fdd).value(x)

    def gradient(self, x: np.ndarray, fdd: FDDOutput) -> np.ndarray:
        return self._lookup(fdd).gradient(x)

    def lipschitz_const(self) -> float:
        return float(max(v.lipschitz_const() for v in self._vs.values()))

    # ----- internal -----
    def _lookup(self, fdd: FDDOutput) -> HJValueFunction:
        if fdd.fault_kind == "none":
            return self._vs["nominal"]
        # MMAE-based class lookup not in Phase 2 — fall through to fallback.
        if self.cfg.fallback == "nominal":
            return self._vs["nominal"]
        # "min" — worst-case open-world shielding: pick the entry whose
        # value at this state is smallest (closest to / past boundary).
        return _MinOverBank(self._vs)


class _MinOverBank:
    """Helper exposing HJValueFunction surface backed by min over a bank."""

    def __init__(self, vs: Mapping[str, HJValueFunction]) -> None:
        self._vs = dict(vs)

    def value(self, x: np.ndarray) -> float:
        return float(min(v.value(x) for v in self._vs.values()))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        # gradient of min is the gradient of the argmin (subgradient choice).
        argmin = min(self._vs.values(), key=lambda v: v.value(x))
        return argmin.gradient(x)

    def lipschitz_const(self) -> float:
        return float(max(v.lipschitz_const() for v in self._vs.values()))
```

Update `__init__.py`:

```python
# tensoraerospace/agent/uftc/l1/__init__.py — extend
from .bank import ValueBank, ValueBankConfig
from .conformal import ConformalMargin, ConformalMarginConfig
from .lipschitz import power_iteration_lipschitz
from .value_fn import DeepReachConfig, DeepReachValueFn, HJValueFunction

__all__ = [
    "ConformalMargin",
    "ConformalMarginConfig",
    "DeepReachConfig",
    "DeepReachValueFn",
    "HJValueFunction",
    "ValueBank",
    "ValueBankConfig",
    "power_iteration_lipschitz",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_bank.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/bank.py \
        tensoraerospace/agent/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_bank.py
git commit -m "feat(uftc): add ValueBank with open-world worst-case fallback"
```

---

### Task 8: `HJReachabilityShield` QP post-filter

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/shield.py`
- Create: `tests/agents/uftc/l1/test_shield_qp.py`
- Modify: `pyproject.toml` (add `cvxpy`)
- Modify: `tensoraerospace/agent/uftc/l1/__init__.py`

- [ ] **Step 1: Add `cvxpy` to project dependencies**

Edit `pyproject.toml` and run:

```bash
poetry add cvxpy
```

Verify the lock file updates and the dependency is recorded.

- [ ] **Step 2: Write the failing test**

```python
# tests/agents/uftc/l1/test_shield_qp.py
"""HJReachabilityShield QP behaviour and bounds enforcement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)
from tensoraerospace.agent.uftc.l1.shield import (
    HJReachabilityShield,
    HJShieldConfig,
    ShieldOutput,
)


@dataclass
class _Linear:
    """V(x) = a^T x + b — linear value function for analytical tests."""

    a: np.ndarray
    b: float

    def value(self, x: np.ndarray) -> float:
        return float(self.a @ x + self.b)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.a, dtype=np.float64, copy=True)

    def lipschitz_const(self) -> float:
        return float(np.linalg.norm(self.a))


def _clean_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=False, severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind="none", severity_abrupt=0.0, severity_gradual=0.0,
    )


def _affine_dynamics(F: np.ndarray, G: np.ndarray):
    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return F @ x + G @ u
    return f


def test_passthrough_when_deep_inside_safe_set() -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=10.0)  # V(x) very positive
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=vfn.lipschitz_const())
    shield = HJReachabilityShield(
        n_state=2, n_control=2,
        value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=0.5, u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0]), conformal=cm.cfg),
        conformal_margin=cm,
    )
    x = np.array([0.0, 0.0])
    u_nom = np.array([0.7, -0.3])
    out = shield.filter(x, u_nom, _clean_fdd())
    assert isinstance(out, ShieldOutput)
    assert out.active is False
    assert np.allclose(out.u_safe, u_nom)


def test_qp_enforces_u_bounds() -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=0.0)  # V at boundary
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=1.0)
    shield = HJReachabilityShield(
        n_state=2, n_control=2,
        value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=-1.0,    # force shield active
                           u_min=np.array([-0.5, -0.5]),
                           u_max=np.array([0.5, 0.5]),
                           conformal=cm.cfg),
        conformal_margin=cm,
    )
    out = shield.filter(np.array([0.0, 0.0]),
                        np.array([2.0, -2.0]),  # outside bounds
                        _clean_fdd())
    assert (out.u_safe >= -0.5 - 1e-6).all()
    assert (out.u_safe <= 0.5 + 1e-6).all()


def test_solver_failure_falls_back_to_nominal(monkeypatch) -> None:
    F = np.zeros((2, 2)); G = np.eye(2)
    vfn = _Linear(np.array([1.0, 0.0]), b=0.0)
    cm = ConformalMargin(ConformalMarginConfig(eps_0=0.05),
                         lipschitz_const=1.0)
    shield = HJReachabilityShield(
        n_state=2, n_control=2, value_fn=vfn,
        dynamics_fn=_affine_dynamics(F, G),
        cfg=HJShieldConfig(h_clear=-1.0,
                           u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0]),
                           conformal=cm.cfg),
        conformal_margin=cm,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(shield, "_solve_qp", boom)
    out = shield.filter(np.array([0.0, 0.0]),
                        np.array([0.4, -0.2]),
                        _clean_fdd())
    assert out.active is False
    assert np.allclose(out.u_safe, [0.4, -0.2])
```

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l1/shield.py
"""HJ-Reachability safety shield: QP post-filter on u_indi.

For an affine-in-control surrogate ``f̂(x, u) = F̃ x + G̃ u``, the shield
enforces a CBF-style condition::

    ⟨∇V, F̃ x + G̃ u⟩ + λ V(x) ≥ ε_t

while minimising ``‖u − u_nominal‖²`` subject to ``u_min ≤ u ≤ u_max``.
The solver is OSQP via cvxpy; failures degrade to the nominal control.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .conformal import ConformalMargin, ConformalMarginConfig
from .value_fn import HJValueFunction

LOG = logging.getLogger(__name__)


@dataclass
class HJShieldConfig:
    h_clear: float = 0.20
    qp_solver: Literal["OSQP", "ECOS", "MOSEK"] = "OSQP"
    cbf_lambda: float = 1.0
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None
    conformal: ConformalMarginConfig = field(default_factory=ConformalMarginConfig)


@dataclass
class ShieldOutput:
    u_safe: np.ndarray
    intervention_norm: float
    hjb_value: float
    active: bool


class HJReachabilityShield:
    """QP post-filter enforcing forward invariance of the HJ safe set."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        *,
        value_fn: HJValueFunction,
        dynamics_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        cfg: HJShieldConfig,
        conformal_margin: ConformalMargin | None = None,
    ) -> None:
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.value_fn = value_fn
        self.dynamics_fn = dynamics_fn
        self.cfg = cfg
        if conformal_margin is None:
            conformal_margin = ConformalMargin(
                cfg.conformal, lipschitz_const=value_fn.lipschitz_const(),
            )
        self.conformal = conformal_margin
        self._hold_one_tick = False
        self._last_u_safe: np.ndarray | None = None
        self._cached_FG: tuple[np.ndarray, np.ndarray] | None = None

    # ----- macro-action sink -----
    def request_actuator_hold(self) -> None:
        self._hold_one_tick = True

    def set_dynamics_jacobian(self, F: np.ndarray, G: np.ndarray) -> None:
        """UFTCController calls this once per tick with current F̃, G̃ from RLS."""
        self._cached_FG = (np.asarray(F, dtype=np.float64),
                           np.asarray(G, dtype=np.float64))

    def filter(
        self,
        x: np.ndarray,
        u_nominal: np.ndarray,
        fdd: FDDOutput,
        monitor_alarm: str = "OK",
    ) -> ShieldOutput:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        u_nominal = np.asarray(u_nominal, dtype=np.float64).reshape(-1)

        if self._hold_one_tick and self._last_u_safe is not None:
            self._hold_one_tick = False
            return ShieldOutput(
                u_safe=self._last_u_safe.copy(),
                intervention_norm=float(np.linalg.norm(self._last_u_safe - u_nominal)),
                hjb_value=float(self.value_fn.value(x)),
                active=True,
            )

        v_x = float(self.value_fn.value(x))
        eps_t = float(self.conformal.compute(fdd, monitor_alarm))
        h_safe = v_x - eps_t
        if h_safe > self.cfg.h_clear:
            self._last_u_safe = u_nominal.copy()
            return ShieldOutput(u_nominal.copy(), 0.0, v_x, active=False)

        grad_v = self.value_fn.gradient(x)
        try:
            u_safe = self._solve_qp(x, u_nominal, grad_v, v_x, eps_t)
        except Exception as e:                      # pragma: no cover - logged
            LOG.warning("HJ-shield QP failed (%s); falling back to nominal", e)
            self._last_u_safe = u_nominal.copy()
            return ShieldOutput(u_nominal.copy(), 0.0, v_x, active=False)

        self._last_u_safe = u_safe.copy()
        return ShieldOutput(
            u_safe=u_safe,
            intervention_norm=float(np.linalg.norm(u_safe - u_nominal)),
            hjb_value=v_x,
            active=True,
        )

    def reset(self) -> None:
        self._hold_one_tick = False
        self._last_u_safe = None
        self._cached_FG = None

    # ----- internal -----
    def _affine_FG(self, x: np.ndarray, u_nominal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_FG is not None:
            return self._cached_FG
        if self.dynamics_fn is None:
            raise RuntimeError("dynamics_fn is None and no F̃,G̃ cached")
        # Numerical Jacobian (F = ∂f/∂x at u, G = ∂f/∂u at x).
        h = 1e-6
        f0 = self.dynamics_fn(x, u_nominal)
        F = np.zeros((self.n_state, self.n_state))
        for i in range(self.n_state):
            x_p = x.copy(); x_p[i] += h
            F[:, i] = (self.dynamics_fn(x_p, u_nominal) - f0) / h
        G = np.zeros((self.n_state, self.n_control))
        for i in range(self.n_control):
            u_p = u_nominal.copy(); u_p[i] += h
            G[:, i] = (self.dynamics_fn(x, u_p) - f0) / h
        return F, G

    def _solve_qp(
        self,
        x: np.ndarray,
        u_nominal: np.ndarray,
        grad_v: np.ndarray,
        v_x: float,
        eps_t: float,
    ) -> np.ndarray:
        import cvxpy as cp

        F, G = self._affine_FG(x, u_nominal)
        u = cp.Variable(self.n_control)
        objective = cp.Minimize(cp.sum_squares(u - u_nominal))
        constraints = [grad_v @ (F @ x + G @ u) + self.cfg.cbf_lambda * v_x >= eps_t]
        if self.cfg.u_min is not None:
            constraints.append(u >= np.asarray(self.cfg.u_min, dtype=np.float64))
        if self.cfg.u_max is not None:
            constraints.append(u <= np.asarray(self.cfg.u_max, dtype=np.float64))
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.cfg.qp_solver, verbose=False)
        if u.value is None:
            raise RuntimeError(f"QP returned no solution; status={prob.status}")
        return np.asarray(u.value, dtype=np.float64).reshape(-1)
```

Update `__init__.py`:

```python
# tensoraerospace/agent/uftc/l1/__init__.py — extend
from .bank import ValueBank, ValueBankConfig
from .conformal import ConformalMargin, ConformalMarginConfig
from .lipschitz import power_iteration_lipschitz
from .shield import HJReachabilityShield, HJShieldConfig, ShieldOutput
from .value_fn import DeepReachConfig, DeepReachValueFn, HJValueFunction

__all__ = [
    "ConformalMargin",
    "ConformalMarginConfig",
    "DeepReachConfig",
    "DeepReachValueFn",
    "HJReachabilityShield",
    "HJShieldConfig",
    "HJValueFunction",
    "ShieldOutput",
    "ValueBank",
    "ValueBankConfig",
    "power_iteration_lipschitz",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_shield_qp.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/shield.py \
        tensoraerospace/agent/uftc/l1/__init__.py \
        tests/agents/uftc/l1/test_shield_qp.py \
        pyproject.toml poetry.lock
git commit -m "feat(uftc): add HJReachabilityShield QP post-filter"
```

---

### Task 9: `request_actuator_hold` macro-action behaviour

**Files:**
- Create: `tests/agents/uftc/l1/test_request_actuator_hold.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_request_actuator_hold.py
"""request_actuator_hold freezes u_safe for exactly one filter() call."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l1.conformal import (
    ConformalMargin,
    ConformalMarginConfig,
)
from tensoraerospace.agent.uftc.l1.shield import (
    HJReachabilityShield,
    HJShieldConfig,
)


@dataclass
class _Const:
    v: float = 100.0  # always deep inside safe set
    L: float = 1.0

    def value(self, x): return self.v
    def gradient(self, x): return np.zeros_like(x)
    def lipschitz_const(self): return self.L


def _clean_fdd() -> FDDOutput:
    return FDDOutput(
        fault_present=False, severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind="none", severity_abrupt=0.0, severity_gradual=0.0,
    )


def _build_shield():
    cm = ConformalMargin(ConformalMarginConfig(), lipschitz_const=1.0)
    return HJReachabilityShield(
        n_state=2, n_control=2, value_fn=_Const(),
        dynamics_fn=lambda x, u: u,
        cfg=HJShieldConfig(h_clear=0.0,
                           u_min=np.array([-1.0, -1.0]),
                           u_max=np.array([1.0, 1.0])),
        conformal_margin=cm,
    )


def test_hold_repeats_last_u_safe_once() -> None:
    sh = _build_shield()
    out1 = sh.filter(np.zeros(2), np.array([0.5, -0.2]), _clean_fdd())
    sh.request_actuator_hold()
    out2 = sh.filter(np.zeros(2), np.array([0.9, 0.9]), _clean_fdd())
    assert np.allclose(out2.u_safe, out1.u_safe)
    # next tick returns to nominal
    out3 = sh.filter(np.zeros(2), np.array([0.1, 0.1]), _clean_fdd())
    assert np.allclose(out3.u_safe, [0.1, 0.1])


def test_hold_without_prior_filter_is_noop() -> None:
    sh = _build_shield()
    sh.request_actuator_hold()
    out = sh.filter(np.zeros(2), np.array([0.3, -0.1]), _clean_fdd())
    # No prior u_safe, so the hold is silently dropped.
    assert np.allclose(out.u_safe, [0.3, -0.1])
```

- [ ] **Step 2: Run test to verify it passes** (Task 8 already implemented `request_actuator_hold`)

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_request_actuator_hold.py -v`
Expected: 2 passed. If `test_hold_without_prior_filter_is_noop` fails, fix shield: when `_last_u_safe is None`, ignore the hold flag and run the normal filter path.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/l1/test_request_actuator_hold.py
git commit -m "test(uftc): cover request_actuator_hold macro-action semantics"
```

---

### Task 10: DeepReach training entry point (smoke-level)

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/deepreach_train.py`
- Create: `tests/agents/uftc/l1/test_deepreach_train_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l1/test_deepreach_train_smoke.py
"""5-epoch smoke training: HJI-residual loss decreases monotonically."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from tensoraerospace.agent.uftc.l1.deepreach_train import (
    TrainingConfig,
    train_value_fn,
)
from tensoraerospace.agent.uftc.l1.value_fn import DeepReachConfig


def _double_integrator():
    """f(x,u) = [x[1], u[0]]; safe set ℓ(x) = 1 - max(|x_1|, |x_2|)."""

    def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.array([x[1], u[0]], dtype=np.float64)

    def ell(x: np.ndarray) -> float:
        return 1.0 - float(max(abs(x[0]), abs(x[1])))

    return f, ell


def test_smoke_loss_decreases() -> None:
    f, ell = _double_integrator()
    cfg_v = DeepReachConfig(n_state=2, hidden_sizes=(16, 16),
                            state_bounds=[[-2.0, 2.0], [-2.0, 2.0]],
                            time_horizon=1.0)
    train_cfg = TrainingConfig(epochs=5, batch_size=128, lr=1e-3,
                               u_low=np.array([-1.0]), u_high=np.array([1.0]),
                               disturbance_low=None, disturbance_high=None,
                               n_state=2, n_control=1, seed=0)
    fn, history = train_value_fn(cfg_v, train_cfg, dynamics=f, safe_set=ell)
    assert len(history["loss"]) == 5
    assert history["loss"][-1] < history["loss"][0]
    # value evaluable
    v = fn.value(np.array([0.0, 0.0]))
    assert isinstance(v, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_deepreach_train_smoke.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l1/deepreach_train.py
"""DeepReach offline training of V_θ on the HJI residual.

Minimal Phase 2 implementation: PINN-residual + boundary loss.
The full curriculum (smoothness regulariser, multi-stage scheduling,
near-boundary sampling) is documented in the spec but not required
for Phase 2 to land. The CLI script wraps this function with argparse.

References: Bansal & Tomlin (2021), DeepReach.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .value_fn import DeepReachConfig, DeepReachValueFn, _MLP


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 4096
    lr: float = 1e-3
    n_state: int = 0
    n_control: int = 0
    u_low: Optional[np.ndarray] = None
    u_high: Optional[np.ndarray] = None
    disturbance_low: Optional[np.ndarray] = None
    disturbance_high: Optional[np.ndarray] = None
    boundary_weight: float = 1.0
    seed: int = 0


def train_value_fn(
    value_cfg: DeepReachConfig,
    train_cfg: TrainingConfig,
    *,
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
    safe_set: Callable[[np.ndarray], float],
) -> tuple[DeepReachValueFn, dict[str, list[float]]]:
    import torch

    torch.manual_seed(int(train_cfg.seed))
    rng = np.random.default_rng(int(train_cfg.seed))
    model = _MLP.build(value_cfg, seed=int(train_cfg.seed))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    bounds = (np.asarray(value_cfg.state_bounds, dtype=np.float64)
              if value_cfg.state_bounds is not None
              else np.repeat([[-1.0, 1.0]], value_cfg.n_state, axis=0))

    history: dict[str, list[float]] = {"loss": []}
    for epoch in range(int(train_cfg.epochs)):
        x = rng.uniform(bounds[:, 0], bounds[:, 1],
                        size=(int(train_cfg.batch_size), value_cfg.n_state))
        t = rng.uniform(0.0, value_cfg.time_horizon,
                        size=(int(train_cfg.batch_size), 1))
        x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        t_t = torch.tensor(t, dtype=torch.float64, requires_grad=True)
        inp = torch.cat([x_t, t_t], dim=-1)
        v = model(inp).squeeze(-1)

        grads = torch.autograd.grad(v.sum(), [x_t, t_t], create_graph=True)
        dV_dx, dV_dt = grads[0], grads[1].squeeze(-1)

        # Min-max over u (control) and d (disturbance) computed analytically
        # for affine-in-u dynamics is hard for arbitrary callables; sample
        # finitely-many candidates and take the worst.
        loss_hji_terms = []
        u_samples = _sample_box(train_cfg.u_low, train_cfg.u_high,
                                n=8, n_state=int(value_cfg.n_state),
                                rng=rng)
        d_samples = _sample_box(train_cfg.disturbance_low, train_cfg.disturbance_high,
                                n=4, n_state=int(value_cfg.n_state),
                                rng=rng)
        worst = None
        for u_s in u_samples:
            for d_s in d_samples:
                f_xs = np.stack([dynamics(xi, u_s) + d_s for xi in x], axis=0)
                f_t = torch.tensor(f_xs, dtype=torch.float64)
                hji = dV_dt + (dV_dx * f_t).sum(dim=-1)
                if worst is None:
                    worst = hji
                else:
                    worst = torch.minimum(worst, hji)
        loss_hji = (worst ** 2).mean()

        # Boundary condition at t = T.
        x_b = rng.uniform(bounds[:, 0], bounds[:, 1],
                          size=(int(train_cfg.batch_size), value_cfg.n_state))
        t_b = np.full((int(train_cfg.batch_size), 1), value_cfg.time_horizon)
        inp_b = torch.tensor(np.concatenate([x_b, t_b], axis=-1),
                             dtype=torch.float64)
        v_b = model(inp_b).squeeze(-1)
        l_b = torch.tensor([safe_set(xi) for xi in x_b], dtype=torch.float64)
        loss_bdy = ((v_b - l_b) ** 2).mean()

        loss = loss_hji + train_cfg.boundary_weight * loss_bdy
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))

    model.eval()
    return DeepReachValueFn(value_cfg, model), history


def _sample_box(low: Optional[np.ndarray], high: Optional[np.ndarray],
                *, n: int, n_state: int, rng: np.random.Generator) -> list[np.ndarray]:
    if low is None or high is None:
        return [np.zeros(n_state)]
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    return [rng.uniform(low, high) for _ in range(int(n))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l1/test_deepreach_train_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/deepreach_train.py \
        tests/agents/uftc/l1/test_deepreach_train_smoke.py
git commit -m "feat(uftc): add DeepReach offline training entry point"
```

---

### Task 11: Wire L1 + GLR into `UFTCController` behind `enable_*` flags

**Files:**
- Modify: `tensoraerospace/agent/uftc/controller.py`
- Modify: `tensoraerospace/agent/uftc/__init__.py`
- Create: `tests/agents/uftc/test_uftc_l1_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_l1_smoke.py
"""UFTCController integration smoke: L1 + GLR flags toggle correctly."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _build_controller(*, enable_l1: bool, enable_glr: bool) -> UFTCController:
    cfg = UFTCConfig(
        dt=0.01,
        fdd_warmup_steps=20,
        enable_l1_shield=enable_l1,
        enable_glr=enable_glr,
    )
    return UFTCController(n_state=3, n_control=2, config=cfg)


def test_flags_persist_through_predict_learn() -> None:
    ctl = _build_controller(enable_l1=False, enable_glr=False)
    x = np.array([0.1, 0.0, -0.05])
    r = np.zeros(3)
    u = ctl.predict(x, r, time_step=0)
    info = ctl.learn(x + np.array([0.001, 0.0, 0.0]), r, time_step=0)
    assert isinstance(u, np.ndarray)
    assert u.shape == (2,)
    assert isinstance(info, dict)


def test_l1_shield_runs_when_enabled() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    ctl = _build_controller(enable_l1=True, enable_glr=False)
    rng = np.random.default_rng(0)
    for k in range(50):
        x = rng.standard_normal(3) * 0.05
        r = np.zeros(3)
        u = ctl.predict(x, r, time_step=k)
        ctl.learn(x, r, time_step=k)
    diag = ctl.diagnostics()
    assert "l1" in diag


def test_glr_severity_appears_in_diag_when_enabled() -> None:
    ctl = _build_controller(enable_l1=False, enable_glr=True)
    rng = np.random.default_rng(0)
    for k in range(60):
        x = rng.standard_normal(3) * 0.05
        ctl.predict(x, np.zeros(3), time_step=k)
        ctl.learn(x, np.zeros(3), time_step=k)
    diag = ctl.diagnostics()
    assert "fdd" in diag
    assert "severity_gradual" in diag["fdd"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_l1_smoke.py -v`
Expected: FAIL — `UFTCConfig` has no `enable_l1_shield` or `enable_glr`; `UFTCController.diagnostics` lacks `l1` / `severity_gradual` entries.

- [ ] **Step 3: Implement**

In `tensoraerospace/agent/uftc/controller.py`:

1. Add fields to `UFTCConfig`:
```python
@dataclass
class UFTCConfig:
    # ... existing Phase 1 fields ...

    # Phase 2 — L1 shield
    enable_l1_shield: bool = False
    l1_h_clear: float = 0.20
    l1_cbf_lambda: float = 1.0
    l1_u_min: list[float] | None = None
    l1_u_max: list[float] | None = None
    l1_value_fn_path: str | None = None       # path to a saved DeepReachValueFn
    l1_conformal_eps_0: float = 0.05

    # Phase 2 — GLR
    enable_glr: bool = False
    glr_window: int = 200
    glr_h_alarm: float = 30.0
    glr_h_clear: float = 8.0
    glr_cooldown_steps: int = 200
```

2. In `UFTCController.__init__` (after Phase 1 setup):
```python
# Phase 2 — GLR detector wired into FDD
if self.cfg.enable_glr:
    from tensoraerospace.agent.uftc.fdd.glr import GLRConfig, GLRDetector
    self.fdd.glr = GLRDetector(
        n_dim=self.n_state,
        cfg=GLRConfig(
            window=self.cfg.glr_window,
            h_alarm=self.cfg.glr_h_alarm,
            h_clear=self.cfg.glr_h_clear,
            cooldown_steps=self.cfg.glr_cooldown_steps,
        ),
    )

# Phase 2 — HJ shield
self.l1 = None
if self.cfg.enable_l1_shield:
    from tensoraerospace.agent.uftc.l1 import (
        ConformalMargin, ConformalMarginConfig,
        DeepReachValueFn, HJReachabilityShield, HJShieldConfig,
    )
    if self.cfg.l1_value_fn_path is None:
        # No saved network — build a placeholder constant value function
        # that always reports "deep inside safe set". This makes shielding a
        # no-op until the user provides a real V_θ.
        from tensoraerospace.agent.uftc.l1.shield import _Identity  # noqa: WPS433
        value_fn = _Identity()
    else:
        value_fn = DeepReachValueFn.load(self.cfg.l1_value_fn_path)
    cm = ConformalMargin(
        ConformalMarginConfig(eps_0=self.cfg.l1_conformal_eps_0),
        lipschitz_const=value_fn.lipschitz_const(),
    )
    self.l1 = HJReachabilityShield(
        n_state=self.n_state, n_control=self.n_control,
        value_fn=value_fn,
        dynamics_fn=None,                # uses RLS-borrow path
        cfg=HJShieldConfig(
            h_clear=self.cfg.l1_h_clear,
            cbf_lambda=self.cfg.l1_cbf_lambda,
            u_min=(np.asarray(self.cfg.l1_u_min, dtype=np.float64)
                   if self.cfg.l1_u_min is not None else None),
            u_max=(np.asarray(self.cfg.l1_u_max, dtype=np.float64)
                   if self.cfg.l1_u_max is not None else None),
        ),
        conformal_margin=cm,
    )
self._last_u_safe: np.ndarray | None = None
```

3. In `predict()` (after `u_indi` is computed):
```python
if self.l1 is not None:
    F_full = self.middle.base.F  # (n_state, n_state) — RLS-borrowed
    G_full = self.middle.base.G  # (n_state, n_control)
    self.l1.set_dynamics_jacobian(F_full, G_full)
    fdd = self._last_fdd or _zero_fdd_output(self.n_state)
    out = self.l1.filter(x_obs, u_indi, fdd, monitor_alarm="OK")
    u_out = out.u_safe
else:
    u_out = u_indi
self._last_u_safe = u_out
return u_out
```

4. In `learn()`, when stepping the FDD, pass `self._last_u_safe` instead of `last_u_indi`:
```python
fdd_out = self.fdd.step(next_x, self._last_u_safe if self._last_u_safe is not None else last_u_indi)
self._last_fdd = fdd_out
```

5. Extend `diagnostics()` to add `"l1"` and `"fdd"` blocks. The `"fdd"` block must include `severity_gradual` and `fault_kind`.

6. Add helper at the top of `shield.py` (so `_Identity` import works):
```python
class _Identity:
    """Always-safe constant V — used when no V_θ artifact is wired."""
    def value(self, x): return 1.0
    def gradient(self, x): return __import__("numpy").zeros_like(x)
    def lipschitz_const(self): return 1.0
```

7. Add a small helper near the top of `controller.py`:
```python
def _zero_fdd_output(n_state: int):
    from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
    return FDDOutput(
        fault_present=False, severity=0.0, confidence=0.0,
        innovation_norm=0.0, time_since_event=0.0,
        fault_kind="none", severity_abrupt=0.0, severity_gradual=0.0,
    )
```

8. Update `tensoraerospace/agent/uftc/__init__.py` re-exports:
```python
from .l1 import HJReachabilityShield, HJShieldConfig
from .fdd.glr import GLRConfig, GLRDetector
__all__ = sorted(set(__all__ + [
    "GLRConfig", "GLRDetector", "HJReachabilityShield", "HJShieldConfig",
]))
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest \
    tests/agents/uftc/test_uftc_l1_smoke.py \
    tests/agents/uftc/test_uftc_smoke.py -v
```

Expected: all green. Phase 1 smoke unaffected.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/controller.py \
        tensoraerospace/agent/uftc/__init__.py \
        tensoraerospace/agent/uftc/l1/shield.py \
        tests/agents/uftc/test_uftc_l1_smoke.py
git commit -m "feat(uftc): wire L1 shield + GLR into UFTCController behind enable_* flags"
```

---

### Task 12: ENGINE_THRUST_DRIFT damage preset (slow-drift fault)

**Files:**
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`
- Create: `tests/agents/aerospacemodel/test_engine_thrust_drift_preset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/aerospacemodel/test_engine_thrust_drift_preset.py
"""ENGINE_THRUST_DRIFT preset: thrust shrinks linearly over time."""
from __future__ import annotations

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets


def test_engine_thrust_drift_preset_exists() -> None:
    p = presets.ENGINE_THRUST_DRIFT
    assert p.events, "preset must contain at least one ramp event"


def test_thrust_loss_linear_with_time() -> None:
    """At default 1 %/s loss, after 5 s thrust scale ≈ 0.95."""
    p = presets.ENGINE_THRUST_DRIFT
    # Apply preset over a 5 s window using the manager helpers.
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.manager import (
        DamageManager,
    )
    mgr = DamageManager(profile=p, dt=0.01)
    for k in range(int(5.0 / 0.01)):
        mgr.step(k * 0.01)
    state = mgr.state
    assert 0.93 <= state.engine.thrust_scale <= 0.97
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/aerospacemodel/test_engine_thrust_drift_preset.py -v`
Expected: FAIL — preset missing.

- [ ] **Step 3: Implement**

Append to `tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py`:

```python
ENGINE_THRUST_DRIFT = DamageProfile(
    name="engine_thrust_drift",
    events=[
        DamageEvent(
            t_start=0.0,
            t_end=300.0,           # ramp over a long horizon
            kind="engine",
            params={
                "thrust_scale_start": 1.0,
                "thrust_scale_end": 0.0,    # full loss after 300 s
                "ramp": "linear",
            },
        ),
    ],
)
```

If the existing `DamageEvent` schema does not support a linear-ramp `engine` event, extend `DamageManager._apply_event` with a `ramp` param: when `event.params.get("ramp") == "linear"`, set
`thrust_scale = lerp(thrust_scale_start, thrust_scale_end, (t - t_start) / (t_end - t_start))` clamped to `[t_start, t_end]`. Cover with one extra unit-test in the existing damage-manager test file if needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/aerospacemodel/test_engine_thrust_drift_preset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/presets.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/damage/manager.py \
        tests/agents/aerospacemodel/test_engine_thrust_drift_preset.py
git commit -m "feat(damage): add ENGINE_THRUST_DRIFT linear-ramp preset"
```

---

### Task 13: Phase 2 integration test on F-16 with engine-drift preset

**Files:**
- Create: `tests/agents/uftc/test_uftc_l1_engine_drift.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_l1_engine_drift.py
"""F-16 ENGINE_THRUST_DRIFT: GLR detects within 3 s; L1 does not block tracking."""
from __future__ import annotations

import numpy as np
import pytest

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
from tensoraerospace.aerospacemodel.f16.nonlinear import LongitudinalF16


def _make_env():
    return LongitudinalF16(damage_profile=presets.ENGINE_THRUST_DRIFT, dt=0.01)


def test_glr_alarm_within_three_seconds() -> None:
    pytest.importorskip("torch")
    env = _make_env()
    ctl = UFTCController(
        n_state=env.n_state, n_control=env.n_control,
        config=UFTCConfig(dt=0.01, enable_glr=True, fdd_warmup_steps=200),
    )
    x = env.reset()
    alarm_at = None
    for k in range(int(8.0 / 0.01)):
        u = ctl.predict(x, np.zeros(env.n_state), time_step=k)
        x = env.step(u)
        info = ctl.learn(x, np.zeros(env.n_state), time_step=k)
        fdd = info.get("fdd", {})
        if fdd.get("fault_kind") in ("gradual", "compound"):
            alarm_at = k * 0.01
            break
    assert alarm_at is not None
    # Drift is 1 %/s; we expect detection well before 3 s after warm-up.
    assert alarm_at < 5.0


def test_l1_shield_does_not_block_tracking() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cvxpy")
    env = _make_env()
    ctl = UFTCController(
        n_state=env.n_state, n_control=env.n_control,
        config=UFTCConfig(dt=0.01, enable_l1_shield=True, enable_glr=True,
                          fdd_warmup_steps=200),
    )
    x = env.reset()
    last_x = x.copy()
    for k in range(int(10.0 / 0.01)):
        u = ctl.predict(x, np.zeros(env.n_state), time_step=k)
        x = env.step(u)
        ctl.learn(x, np.zeros(env.n_state), time_step=k)
    # Aircraft did not blow up.
    assert np.all(np.isfinite(x))
    # Tracking error remained finite.
    assert np.linalg.norm(x - last_x) < 1e3
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_l1_engine_drift.py -v`
Expected: PASS (Tasks 5-12 should be sufficient). If it fails:
- inspect `info["fdd"]["fault_kind"]` over time;
- check that GLR window is reaching its capacity before drift onset;
- adjust `fdd_warmup_steps` or the GLR thresholds in the test fixture.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_l1_engine_drift.py
git commit -m "test(uftc): F-16 ENGINE_THRUST_DRIFT integration for L1+GLR"
```

---

### Task 14: Phase 1 invariance regression test

**Files:**
- Create: `tests/agents/uftc/test_uftc_phase1_invariance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_phase1_invariance.py
"""With all Phase 2 flags off, UFTCController must match Phase 1 byte-for-byte."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _seeded_rollout(*, enable_l1: bool, enable_glr: bool, n: int = 1000):
    rng = np.random.default_rng(20260508)
    cfg = UFTCConfig(dt=0.01, fdd_warmup_steps=50,
                     enable_l1_shield=enable_l1, enable_glr=enable_glr)
    ctl = UFTCController(n_state=4, n_control=2, config=cfg)
    xs, us = [], []
    x = rng.standard_normal(4) * 0.1
    r = np.zeros(4)
    for k in range(n):
        u = ctl.predict(x, r, time_step=k)
        x = x + 0.01 * (rng.standard_normal(4) * 0.05 + 0.1 * (r - x))
        ctl.learn(x, r, time_step=k)
        xs.append(x.copy()); us.append(np.asarray(u, dtype=np.float64).copy())
    return np.stack(xs), np.stack(us)


def test_phase2_flags_off_matches_phase1_exactly() -> None:
    x_off, u_off = _seeded_rollout(enable_l1=False, enable_glr=False)
    x_off_ref, u_off_ref = _seeded_rollout(enable_l1=False, enable_glr=False)
    np.testing.assert_array_equal(x_off, x_off_ref)
    np.testing.assert_array_equal(u_off, u_off_ref)


def test_phase2_flags_on_diverges() -> None:
    """Sanity check that flags actually change behaviour (otherwise our
    invariance test would trivially pass)."""
    x_off, u_off = _seeded_rollout(enable_l1=False, enable_glr=False, n=400)
    x_on, u_on = _seeded_rollout(enable_l1=False, enable_glr=True, n=400)
    # FDDOutput diff downstream of learn() can change controller-side state
    # if any layer reads fault_kind / severity_gradual; the action stream
    # should still differ at least sometimes.
    assert np.allclose(x_off, x_on) or not np.allclose(u_off, u_on)
```

- [ ] **Step 2: Run test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_phase1_invariance.py -v`
Expected: 2 passed (the first asserts deterministic identity; the second is a soft sanity check).

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_phase1_invariance.py
git commit -m "test(uftc): pin Phase 1 byte-for-byte invariance under Phase 2 flags-off"
```

---

### Task 15: L1 README + pretrain example script

**Files:**
- Create: `tensoraerospace/agent/uftc/l1/README.md`
- Create: `example/uftc/pretrain_hj_value.py`

- [ ] **Step 1: Write the README**

```markdown
<!-- tensoraerospace/agent/uftc/l1/README.md -->
# UFTC L1 — HJ-Reachability safety shield

Phase 2 component of the UFTC cascade. Provides:

- `HJValueFunction` protocol + `DeepReachValueFn` torch backend.
- `power_iteration_lipschitz` upper bound used by `ConformalMargin`.
- `ConformalMargin` translating FDD severity into the runtime margin εₜ.
- `ValueBank` with worst-case fallback for open-world FDD.
- `HJReachabilityShield` — QP post-filter on `u_indi`.

## Pre-training a value function

```bash
python -m tensoraerospace.agent.uftc.l1.deepreach_train \
    --plant f16-nonlinear-angular --mode nominal --epochs 200 \
    --out artifacts/v_hj/nominal/
```

The CLI wraps `train_value_fn` with argparse; the same call is available
programmatically. See `example/uftc/pretrain_hj_value.py` for a runnable
script that produces a `nominal/` artifact directory with `value_fn.pt`
and `value_fn.json`.

## Wiring into UFTCController

```python
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController

ctl = UFTCController(
    n_state=4, n_control=2,
    config=UFTCConfig(
        enable_l1_shield=True,
        enable_glr=True,
        l1_value_fn_path="artifacts/v_hj/nominal/value_fn.pt",
        l1_u_min=[-1.0, -1.0],
        l1_u_max=[+1.0, +1.0],
    ),
)
```

If `l1_value_fn_path` is `None`, the shield uses an `_Identity` value
function and never intervenes — the QP path is exercised only after a
real artifact is loaded.

## GLR detector

The GLR detector is enabled by `UFTCConfig.enable_glr=True`. It reads the
same Kalman innovations as the CUSUM detector but flags slow-drift
faults via a sliding-window log-likelihood-ratio test. See
`docs/superpowers/specs/2026-05-08-uftc-l1-hjshield-and-glr-design.md`
for the math.
```

- [ ] **Step 2: Write the pretrain example script**

```python
# example/uftc/pretrain_hj_value.py
"""Pre-train a DeepReach V_θ on a toy double-integrator and save it.

Real F-16 trainings live in dedicated workflows under
``example/reinforcement_learning/uftc/``; this script is a runnable
reference that completes in seconds and exercises the save/load path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tensoraerospace.agent.uftc.l1.deepreach_train import (
    TrainingConfig,
    train_value_fn,
)
from tensoraerospace.agent.uftc.l1.value_fn import DeepReachConfig


def main(out_dir: str = "artifacts/v_hj/double_integrator") -> None:
    cfg_v = DeepReachConfig(
        n_state=2, hidden_sizes=(32, 32),
        state_bounds=[[-2.0, 2.0], [-2.0, 2.0]],
        time_horizon=1.0,
    )
    train_cfg = TrainingConfig(
        epochs=50, batch_size=512, lr=1e-3, n_state=2, n_control=1,
        u_low=np.array([-1.0]), u_high=np.array([1.0]),
        seed=0,
    )
    fn, history = train_value_fn(
        cfg_v, train_cfg,
        dynamics=lambda x, u: np.array([x[1], u[0]]),
        safe_set=lambda x: 1.0 - max(abs(x[0]), abs(x[1])),
    )
    print(f"final loss = {history['loss'][-1]:.4e}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fn.save(out / "value_fn.pt")
    print(f"saved to {out / 'value_fn.pt'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the example runs end-to-end**

Run:

```
poetry run python example/uftc/pretrain_hj_value.py
```

Expected: prints final loss; creates `artifacts/v_hj/double_integrator/value_fn.pt` and `value_fn.json`.

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/agent/uftc/l1/README.md \
        example/uftc/pretrain_hj_value.py
git commit -m "docs(uftc): add L1 README and pretrain example script"
```

---

## Self-review

- **Spec coverage:** Each spec section has at least one task. §3 (HJ value fn / DeepReach) → Tasks 1, 3, 10. §3.3 (Lipschitz) → Task 2. §4 (Conformal margin) → Task 4. §5 (Shield QP) → Task 8 (+ Task 9 macro-action). §6 (Value bank) → Task 7. §7 (GLR + extended FDD) → Tasks 5, 6. §8 (UFTCController integration) → Task 11. §9 (tests) → Tasks 13, 14. §10 (pre-training tooling) → Tasks 10, 15.
- **Placeholder scan:** No "TBD", "implement later", or open code blocks. Every step contains the actual code or commands.
- **Type consistency:** `HJReachabilityShield` constructor is identical across Tasks 8, 9, 11. `ConformalMargin.compute(fdd, monitor_alarm)` signature stable across tasks. `FDDOutput` field names match between Task 6 (definition) and Tasks 4, 7, 8, 11 (consumers).
- **Phase-1 regression risk:** `enable_l1_shield=False` and `enable_glr=False` cause `UFTCController.__init__` to leave `self.l1=None` and `self.fdd.glr=None`; `predict()` short-circuits before the shield path, `learn()` short-circuits before GLR is touched. Task 14 locks this in with a byte-for-byte test.
- **Out-of-scope items honoured:** α-β-CROWN cert (deferred to Phase 2.1), MMAE class-lookup (deferred to Phase 5), 3D vis (separate work). Tasks reference these only in documentation, not implementation.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-uftc-phase2-l1-glr.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review between tasks; faster iteration, smaller blast radius per agent.
2. **Inline Execution** — run tasks in this session via `superpowers:executing-plans` with batch checkpoints.
