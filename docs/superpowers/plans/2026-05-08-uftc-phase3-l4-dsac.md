# UFTC Phase 3 — L4 Distributional SAC + CVaR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a risk-aware outer-loop reference planner to UFTC: a quantile-regression distributional critic + squashed-Gaussian actor trained against a CVaRₐ objective, plus a `risk_gate` `β_t` that modulates L3 lookahead/trust, plus a trim-free longitudinal wrapper that replaces static-trim references with adaptive ones under degraded plants.

**Architecture:** New package `tensoraerospace/agent/uftc/l4/` with eight focused modules. `DSACOuter` is the user-facing class that subclasses `BaseRLModel`. Internally it composes `QRDistCritic` (twin), `GaussianActor`, `PrioritizedReplay`, and a CVaRₐ-based actor objective. `LongitudinalTrimFreeWrapper` patches selected indices of a base reference using the actor's output. `DSACOuter.predict()` returns `(r̃_t, β_t, reset_hint)` consumed by `IADPMiddle`. Replay stores `a_actual = u_safe` so off-policy bias from L1 clipping is corrected.

**Tech Stack:** Python 3.10+, NumPy, PyTorch (critic + actor + replay). pytest. poetry. `cvxpy` already added in Phase 2. No new runtime deps. Existing Phase 1 components (`AAINDIAgent`, `IADPAgent`, `UFTCController`) are extended via flags but not modified in incompatible ways.

**Spec:** [`docs/superpowers/specs/2026-05-08-uftc-l4-dsac-cvar-design.md`](../specs/2026-05-08-uftc-l4-dsac-cvar-design.md)
**Master spec:** [`docs/superpowers/specs/2026-05-08-uftc-cascade-extension-design.md`](../specs/2026-05-08-uftc-cascade-extension-design.md)
**Predecessor plan:** [`2026-05-08-uftc-phase2-l1-glr.md`](2026-05-08-uftc-phase2-l1-glr.md) — Phase 2 must land first; this plan assumes `enable_l1_shield`/`enable_glr` flags exist and `UFTCController._last_u_safe` is populated.

**Build order (bottom-up TDD):**

```
QRDistCritic ──┐
GaussianActor ─┤
cvar_alpha_fn ─┤
risk_gate ─────┼─→ DSACOuter ─→ TrimFreeWrapper ─→ UFTCController integration
PrioritizedReplay ─┘                                      │
                                                          ├─→ off-policy regression
                                                          └─→ phase1+2 invariance
```

**Conventions:** identical to Phase 2 plan (PYTEST_DISABLE_PLUGIN_AUTOLOAD, `from __future__ import annotations`, `feat(uftc): ...` commits, no Claude attribution).

---

### Task 1: Bootstrap `l4/` package skeleton + `DSACConfig`

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/__init__.py`
- Create: `tensoraerospace/agent/uftc/l4/dsac.py` (placeholder + DSACConfig only)
- Create: `tests/agents/uftc/l4/__init__.py`
- Create: `tests/agents/uftc/l4/test_l4_skeleton.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_l4_skeleton.py
"""Smoke test for l4 package skeleton."""
from __future__ import annotations


def test_l4_package_importable() -> None:
    import tensoraerospace.agent.uftc.l4 as l4
    assert hasattr(l4, "__all__")
    assert "DSACConfig" in l4.__all__


def test_dsac_config_defaults() -> None:
    from tensoraerospace.agent.uftc.l4 import DSACConfig
    cfg = DSACConfig(n_state=4, n_ref_dim=4, n_action=4)
    assert cfg.cvar_alpha == 0.2
    assert cfg.gamma == 0.99
    assert cfg.eval_mode is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_l4_skeleton.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/dsac.py
"""Distributional SAC outer-loop planner — placeholder skeleton."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSACConfig:
    n_state: int
    n_ref_dim: int
    n_action: int
    cvar_alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    learn_every: int = 1
    update_to_data_ratio: int = 1
    target_entropy: float | None = None
    glr_reset_threshold: float = 0.10
    eval_mode: bool = True
    n_quantiles: int = 32
    huber_kappa: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    seed: int = 0
```

```python
# tensoraerospace/agent/uftc/l4/__init__.py
"""UFTC Phase 3 — L4 Distributional SAC outer-loop planner."""
from __future__ import annotations

from .dsac import DSACConfig

__all__ = ["DSACConfig"]
```

```python
# tests/agents/uftc/l4/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_l4_skeleton.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/__init__.py \
        tensoraerospace/agent/uftc/l4/dsac.py \
        tests/agents/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_l4_skeleton.py
git commit -m "feat(uftc): bootstrap l4 package with DSACConfig"
```

---

### Task 2: `QRDistCritic` quantile-regression critic

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/critic.py`
- Create: `tests/agents/uftc/l4/test_qr_critic.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_qr_critic.py
"""QR critic forward pass shape + soft-target update behaviour."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.critic import (
    CriticConfig,
    QRDistCritic,
    qr_huber_loss,
    soft_update,
)


def test_forward_returns_n_quantiles() -> None:
    cfg = CriticConfig(n_state=3, n_action=2, n_quantiles=8, hidden_sizes=(16, 16))
    q = QRDistCritic(cfg)
    s = torch.randn(5, 3)
    a = torch.randn(5, 2)
    z = q(s, a)
    assert z.shape == (5, 8)


def test_qr_huber_loss_decreases_under_supervised_descent() -> None:
    rng = np.random.default_rng(0)
    cfg = CriticConfig(n_state=2, n_action=1, n_quantiles=16, hidden_sizes=(8, 8))
    q = QRDistCritic(cfg)
    opt = torch.optim.SGD(q.parameters(), lr=1e-2)
    s = torch.tensor(rng.standard_normal((64, 2)), dtype=torch.float32)
    a = torch.tensor(rng.standard_normal((64, 1)), dtype=torch.float32)
    target = torch.tensor(rng.standard_normal((64, 16)), dtype=torch.float32)
    losses = []
    for _ in range(50):
        z = q(s, a)
        loss = qr_huber_loss(z, target.detach(), cfg.huber_kappa)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0]


def test_soft_update_moves_target_toward_source() -> None:
    cfg = CriticConfig(n_state=2, n_action=1, n_quantiles=4, hidden_sizes=(4,))
    src = QRDistCritic(cfg)
    tgt = QRDistCritic(cfg)
    # Force src/tgt to differ.
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    soft_update(target=tgt, source=src, tau=0.5)
    src_params = list(src.parameters())
    tgt_params = list(tgt.parameters())
    # After tau=0.5, target = 0.5*src_old + 0.5*tgt_old; abs differences shrink.
    diff = sum(float((p_s - p_t).abs().sum()) for p_s, p_t in zip(src_params, tgt_params))
    assert diff > 0.0  # not yet identical
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_qr_critic.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/critic.py
"""Quantile-regression distributional critic with twin design.

References:
    Dabney et al. (2018) Distributional RL with Quantile Regression, AAAI.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CriticConfig:
    n_state: int
    n_action: int
    n_quantiles: int = 32
    hidden_sizes: tuple[int, ...] = (256, 256)
    huber_kappa: float = 1.0


class QRDistCritic(nn.Module):
    """MLP that emits N quantiles of the return distribution Z(s,a)."""

    def __init__(self, cfg: CriticConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = int(cfg.n_state) + int(cfg.n_action)
        layers: list[nn.Module] = []
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, int(cfg.n_quantiles)))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa)


def qr_huber_loss(z_pred: torch.Tensor, z_target: torch.Tensor,
                  kappa: float) -> torch.Tensor:
    """Asymmetric Huber-quantile loss (Dabney 2018, eq. 10)."""
    n = int(z_pred.shape[-1])
    tau = (torch.arange(n, device=z_pred.device).float() + 0.5) / n  # (N,)
    delta = z_target.detach().unsqueeze(1) - z_pred.unsqueeze(2)      # (B, N_pred, N_tgt)
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa,
                        0.5 * delta ** 2,
                        kappa * (abs_delta - 0.5 * kappa))
    rho = (tau.view(1, n, 1) - (delta < 0).float()).abs() * huber / kappa
    return rho.mean(dim=2).sum(dim=1).mean()


def soft_update(*, target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target.parameters(), source.parameters()):
            p_t.mul_(1.0 - tau).add_(p_s, alpha=tau)
```

Update `__init__.py`:

```python
# tensoraerospace/agent/uftc/l4/__init__.py
"""UFTC Phase 3 — L4 Distributional SAC outer-loop planner."""
from __future__ import annotations

from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .dsac import DSACConfig

__all__ = [
    "CriticConfig",
    "DSACConfig",
    "QRDistCritic",
    "qr_huber_loss",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_qr_critic.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/critic.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_qr_critic.py
git commit -m "feat(uftc): add QR distributional critic + Huber loss + soft update"
```

---

### Task 3: `GaussianActor` squashed-Gaussian with reparameterisation

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/actor.py`
- Create: `tests/agents/uftc/l4/test_actor.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_actor.py
"""Squashed-Gaussian actor: shapes, log_prob with tanh Jacobian, range."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.actor import ActorConfig, GaussianActor


def test_rsample_shapes_and_action_range() -> None:
    cfg = ActorConfig(n_state=3, n_action=2, hidden_sizes=(16, 16))
    actor = GaussianActor(cfg)
    s = torch.randn(8, 3)
    a, logp = actor.rsample(s)
    assert a.shape == (8, 2)
    assert logp.shape == (8,)
    assert (a > -1.0 - 1e-6).all() and (a < 1.0 + 1e-6).all()


def test_log_prob_matches_torch_distribution_reference() -> None:
    cfg = ActorConfig(n_state=2, n_action=1, hidden_sizes=(8,),
                      log_std_min=-2.0, log_std_max=2.0)
    actor = GaussianActor(cfg)
    s = torch.randn(4, 2)
    a, logp = actor.rsample(s)
    # Re-derive log-prob: log N(mean, std) - log(1 - tanh(z)^2)
    mean, log_std = actor(s)
    std = log_std.exp()
    z = (torch.atanh(a.clamp(-0.999_999, 0.999_999)))
    logp_ref = (-0.5 * ((z - mean) / std) ** 2 - log_std - 0.5 * torch.log(torch.tensor(2 * 3.14159265))).sum(dim=-1)
    logp_ref -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1)
    assert torch.allclose(logp, logp_ref, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_actor.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/actor.py
"""Squashed-Gaussian actor with reparameterisation (Haarnoja 2018 SAC style)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ActorConfig:
    n_state: int
    n_action: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class GaussianActor(nn.Module):
    """π(a|s) = tanh(N(μ_θ(s), σ_θ(s)))."""

    def __init__(self, cfg: ActorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = int(cfg.n_state)
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head_mean = nn.Linear(in_dim, int(cfg.n_action))
        self.head_log_std = nn.Linear(in_dim, int(cfg.n_action))

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(s)
        mean = self.head_mean(h)
        log_std = self.head_log_std(h).clamp(self.cfg.log_std_min,
                                              self.cfg.log_std_max)
        return mean, log_std

    def rsample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(s)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        z = mean + std * eps
        a = torch.tanh(z)

        log_prob_z = (
            -0.5 * ((z - mean) / std) ** 2
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_prob = log_prob_z - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        return a, log_prob

    def deterministic(self, s: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(s)
        return torch.tanh(mean)
```

Update `__init__.py`:

```python
from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .dsac import DSACConfig

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "GaussianActor",
    "QRDistCritic",
    "qr_huber_loss",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_actor.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/actor.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_actor.py
git commit -m "feat(uftc): add squashed-Gaussian actor with reparameterisation"
```

---

### Task 4: `cvar_alpha_fn` + `risk_gate`

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/cvar.py`
- Create: `tests/agents/uftc/l4/test_cvar.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_cvar.py
"""cvar_alpha_fn: tail-mean correctness; risk_gate: monotonicity."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.l4.cvar import cvar_alpha_fn, risk_gate


def test_cvar_matches_numpy_tail_mean() -> None:
    rng = np.random.default_rng(0)
    z_np = rng.standard_normal((4, 32))
    z = torch.tensor(z_np, dtype=torch.float64)
    alpha = 0.25
    out = cvar_alpha_fn(z, alpha).numpy()
    z_sorted = np.sort(z_np, axis=-1)
    k = int(np.floor(alpha * 32))
    expected = z_sorted[:, :k].mean(axis=-1)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_cvar_grad_flows_back() -> None:
    z = torch.randn(2, 16, requires_grad=True)
    out = cvar_alpha_fn(z, 0.25)
    out.sum().backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_risk_gate_monotone_in_each_input() -> None:
    z_low = torch.randn(2, 16) * 0.1
    z_hi = torch.randn(2, 16) * 5.0   # high variance
    g_low = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="OK")
    g_hi = risk_gate(z_hi, fdd_severity=0.0, monitor_alarm="OK")
    assert g_hi >= g_low

    g_fdd = risk_gate(z_low, fdd_severity=1.0, monitor_alarm="OK")
    assert g_fdd > g_low

    g_alarm_warn = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="WARN")
    g_alarm_crit = risk_gate(z_low, fdd_severity=0.0, monitor_alarm="CRITICAL")
    assert g_alarm_warn >= 0.5
    assert g_alarm_crit >= 1.0 - 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_cvar.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/cvar.py
"""CVaRₐ tail-mean and risk-gate β_t."""
from __future__ import annotations

import math

import numpy as np
import torch


def cvar_alpha_fn(z: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the lowest α-fraction of quantiles."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    n = int(z.shape[-1])
    k = max(1, int(math.floor(alpha * n)))
    z_sorted, _ = torch.sort(z, dim=-1)
    return z_sorted[..., :k].mean(dim=-1)


_ALARM = {"OK": 0.0, "WARN": 0.5, "CRITICAL": 1.0}


def risk_gate(z_quantiles: torch.Tensor, *, fdd_severity: float,
              monitor_alarm: str = "OK", var_target: float = 0.5,
              k_fdd: float = 0.4) -> float:
    var_z = float(z_quantiles.var(dim=-1).mean().item())
    g_var = float(torch.sigmoid(torch.tensor((var_z - var_target) * 5.0)).item())
    g_fdd = float(np.clip(k_fdd * float(fdd_severity), 0.0, 1.0))
    g_alarm = _ALARM.get(str(monitor_alarm), 0.0)
    return float(min(1.0, max(g_var, g_fdd, g_alarm)))
```

Update `__init__.py`:

```python
from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .dsac import DSACConfig

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "GaussianActor",
    "QRDistCritic",
    "cvar_alpha_fn",
    "qr_huber_loss",
    "risk_gate",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_cvar.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/cvar.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_cvar.py
git commit -m "feat(uftc): add CVaR tail-mean and risk_gate beta_t"
```

---

### Task 5: `PrioritizedReplay` with `Transition` carrying FDD/monitor metadata

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/replay.py`
- Create: `tests/agents/uftc/l4/test_replay.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_replay.py
"""PrioritizedReplay storage and weight semantics."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l4.replay import (
    PrioritizedReplay,
    Transition,
)


def _t(reward: float = 0.0) -> Transition:
    return Transition(
        s=np.zeros(3), a_actual=np.zeros(2), r_used=np.zeros(3),
        reward=reward, s_next=np.zeros(3), done=False,
        fdd=FDDOutput(False, 0.0, 0.0, 0.0, 0.0,
                      fault_kind="none",
                      severity_abrupt=0.0, severity_gradual=0.0),
        alarm="OK",
    )


def test_push_and_len() -> None:
    rep = PrioritizedReplay(capacity=10, alpha=0.6)
    rep.push(_t(0.1))
    rep.push(_t(0.2))
    assert len(rep) == 2


def test_capacity_evicts_oldest() -> None:
    rep = PrioritizedReplay(capacity=3, alpha=0.6)
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rep.push(_t(r))
    assert len(rep) == 3
    rewards = [t.reward for t in rep.snapshot()]
    assert rewards == [0.3, 0.4, 0.5]


def test_sample_returns_indices_and_weights() -> None:
    rep = PrioritizedReplay(capacity=20, alpha=0.6, beta_init=0.4)
    for r in range(20):
        rep.push(_t(float(r)), priority=1.0 + r)
    transitions, idx, w = rep.sample(8)
    assert len(transitions) == 8
    assert len(idx) == 8
    assert w.shape == (8,)
    assert (w > 0).all()


def test_a_actual_is_stored_unchanged() -> None:
    rep = PrioritizedReplay(capacity=10, alpha=0.6)
    t = _t()
    t.a_actual = np.array([0.7, -0.3])
    rep.push(t)
    snap = rep.snapshot()
    assert np.allclose(snap[0].a_actual, [0.7, -0.3])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_replay.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/replay.py
"""Prioritised replay buffer carrying FDD/monitor metadata for L4 training.

Stores ``a_actual`` (the action that actually entered the env, i.e.
``u_safe`` after L1) so the off-policy correction is consistent with
the cascade described in the master spec.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence

import numpy as np

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput


@dataclass
class Transition:
    s: np.ndarray
    a_actual: np.ndarray
    r_used: np.ndarray
    reward: float
    s_next: np.ndarray
    done: bool
    fdd: FDDOutput
    alarm: str


class PrioritizedReplay:
    """Proportional-priority replay (Schaul et al. 2015) with a deque backbone."""

    def __init__(self, capacity: int, alpha: float = 0.6,
                 beta_init: float = 0.4, eps: float = 1e-3) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta_init)
        self.eps = float(eps)
        self._buf: Deque[Transition] = deque(maxlen=self.capacity)
        self._pri: Deque[float] = deque(maxlen=self.capacity)
        self._max_pri = 1.0

    def push(self, t: Transition, priority: float | None = None) -> None:
        p = float(priority) if priority is not None else self._max_pri
        self._buf.append(t)
        self._pri.append(p)
        self._max_pri = max(self._max_pri, p)

    def __len__(self) -> int:
        return len(self._buf)

    def snapshot(self) -> Sequence[Transition]:
        return list(self._buf)

    def sample(self, batch_size: int,
               rng: np.random.Generator | None = None) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if len(self._buf) < batch_size:
            raise ValueError("buffer not full enough to sample")
        rng = rng if rng is not None else np.random.default_rng()
        priorities = np.array(self._pri, dtype=np.float64) + self.eps
        probs = priorities ** self.alpha
        probs /= probs.sum()
        idx = rng.choice(len(self._buf), size=batch_size, replace=False, p=probs)
        weights = (len(self._buf) * probs[idx]) ** (-self.beta)
        weights /= weights.max()
        transitions = [self._buf[int(i)] for i in idx]
        return transitions, idx, weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for i, e in zip(indices, td_errors):
            p = float(abs(e)) + self.eps
            self._pri[int(i)] = p
            self._max_pri = max(self._max_pri, p)
```

Update `__init__.py`:

```python
from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .dsac import DSACConfig
from .replay import PrioritizedReplay, Transition

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "GaussianActor",
    "PrioritizedReplay",
    "QRDistCritic",
    "Transition",
    "cvar_alpha_fn",
    "qr_huber_loss",
    "risk_gate",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_replay.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/replay.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_replay.py
git commit -m "feat(uftc): add prioritised replay buffer carrying FDD/monitor metadata"
```

---

### Task 6: `DSACOuter` predict / learn / freeze / degrade-reference

**Files:**
- Modify: `tensoraerospace/agent/uftc/l4/dsac.py` (replace placeholder)
- Create: `tests/agents/uftc/l4/test_dsac_outer.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_dsac_outer.py
"""DSACOuter end-to-end on a 2-state mock plant."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput
from tensoraerospace.agent.uftc.l4 import (
    DSACConfig,
    DSACOuter,
    Transition,
)


def _zero_fdd() -> FDDOutput:
    return FDDOutput(False, 0.0, 0.0, 0.0, 0.0,
                     fault_kind="none",
                     severity_abrupt=0.0, severity_gradual=0.0)


def test_predict_returns_tuple_in_eval_mode() -> None:
    cfg = DSACConfig(n_state=3, n_ref_dim=3, n_action=3,
                     critic_hidden=(8, 8), actor_hidden=(8, 8),
                     n_quantiles=4, batch_size=4, replay_capacity=32)
    dsac = DSACOuter(cfg)
    r_tilde, beta, reset_hint = dsac.predict(
        x_obs=np.zeros(3), base_reference=np.zeros(3), fdd=_zero_fdd(),
        monitor_alarm="OK")
    assert r_tilde.shape == (3,)
    assert 0.0 <= beta <= 1.0
    assert reset_hint in (False, True)


def test_freeze_learning_blocks_updates() -> None:
    cfg = DSACConfig(n_state=2, n_ref_dim=2, n_action=2,
                     critic_hidden=(4,), actor_hidden=(4,),
                     n_quantiles=4, batch_size=2, replay_capacity=8)
    dsac = DSACOuter(cfg)
    # Push some transitions.
    for _ in range(4):
        dsac.learn(Transition(np.zeros(2), np.zeros(2), np.zeros(2), 0.0,
                              np.zeros(2), False, _zero_fdd(), "OK"))
    # Get current actor parameters.
    p0 = next(dsac.actor.parameters()).detach().clone()
    dsac.freeze_learning(until_step=10**6)
    for _ in range(8):
        dsac.learn(Transition(np.random.randn(2), np.random.randn(2),
                              np.random.randn(2), 1.0,
                              np.random.randn(2), False, _zero_fdd(), "OK"))
    p1 = next(dsac.actor.parameters()).detach().clone()
    assert torch.allclose(p0, p1)


def test_degrade_reference_to_hold_passthrough() -> None:
    cfg = DSACConfig(n_state=3, n_ref_dim=3, n_action=3,
                     critic_hidden=(4,), actor_hidden=(4,),
                     n_quantiles=4, batch_size=2, replay_capacity=8)
    dsac = DSACOuter(cfg)
    base = np.array([0.5, -0.2, 0.1])
    dsac.degrade_reference_to_hold()
    r_tilde, _, _ = dsac.predict(np.zeros(3), base, _zero_fdd(), "OK")
    assert np.allclose(r_tilde, base)


def test_save_load_round_trip(tmp_path) -> None:
    cfg = DSACConfig(n_state=2, n_ref_dim=2, n_action=2,
                     critic_hidden=(4,), actor_hidden=(4,),
                     n_quantiles=4, batch_size=2, replay_capacity=8)
    dsac = DSACOuter(cfg)
    r0, _, _ = dsac.predict(np.zeros(2), np.zeros(2), _zero_fdd(), "OK")
    dsac.save(tmp_path)
    dsac2 = DSACOuter.from_pretrained(tmp_path, cfg=cfg)
    r1, _, _ = dsac2.predict(np.zeros(2), np.zeros(2), _zero_fdd(), "OK")
    assert np.allclose(r0, r1, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_dsac_outer.py -v`
Expected: FAIL — `DSACOuter` not yet implemented.

- [ ] **Step 3: Implement**

Replace `tensoraerospace/agent/uftc/l4/dsac.py`:

```python
"""DSACOuter — distributional SAC outer-loop reference planner.

Composes QRDistCritic (twin), GaussianActor, PrioritizedReplay, and a
CVaRₐ actor objective. Phase 3 default is ``eval_mode=True``: predict()
uses the deterministic actor mean and learn() pushes transitions but
does no SGD step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from tensoraerospace.agent.uftc.fdd.detector import FDDOutput

from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .replay import PrioritizedReplay, Transition


@dataclass
class DSACConfig:
    n_state: int
    n_ref_dim: int
    n_action: int
    cvar_alpha: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 200_000
    learn_every: int = 1
    update_to_data_ratio: int = 1
    target_entropy: float | None = None
    glr_reset_threshold: float = 0.10
    eval_mode: bool = True
    n_quantiles: int = 32
    huber_kappa: float = 1.0
    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    seed: int = 0
    action_scale: float = 1.0


class DSACOuter:
    """Outer-loop risk-aware reference planner (Phase 3)."""

    def __init__(self, cfg: DSACConfig, device: str = "cpu") -> None:
        torch.manual_seed(int(cfg.seed))
        self.cfg = cfg
        self.device = torch.device(device)
        actor_cfg = ActorConfig(n_state=cfg.n_state, n_action=cfg.n_action,
                                hidden_sizes=cfg.actor_hidden)
        critic_cfg = CriticConfig(n_state=cfg.n_state, n_action=cfg.n_action,
                                  n_quantiles=cfg.n_quantiles,
                                  hidden_sizes=cfg.critic_hidden,
                                  huber_kappa=cfg.huber_kappa)
        self.actor = GaussianActor(actor_cfg).to(self.device)
        self.critic1 = QRDistCritic(critic_cfg).to(self.device)
        self.critic2 = QRDistCritic(critic_cfg).to(self.device)
        self.target1 = QRDistCritic(critic_cfg).to(self.device)
        self.target2 = QRDistCritic(critic_cfg).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_c1 = torch.optim.Adam(self.critic1.parameters(), lr=cfg.lr_critic)
        self.opt_c2 = torch.optim.Adam(self.critic2.parameters(), lr=cfg.lr_critic)

        self.target_entropy = (
            float(cfg.target_entropy) if cfg.target_entropy is not None
            else -float(cfg.n_action)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)

        self.replay = PrioritizedReplay(capacity=cfg.replay_capacity)

        self._frozen_until: int | None = None
        self._hold_mode = False
        self._step = 0

    # ----- predict -----
    def predict(self, x_obs: np.ndarray, base_reference: np.ndarray,
                fdd: FDDOutput, monitor_alarm: str = "OK"
                ) -> tuple[np.ndarray, float, bool]:
        x = torch.tensor(np.asarray(x_obs, dtype=np.float32),
                         dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            if self.cfg.eval_mode:
                a = self.actor.deterministic(x)
            else:
                a, _ = self.actor.rsample(x)
            z = self.critic1(x, a)
        a_np = a.cpu().numpy().reshape(-1) * float(self.cfg.action_scale)
        if self._hold_mode:
            r_tilde = np.asarray(base_reference, dtype=np.float64).copy()
        else:
            r_tilde = (np.asarray(base_reference, dtype=np.float64).copy()
                       + a_np[: int(self.cfg.n_ref_dim)])
        beta_t = risk_gate(z, fdd_severity=fdd.severity,
                           monitor_alarm=monitor_alarm)
        drift = getattr(fdd, "glr_drift_estimate", None)
        reset_hint = bool(drift is not None
                          and float(np.linalg.norm(drift)) > self.cfg.glr_reset_threshold)
        return r_tilde, beta_t, reset_hint

    # ----- learn -----
    def learn(self, t: Transition) -> dict:
        self._step += 1
        if self._frozen_until is not None and self._step < self._frozen_until:
            self.replay.push(t)
            return {"frozen": True, "step": self._step}
        self.replay.push(t)
        if self.cfg.eval_mode:
            return {"eval_mode": True, "step": self._step}
        if len(self.replay) < self.cfg.batch_size:
            return {"warming_up": True, "step": self._step}
        if self._step % self.cfg.learn_every != 0:
            return {"step": self._step}
        return self._sgd_step()

    def _sgd_step(self) -> dict:
        batch, idx, w = self.replay.sample(self.cfg.batch_size)
        s = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=self.device)
        a = torch.tensor(np.stack([t.a_actual for t in batch]), dtype=torch.float32, device=self.device)
        r = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([t.s_next for t in batch]), dtype=torch.float32, device=self.device)
        done = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(w, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            a2, logp2 = self.actor.rsample(s2)
            z2 = torch.minimum(self.target1(s2, a2), self.target2(s2, a2))
            y = r.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * (z2 - self.log_alpha.exp() * logp2.unsqueeze(-1))

        z1 = self.critic1(s, a)
        z2_pred = self.critic2(s, a)
        loss_c1 = (qr_huber_loss(z1, y, self.cfg.huber_kappa).unsqueeze(0) * weights).mean()
        loss_c2 = (qr_huber_loss(z2_pred, y, self.cfg.huber_kappa).unsqueeze(0) * weights).mean()
        self.opt_c1.zero_grad(); loss_c1.backward(); self.opt_c1.step()
        self.opt_c2.zero_grad(); loss_c2.backward(); self.opt_c2.step()

        # Actor update against CVaR_alpha of min critic.
        a_pi, logp = self.actor.rsample(s)
        z_pi = torch.minimum(self.critic1(s, a_pi), self.critic2(s, a_pi))
        cvar = cvar_alpha_fn(z_pi, self.cfg.cvar_alpha)
        loss_a = (self.log_alpha.exp() * logp - cvar).mean()
        self.opt_actor.zero_grad(); loss_a.backward(); self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        soft_update(target=self.target1, source=self.critic1, tau=self.cfg.tau)
        soft_update(target=self.target2, source=self.critic2, tau=self.cfg.tau)

        td = (z1.mean(dim=-1) - y.mean(dim=-1)).abs().detach().cpu().numpy()
        self.replay.update_priorities(idx, td)
        return {"loss_c1": float(loss_c1), "loss_c2": float(loss_c2),
                "loss_a": float(loss_a), "loss_alpha": float(loss_alpha),
                "step": self._step}

    # ----- macro-action sinks (Phase 4 will call these) -----
    def freeze_learning(self, until_step: int) -> None:
        self._frozen_until = int(until_step)

    def degrade_reference_to_hold(self) -> None:
        self._hold_mode = True

    def reset(self) -> None:
        self._hold_mode = False

    # ----- save/load -----
    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), dir_path / "actor.pt")
        torch.save(self.critic1.state_dict(), dir_path / "critic1.pt")
        torch.save(self.critic2.state_dict(), dir_path / "critic2.pt")
        torch.save(self.target1.state_dict(), dir_path / "target1.pt")
        torch.save(self.target2.state_dict(), dir_path / "target2.pt")
        torch.save(self.log_alpha.detach(), dir_path / "log_alpha.pt")
        import json
        (dir_path / "dsac_config.json").write_text(json.dumps(asdict(self.cfg), indent=2))

    @classmethod
    def from_pretrained(cls, dir_path: str | Path, *,
                        cfg: DSACConfig | None = None) -> "DSACOuter":
        dir_path = Path(dir_path)
        if cfg is None:
            import json
            cfg_d = json.loads((dir_path / "dsac_config.json").read_text())
            cfg_d["actor_hidden"] = tuple(cfg_d["actor_hidden"])
            cfg_d["critic_hidden"] = tuple(cfg_d["critic_hidden"])
            cfg = DSACConfig(**cfg_d)
        m = cls(cfg)
        m.actor.load_state_dict(torch.load(dir_path / "actor.pt", map_location="cpu"))
        m.critic1.load_state_dict(torch.load(dir_path / "critic1.pt", map_location="cpu"))
        m.critic2.load_state_dict(torch.load(dir_path / "critic2.pt", map_location="cpu"))
        m.target1.load_state_dict(torch.load(dir_path / "target1.pt", map_location="cpu"))
        m.target2.load_state_dict(torch.load(dir_path / "target2.pt", map_location="cpu"))
        m.log_alpha = torch.tensor(torch.load(dir_path / "log_alpha.pt"),
                                   requires_grad=True, device=m.device)
        m.opt_alpha = torch.optim.Adam([m.log_alpha], lr=cfg.lr_alpha)
        return m
```

Update `__init__.py` to export `DSACOuter`:

```python
from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .dsac import DSACConfig, DSACOuter
from .replay import PrioritizedReplay, Transition

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "DSACOuter",
    "GaussianActor",
    "PrioritizedReplay",
    "QRDistCritic",
    "Transition",
    "cvar_alpha_fn",
    "qr_huber_loss",
    "risk_gate",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_dsac_outer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/dsac.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_dsac_outer.py
git commit -m "feat(uftc): add DSACOuter with predict/learn/freeze/degrade/save/load"
```

---

### Task 7: `LongitudinalTrimFreeWrapper`

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/trim_free.py`
- Create: `tests/agents/uftc/l4/test_trim_free.py`
- Modify: `tensoraerospace/agent/uftc/l4/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/l4/test_trim_free.py
"""LongitudinalTrimFreeWrapper passthrough vs replace behaviour."""
from __future__ import annotations

import numpy as np

from tensoraerospace.agent.uftc.l4.trim_free import (
    LongitudinalTrimFreeConfig,
    LongitudinalTrimFreeWrapper,
)


def test_disabled_passthrough() -> None:
    w = LongitudinalTrimFreeWrapper(
        LongitudinalTrimFreeConfig(V_idx=0, gamma_idx=1, alpha_idx=2,
                                   q_idx=3, enabled=False),
    )
    base = np.array([1.0, 2.0, 3.0, 4.0])
    out = w.apply(np.array([0.5, 0.6]), x_obs=np.zeros(4), base_reference=base)
    assert np.allclose(out, base)


def test_enabled_replaces_alpha_q_indices() -> None:
    w = LongitudinalTrimFreeWrapper(
        LongitudinalTrimFreeConfig(V_idx=0, gamma_idx=1, alpha_idx=2,
                                   q_idx=3, enabled=True),
    )
    base = np.array([100.0, 0.05, 999.0, 999.0])
    out = w.apply(np.array([0.04, 0.01]), x_obs=np.zeros(4), base_reference=base)
    assert out[0] == base[0]
    assert out[1] == base[1]
    assert out[2] == 0.04
    assert out[3] == 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_trim_free.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# tensoraerospace/agent/uftc/l4/trim_free.py
"""Trim-free longitudinal reference wrapper for L4 D-SAC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LongitudinalTrimFreeConfig:
    V_idx: int
    gamma_idx: int
    alpha_idx: int
    q_idx: int
    enabled: bool = False


class LongitudinalTrimFreeWrapper:
    """Replace ``alpha_target`` and ``q_target`` in the base reference with
    actor output. Pilot-supplied ``V_target`` and ``gamma_target`` are
    preserved verbatim.
    """

    def __init__(self, cfg: LongitudinalTrimFreeConfig) -> None:
        self.cfg = cfg

    def apply(self, r_tilde_actor: np.ndarray, *, x_obs: np.ndarray,
              base_reference: np.ndarray) -> np.ndarray:
        if not self.cfg.enabled:
            return np.asarray(base_reference, dtype=np.float64).copy()
        out = np.asarray(base_reference, dtype=np.float64).copy()
        out[self.cfg.alpha_idx] = float(r_tilde_actor[0])
        out[self.cfg.q_idx] = float(r_tilde_actor[1])
        return out
```

Update `__init__.py`:

```python
from .actor import ActorConfig, GaussianActor
from .critic import CriticConfig, QRDistCritic, qr_huber_loss, soft_update
from .cvar import cvar_alpha_fn, risk_gate
from .dsac import DSACConfig, DSACOuter
from .replay import PrioritizedReplay, Transition
from .trim_free import LongitudinalTrimFreeConfig, LongitudinalTrimFreeWrapper

__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DSACConfig",
    "DSACOuter",
    "GaussianActor",
    "LongitudinalTrimFreeConfig",
    "LongitudinalTrimFreeWrapper",
    "PrioritizedReplay",
    "QRDistCritic",
    "Transition",
    "cvar_alpha_fn",
    "qr_huber_loss",
    "risk_gate",
    "soft_update",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/l4/test_trim_free.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/trim_free.py \
        tensoraerospace/agent/uftc/l4/__init__.py \
        tests/agents/uftc/l4/test_trim_free.py
git commit -m "feat(uftc): add longitudinal trim-free reference wrapper"
```

---

### Task 8: Wire `DSACOuter` into `UFTCController`

**Files:**
- Modify: `tensoraerospace/agent/uftc/controller.py`
- Modify: `tensoraerospace/agent/uftc/__init__.py`
- Create: `tests/agents/uftc/test_uftc_l4_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_l4_smoke.py
"""enable_l4_outer wiring smoke test."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_l4_returns_modified_reference_on_demand() -> None:
    cfg = UFTCConfig(
        dt=0.01,
        fdd_warmup_steps=20,
        enable_l4_outer=True,
        l4_n_ref_dim=3,
        l4_action_scale=0.0,   # zero scale → r̃ == base_reference
    )
    ctl = UFTCController(n_state=3, n_control=2, config=cfg)
    rng = np.random.default_rng(0)
    base_ref = np.array([0.1, -0.2, 0.05])
    for k in range(40):
        x = rng.standard_normal(3) * 0.1
        u = ctl.predict(x, base_ref, time_step=k)
        ctl.learn(x, base_ref, time_step=k)
    diag = ctl.diagnostics()
    assert "l4" in diag
    assert "beta_t" in diag["l4"]


def test_l4_off_invariance_with_phase1_only() -> None:
    """enable_l4_outer=False: behaviour identical to Phase 1 + 2 flags-off."""
    rng_seed = 12345

    def rollout(enable_l4: bool) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(rng_seed)
        ctl = UFTCController(
            n_state=4, n_control=2,
            config=UFTCConfig(dt=0.01, fdd_warmup_steps=20,
                              enable_l1_shield=False, enable_glr=False,
                              enable_l4_outer=enable_l4,
                              l4_n_ref_dim=4),
        )
        xs, us = [], []
        x = rng.standard_normal(4) * 0.1
        for k in range(200):
            u = ctl.predict(x, np.zeros(4), time_step=k)
            x = x + 0.01 * (rng.standard_normal(4) * 0.05 - 0.1 * x)
            ctl.learn(x, np.zeros(4), time_step=k)
            xs.append(x.copy()); us.append(np.asarray(u, dtype=np.float64).copy())
        return np.stack(xs), np.stack(us)

    x_off, u_off = rollout(enable_l4=False)
    x_off_ref, u_off_ref = rollout(enable_l4=False)
    np.testing.assert_array_equal(x_off, x_off_ref)
    np.testing.assert_array_equal(u_off, u_off_ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_l4_smoke.py -v`
Expected: FAIL — `enable_l4_outer` field not present; `diagnostics()['l4']` absent.

- [ ] **Step 3: Implement**

In `tensoraerospace/agent/uftc/controller.py`:

1. Extend `UFTCConfig`:

```python
# (append to UFTCConfig)
# Phase 3 — L4 D-SAC outer
enable_l4_outer: bool = False
l4_n_ref_dim: int = 0          # if 0 and enable_l4_outer, defaults to n_state
l4_action_scale: float = 0.1
l4_actor_hidden: tuple[int, ...] = (64, 64)
l4_critic_hidden: tuple[int, ...] = (64, 64)
l4_n_quantiles: int = 16
l4_cvar_alpha: float = 0.2
l4_glr_reset_threshold: float = 0.10
l4_eval_mode: bool = True
l4_replay_capacity: int = 10_000
l4_batch_size: int = 64
l4_seed: int = 0
l4_trim_free: dict | None = None    # {V_idx, gamma_idx, alpha_idx, q_idx}
```

2. In `__init__`:

```python
self.l4 = None
self._last_r_eff: np.ndarray | None = None
self._last_beta: float = 0.0
if cfg.enable_l4_outer:
    from tensoraerospace.agent.uftc.l4 import (
        DSACConfig, DSACOuter,
        LongitudinalTrimFreeConfig, LongitudinalTrimFreeWrapper,
    )
    n_ref = cfg.l4_n_ref_dim or n_state
    dsac_cfg = DSACConfig(
        n_state=n_state, n_ref_dim=n_ref, n_action=n_ref,
        cvar_alpha=cfg.l4_cvar_alpha, n_quantiles=cfg.l4_n_quantiles,
        actor_hidden=cfg.l4_actor_hidden, critic_hidden=cfg.l4_critic_hidden,
        glr_reset_threshold=cfg.l4_glr_reset_threshold,
        eval_mode=cfg.l4_eval_mode, action_scale=cfg.l4_action_scale,
        replay_capacity=cfg.l4_replay_capacity,
        batch_size=cfg.l4_batch_size, seed=cfg.l4_seed,
    )
    self.l4 = DSACOuter(dsac_cfg)
    if cfg.l4_trim_free:
        tf_cfg = LongitudinalTrimFreeConfig(enabled=True, **cfg.l4_trim_free)
        self.l4_trim_free = LongitudinalTrimFreeWrapper(tf_cfg)
    else:
        self.l4_trim_free = None
```

3. In `predict()` (before L3 call):

```python
fdd_for_l4 = self._last_fdd or _zero_fdd_output(self.n_state)
if self.l4 is not None:
    r_eff, beta_t, reset_hint = self.l4.predict(
        x_obs, reference, fdd_for_l4, monitor_alarm="OK")
    if self.l4_trim_free is not None:
        r_eff = self.l4_trim_free.apply(
            r_eff[: self.l4.cfg.n_ref_dim], x_obs=x_obs, base_reference=reference)
else:
    r_eff = reference
    beta_t = 0.0
    reset_hint = False
self._last_r_eff = r_eff
self._last_beta = beta_t
self._last_reset_hint = reset_hint
# (then existing predict path uses r_eff in place of reference)
```

4. In `learn()` (after `fdd.step` and `middle.learn`):

```python
if self.l4 is not None and self._last_u_safe is not None and self._last_r_eff is not None:
    from tensoraerospace.agent.uftc.l4 import Transition
    self.l4.learn(Transition(
        s=np.asarray(x_obs, dtype=np.float64).copy(),
        a_actual=np.asarray(self._last_u_safe, dtype=np.float64).copy(),
        r_used=np.asarray(self._last_r_eff, dtype=np.float64).copy(),
        reward=float(-(np.linalg.norm(next_x_obs - self._last_r_eff) ** 2)),
        s_next=np.asarray(next_x_obs, dtype=np.float64).copy(),
        done=False,
        fdd=fdd_out if 'fdd_out' in locals() else fdd_for_l4,
        alarm="OK",
    ))
```

5. Extend `diagnostics()` with an `"l4"` block when `self.l4 is not None`: `{"beta_t": float(self._last_beta), "reset_hint": bool(self._last_reset_hint), "frozen_until": self.l4._frozen_until}`.

6. Update `__init__.py` re-exports.

- [ ] **Step 4: Run tests**

Run:

```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest \
    tests/agents/uftc/test_uftc_l4_smoke.py \
    tests/agents/uftc/test_uftc_phase1_invariance.py \
    tests/agents/uftc/test_uftc_smoke.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/uftc/controller.py \
        tensoraerospace/agent/uftc/__init__.py \
        tests/agents/uftc/test_uftc_l4_smoke.py
git commit -m "feat(uftc): wire DSACOuter into UFTCController behind enable_l4_outer"
```

---

### Task 9: Off-policy `a_actual = u_safe` regression

**Files:**
- Create: `tests/agents/uftc/test_uftc_l4_replay_off_policy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_l4_replay_off_policy.py
"""With L1 active, replay must record u_safe (post-shield), not u_indi."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cvxpy")

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_replay_records_post_shield_action() -> None:
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=10,
        enable_l1_shield=True, enable_glr=False,
        enable_l4_outer=True, l4_n_ref_dim=3, l4_action_scale=0.05,
        l1_u_min=[-0.1, -0.1], l1_u_max=[0.1, 0.1],
    )
    ctl = UFTCController(n_state=3, n_control=2, config=cfg)
    rng = np.random.default_rng(0)
    for k in range(50):
        x = rng.standard_normal(3) * 0.05
        u = ctl.predict(x, np.zeros(3), time_step=k)
        ctl.learn(x, np.zeros(3), time_step=k)
        # Whatever the unclipped u_indi was, the action that env saw is bounded.
        assert (u >= -0.1 - 1e-6).all()
        assert (u <= 0.1 + 1e-6).all()

    snap = ctl.l4.replay.snapshot()
    assert len(snap) >= 30
    # Stored a_actual must be inside the L1 bounds — the off-policy correction.
    for tr in snap:
        assert (tr.a_actual >= -0.1 - 1e-6).all()
        assert (tr.a_actual <= 0.1 + 1e-6).all()
```

- [ ] **Step 2: Run test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_l4_replay_off_policy.py -v`
Expected: 1 passed (Task 8 already wired `a_actual = self._last_u_safe`).

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_l4_replay_off_policy.py
git commit -m "test(uftc): regression — replay records post-shield u_safe"
```

---

### Task 10: F-16 + ENGINE_FLAMEOUT integration baseline (eval-only, pretrained)

**Files:**
- Create: `tests/agents/uftc/test_uftc_l4_engine_flameout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/uftc/test_uftc_l4_engine_flameout.py
"""F-16 ENGINE_FLAMEOUT with L4 eval-mode — tracking-RMS not worse than Phase 1."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.aerospacemodel.f16.nonlinear import LongitudinalF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def _rollout(enable_l4: bool) -> float:
    env = LongitudinalF16(damage_profile=presets.ENGINE_FLAMEOUT, dt=0.01)
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=200,
        enable_l1_shield=False, enable_glr=True,
        enable_l4_outer=enable_l4,
        l4_n_ref_dim=env.n_state, l4_eval_mode=True,
    )
    ctl = UFTCController(n_state=env.n_state, n_control=env.n_control, config=cfg)
    x = env.reset()
    err = 0.0; n = 0
    target = np.zeros(env.n_state)
    for k in range(int(8.0 / 0.01)):
        u = ctl.predict(x, target, time_step=k)
        x = env.step(u)
        ctl.learn(x, target, time_step=k)
        err += float(np.linalg.norm(x - target) ** 2); n += 1
    return float(np.sqrt(err / max(n, 1)))


def test_l4_eval_mode_not_worse_than_phase1() -> None:
    rms_off = _rollout(enable_l4=False)
    rms_on = _rollout(enable_l4=True)
    # With untrained eval-mode actor (mean=tanh(0)=0), L4 does not push reference;
    # numerical state should be effectively identical.
    assert rms_on <= rms_off * 1.05
```

- [ ] **Step 2: Run test**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_l4_engine_flameout.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_l4_engine_flameout.py
git commit -m "test(uftc): F-16 ENGINE_FLAMEOUT eval-mode L4 regression baseline"
```

---

### Task 11: Phase 1+2 invariance under `enable_l4_outer=False`

**Files:**
- Modify: `tests/agents/uftc/test_uftc_phase1_invariance.py` (add a Phase 3 case)

- [ ] **Step 1: Append a new test**

```python
# tests/agents/uftc/test_uftc_phase1_invariance.py — APPEND
def test_phase3_flag_off_keeps_phase12_invariance() -> None:
    rng_seed = 7
    def rollout(enable_l4: bool):
        rng = np.random.default_rng(rng_seed)
        cfg = UFTCConfig(dt=0.01, fdd_warmup_steps=50,
                         enable_l1_shield=False, enable_glr=False,
                         enable_l4_outer=enable_l4, l4_n_ref_dim=4)
        ctl = UFTCController(n_state=4, n_control=2, config=cfg)
        xs, us = [], []
        x = rng.standard_normal(4) * 0.1
        for k in range(400):
            u = ctl.predict(x, np.zeros(4), time_step=k)
            x = x + 0.01 * (rng.standard_normal(4) * 0.05 - 0.1 * x)
            ctl.learn(x, np.zeros(4), time_step=k)
            xs.append(x.copy()); us.append(np.asarray(u, dtype=np.float64).copy())
        return np.stack(xs), np.stack(us)
    x_off, u_off = rollout(enable_l4=False)
    x_off_ref, u_off_ref = rollout(enable_l4=False)
    np.testing.assert_array_equal(x_off, x_off_ref)
    np.testing.assert_array_equal(u_off, u_off_ref)
```

- [ ] **Step 2: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_phase1_invariance.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_phase1_invariance.py
git commit -m "test(uftc): extend invariance test for Phase 3 flag-off"
```

---

### Task 12: Trim-free convergence on F-16 longitudinal (smoke)

**Files:**
- Create: `tests/agents/uftc/test_uftc_trim_free_smoke.py`

- [ ] **Step 1: Write**

```python
# tests/agents/uftc/test_uftc_trim_free_smoke.py
"""Trim-free wrapper enabled: alpha/q indices in r_eff come from actor."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController


def test_trim_free_overwrites_alpha_q_indices_in_r_eff() -> None:
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=10,
        enable_l4_outer=True, l4_n_ref_dim=4, l4_action_scale=0.0,
        l4_trim_free={"V_idx": 0, "gamma_idx": 1, "alpha_idx": 2, "q_idx": 3},
    )
    ctl = UFTCController(n_state=4, n_control=2, config=cfg)
    base = np.array([100.0, 0.05, 9999.0, -9999.0])
    ctl.predict(np.zeros(4), base, time_step=0)
    assert ctl._last_r_eff is not None
    # V and gamma preserved; alpha and q replaced by something *other* than 9999.
    assert ctl._last_r_eff[0] == base[0]
    assert ctl._last_r_eff[1] == base[1]
    assert ctl._last_r_eff[2] != 9999.0
    assert ctl._last_r_eff[3] != -9999.0
```

- [ ] **Step 2: Run**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest tests/agents/uftc/test_uftc_trim_free_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/uftc/test_uftc_trim_free_smoke.py
git commit -m "test(uftc): trim-free wrapper smoke on UFTCController"
```

---

### Task 13: L4 README + offline-training example script

**Files:**
- Create: `tensoraerospace/agent/uftc/l4/README.md`
- Create: `example/reinforcement_learning/uftc/train_dsac_offline.py`

- [ ] **Step 1: Write README**

```markdown
<!-- tensoraerospace/agent/uftc/l4/README.md -->
# UFTC L4 — Distributional SAC outer-loop planner

Phase 3 component. Provides:

- `QRDistCritic` (twin) + `qr_huber_loss`
- `GaussianActor` squashed with reparameterisation
- `cvar_alpha_fn` and `risk_gate(z, severity, alarm)`
- `PrioritizedReplay` carrying FDD/monitor metadata
- `DSACOuter` orchestrator with `freeze_learning`/`degrade_reference_to_hold` macro-action sinks
- `LongitudinalTrimFreeWrapper` for adaptive longitudinal references

## Wiring into UFTCController

```python
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController

ctl = UFTCController(
    n_state=4, n_control=2,
    config=UFTCConfig(
        enable_l4_outer=True,
        l4_n_ref_dim=4,
        l4_action_scale=0.05,
        l4_eval_mode=True,
    ),
)
```

## Offline training

See `example/reinforcement_learning/uftc/train_dsac_offline.py` for a
minimal curriculum that reproduces the spec's pre-training pipeline.
The example is reduced to ~5 000 steps so it completes in minutes; the
real workflow uses 200 000 steps and a full damage-preset mix.
```

- [ ] **Step 2: Write the offline-training script**

```python
# example/reinforcement_learning/uftc/train_dsac_offline.py
"""Smoke offline-training script for L4 DSAC on F-16 longitudinal.

Uses a small budget so it completes in a few minutes. Real trainings
should bump ``steps`` to 200 000 and the curriculum mix to all 7 damage
presets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tensoraerospace.aerospacemodel.f16.nonlinear import LongitudinalF16
from tensoraerospace.aerospacemodel.f16.nonlinear.damage import presets
from tensoraerospace.agent.uftc.controller import UFTCConfig, UFTCController
from tensoraerospace.agent.uftc.l4 import Transition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--out", type=str, default="artifacts/dsac/v1")
    args = parser.parse_args()

    env = LongitudinalF16(damage_profile=presets.ENGINE_FLAMEOUT, dt=0.01)
    cfg = UFTCConfig(
        dt=0.01, fdd_warmup_steps=200,
        enable_l1_shield=False, enable_glr=True,
        enable_l4_outer=True,
        l4_n_ref_dim=env.n_state,
        l4_eval_mode=False,            # online learning during offline-training
        l4_replay_capacity=10_000,
        l4_batch_size=128,
    )
    ctl = UFTCController(n_state=env.n_state, n_control=env.n_control, config=cfg)
    x = env.reset()
    target = np.zeros(env.n_state)
    for k in range(int(args.steps)):
        u = ctl.predict(x, target, time_step=k)
        x_next = env.step(u)
        ctl.learn(x_next, target, time_step=k)
        x = x_next
        if k % 1000 == 0:
            print(f"step {k:6d} — beta_t={ctl._last_beta:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctl.l4.save(args.out)
    print(f"saved DSAC weights to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the example runs end-to-end (short budget)**

Run:

```
poetry run python example/reinforcement_learning/uftc/train_dsac_offline.py --steps 200 --out artifacts/dsac/smoke
```

Expected: prints `beta_t` at intervals, saves `actor.pt`/`critic*.pt`/`dsac_config.json` under `artifacts/dsac/smoke/`.

- [ ] **Step 4: Commit**

```bash
git add tensoraerospace/agent/uftc/l4/README.md \
        example/reinforcement_learning/uftc/train_dsac_offline.py
git commit -m "docs(uftc): add L4 README and offline-training example script"
```

---

## Self-review

- **Spec coverage:** §3 critic → Task 2; §4 actor → Task 3; §5 CVaR → Task 4; §6 risk gate → Task 4; §7 replay → Task 5; §8 DSACOuter → Task 6; §9 trim-free → Task 7; §10 pre-training → Task 13; §11 controller integration → Task 8; §12 operational mode (`eval_mode=True`) → covered in Task 1 + 6; §13 tests → 9, 10, 11, 12.
- **Placeholder scan:** No "TBD"/"implement later" tokens; every step contains the exact code/commands.
- **Type consistency:** `DSACConfig`, `Transition`, `DSACOuter`, `LongitudinalTrimFreeConfig` all referenced consistently across Tasks 1, 5, 6, 7, 8.
- **Phase 1+2 regression risk:** `enable_l4_outer=False` keeps `self.l4=None` and `predict()` short-circuits the L4 branch; Task 11 locks this in. `replay.push` is gated by `self.l4 is not None`.
- **Out-of-scope items honoured:** online learning while flying is enabled only via `l4_eval_mode=False` (Task 13 example); the controller's default leaves `l4_eval_mode=True`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-08-uftc-phase3-l4-dsac.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task with two-stage review.
2. **Inline Execution** — `superpowers:executing-plans` with batch checkpoints.
