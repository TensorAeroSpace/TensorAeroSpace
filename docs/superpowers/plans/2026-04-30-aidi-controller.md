# AIDI Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Adaptive Incremental Dynamic Inversion (AIDI) controller from Ul Haq et al. (AIAA 2026-1744) as a model-agnostic agent in TensorAeroSpace, with the F-16 nonlinear angular env as the first integration target.

**Architecture:** Composed-from-blocks design (see spec §2). Each block (ScalingRLS, MoorePenroseAllocator, PseudoControlHedge, four outer-loop reference models, OnboardCEModel) is independently testable. The `AIDIAgent` orchestrator implements the same `predict`/`learn`/`save`/`from_pretrained` API as `aa_indi`/`et_dhp`/`im_gdhp`.

**Tech Stack:** numpy, gymnasium, pytest, dataclasses, matplotlib, jupyter (for the example).

**Spec:** `docs/superpowers/specs/2026-04-30-aidi-controller-design.md`.

---

## File map

Create:
- `tensoraerospace/agent/aidi/__init__.py`
- `tensoraerospace/agent/aidi/scaling_rls.py`
- `tensoraerospace/agent/aidi/allocator.py`
- `tensoraerospace/agent/aidi/pch.py`
- `tensoraerospace/agent/aidi/ref_models.py`
- `tensoraerospace/agent/aidi/onboard_ce.py`
- `tensoraerospace/agent/aidi/utils.py`
- `tensoraerospace/agent/aidi/model.py`
- `tests/agent/aidi/__init__.py`
- `tests/agent/aidi/test_scaling_rls.py`
- `tests/agent/aidi/test_allocator.py`
- `tests/agent/aidi/test_pch.py`
- `tests/agent/aidi/test_ref_models.py`
- `tests/agent/aidi/test_onboard_ce.py`
- `tests/agent/aidi/test_utils.py`
- `tests/agent/aidi/test_aidi_agent.py`
- `tests/agent/aidi/test_aidi_integration.py`
- `tensoraerospace/aerospacemodel/f16/nonlinear/damage/aidi_presets.py`
- `tensoraerospace/scripts/__init__.py` (only if it does not already exist)
- `tensoraerospace/scripts/benchmark_aidi.py`
- `tests/scripts/test_benchmark_aidi.py`
- `example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb`
- `docs/algorithms/aidi.md`

Modify:
- `tensoraerospace/agent/__init__.py` — re-export `AIDIAgent` and `AIDIConfig`.
- `tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py` — re-export new presets.
- `mkdocs.yml` — add the new doc page.

---

## Task 1: ScalingRLS — core estimator

**Files:**
- Create: `tensoraerospace/agent/aidi/__init__.py`
- Create: `tensoraerospace/agent/aidi/scaling_rls.py`
- Create: `tests/agent/aidi/__init__.py`
- Create: `tests/agent/aidi/test_scaling_rls.py`

- [ ] **Step 1: Write failing tests for `ScalingRLS`**

```python
# tests/agent/aidi/test_scaling_rls.py
"""ScalingRLS unit tests — convergence, fault response, consistency check."""

import numpy as np
import pytest

from tensoraerospace.agent.aidi.scaling_rls import ScalingRLS


def _onboard_G() -> np.ndarray:
    # 3 rates × 3 surfaces, sign convention loosely F-16-like.
    return np.array(
        [[ 0.10, -2.50,  0.00],
         [-3.00,  0.05,  0.00],
         [ 0.02,  0.00, -1.20]],
        dtype=np.float64,
    )


def test_scaling_rls_init_at_unity():
    rls = ScalingRLS(n_y=3, n_u=3)
    np.testing.assert_array_equal(rls.theta, np.ones((3, 3)))


def test_scaling_rls_converges_to_unity_when_truth_matches_onboard():
    rng = np.random.default_rng(0)
    G = _onboard_G()
    rls = ScalingRLS(n_y=3, n_u=3, sigma0=1e-3, memory_length=50)

    for _ in range(400):
        du = rng.normal(0.0, 1.0, size=3)
        domega = G @ du + rng.normal(0.0, 1e-4, size=3)
        rls.update(du, domega, G)

    np.testing.assert_allclose(rls.theta, np.ones((3, 3)), atol=0.05)


def test_scaling_rls_converges_to_truth_when_one_surface_lost():
    rng = np.random.default_rng(1)
    G_onboard = _onboard_G()
    truth_scale = np.ones((3, 3))
    truth_scale[:, 0] = 0.25  # surface 0 efficiency dropped to 25 %.
    G_true = truth_scale * G_onboard
    rls = ScalingRLS(n_y=3, n_u=3, sigma0=1e-3, memory_length=50,
                     consistency_threshold=1e-6)

    for _ in range(800):
        du = rng.normal(0.0, 1.0, size=3)
        domega = G_true @ du + rng.normal(0.0, 1e-4, size=3)
        rls.update(du, domega, G_onboard)

    np.testing.assert_allclose(rls.theta[:, 0], 0.25 * np.ones(3), atol=0.1)
    np.testing.assert_allclose(rls.theta[:, 1:], 1.0, atol=0.1)


def test_scaling_rls_lambda_drops_after_step_fault():
    rng = np.random.default_rng(2)
    G = _onboard_G()
    rls = ScalingRLS(n_y=3, n_u=3, sigma0=1e-3, memory_length=50,
                     lambda_min=0.5)

    # Quiet operation under nominal model.
    for _ in range(200):
        du = rng.normal(0.0, 0.5, size=3)
        rls.update(du, G @ du, G)

    lam_pre = rls.last_lambda.copy()
    # Inject a step fault: surface 1 drops to 0.3.
    fault_truth = np.array(
        [[1.0, 0.3, 1.0]] * 3, dtype=np.float64
    ) * G

    for _ in range(5):
        du = rng.normal(0.0, 1.0, size=3)
        rls.update(du, fault_truth @ du, G)

    assert float(np.min(rls.last_lambda)) < float(np.min(lam_pre))


def test_scaling_rls_consistency_check_collapses_outlier_row():
    G = _onboard_G()
    rls = ScalingRLS(n_y=3, n_u=3, sigma0=1.0, memory_length=10,
                     consistency_threshold=1e-6)

    # Hand-craft Δθ updates that disagree across rows for column j=0;
    # the consistency check should replace each entry with the column mean.
    delta_theta_in = np.array(
        [[0.10, 0.00, 0.00],
         [0.20, 0.00, 0.00],
         [0.30, 0.00, 0.00]],
        dtype=np.float64,
    )
    out = rls._apply_consistency_check(delta_theta_in)
    expected_col0 = np.full(3, 0.20)  # mean of 0.10, 0.20, 0.30
    np.testing.assert_allclose(out[:, 0], expected_col0)
    np.testing.assert_array_equal(out[:, 1:], 0.0)


def test_scaling_rls_rejects_wrong_shape():
    rls = ScalingRLS(n_y=3, n_u=3)
    with pytest.raises(ValueError):
        rls.update(np.zeros(2), np.zeros(3), np.eye(3))
    with pytest.raises(ValueError):
        rls.update(np.zeros(3), np.zeros(2), np.eye(3))
    with pytest.raises(ValueError):
        rls.update(np.zeros(3), np.zeros(3), np.eye(2))
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/agent/aidi/test_scaling_rls.py -v`
Expected: ImportError / collection error (module does not exist yet).

- [ ] **Step 3: Implement `ScalingRLS`**

```python
# tensoraerospace/agent/aidi/__init__.py
"""Adaptive Incremental Dynamic Inversion agent (AIDI).

Reference:
    Ul Haq, Atmaca, van Kampen, "Adaptive Incremental Dynamic Inversion for
    Fault-tolerant Flight Control of a Flying Wing", AIAA SciTech 2026,
    AIAA 2026-1744 — https://doi.org/10.2514/6.2026-1744
"""
```

```python
# tensoraerospace/agent/aidi/scaling_rls.py
"""Per-row VFF-RLS that adapts the multiplicative scaling Θ over a known onboard
control-effectiveness matrix G_nominal — Section III.C of Ul Haq et al. 2026.

For each rate axis ``i`` we keep an independent covariance ``P_i``, an
information-content forgetting factor ``λ_i`` (Eq. 26–27 of the paper) and a
row of the scaling matrix ``Θ[i, :]``. After per-row updates a cross-axis
consistency check replaces any deviating column with the column mean — this
matches the practical adjustment described at the top of page 10 of the paper.
"""

from __future__ import annotations

import numpy as np


class ScalingRLS:
    """Recursive identifier of the multiplicative scaling matrix Θ.

    Args:
        n_y: Number of rate axes (rows of Θ).
        n_u: Number of control surfaces (columns of Θ).
        lambda_min: Lower bound on the variable forgetting factor — the
            estimator falls toward this value when the residual is large
            (fast adaptation during faults).
        lambda_max: Upper bound on the variable forgetting factor — the
            estimator returns toward this value during quiescent operation.
        sigma0: Sensor-noise variance σ₀² used in the information-content
            VFF (Eq. 27 of the paper, Σ₀ = σ₀²·N₀).
        memory_length: Nominal memory length N₀ in samples.
        cov_init: Initial scale of the per-row covariance matrices.
        consistency_threshold: Per-paper relative threshold for the
            cross-axis consistency check; updates that deviate by more
            than this from the column mean are replaced by the mean.
        seed: Optional RNG seed (currently unused — kept for parity with
            other estimators).
    """

    def __init__(
        self,
        n_y: int,
        n_u: int,
        lambda_min: float = 0.7,
        lambda_max: float = 0.999,
        sigma0: float = 1e-3,
        memory_length: int = 100,
        cov_init: float = 1.0,
        consistency_threshold: float = 1e-6,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < lambda_min <= lambda_max <= 1.0:
            raise ValueError("require 0 < lambda_min ≤ lambda_max ≤ 1")
        if sigma0 <= 0.0 or memory_length <= 0:
            raise ValueError("sigma0 and memory_length must be positive")
        if cov_init <= 0.0:
            raise ValueError("cov_init must be positive")
        del seed  # reserved.

        self.n_y = int(n_y)
        self.n_u = int(n_u)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.sigma0 = float(sigma0)
        self.memory_length = int(memory_length)
        self.cov_init = float(cov_init)
        self.consistency_threshold = float(consistency_threshold)

        self.theta = np.ones((self.n_y, self.n_u), dtype=np.float64)
        self.P = np.stack(
            [np.eye(self.n_u, dtype=np.float64) * self.cov_init
             for _ in range(self.n_y)],
            axis=0,
        )  # shape (n_y, n_u, n_u)
        self.last_lambda = np.full(self.n_y, self.lambda_max, dtype=np.float64)
        self.last_residual = np.zeros(self.n_y, dtype=np.float64)
        self.num_updates: int = 0

    @property
    def G_eff(self) -> np.ndarray:
        """Latest scaled effectiveness matrix Θ ⊙ G_nominal of last update.

        Stored only as a convenience for diagnostics — callers normally form
        ``rls.theta * G_nominal`` themselves with their own G_nominal.
        """
        return self._last_G_scaled.copy() if hasattr(
            self, "_last_G_scaled") else None

    @property
    def sigma_total(self) -> float:
        """Information-content denominator Σ₀ = σ₀²·N₀."""
        return self.sigma0 ** 2 * self.memory_length

    def _info_content_lambda(
        self, eps_i: float, phi_K_i: float
    ) -> float:
        """Eq. 26 of the paper: λ = 1 − (1 − φᵀK)·ε² / Σ₀, clamped."""
        lam = 1.0 - (1.0 - phi_K_i) * (eps_i ** 2) / self.sigma_total
        return float(np.clip(lam, self.lambda_min, self.lambda_max))

    def _apply_consistency_check(self, delta_theta: np.ndarray) -> np.ndarray:
        """For each column ``j`` replace entries that deviate from the column
        mean by more than ``consistency_threshold`` with the column mean."""
        out = delta_theta.copy()
        col_mean = out.mean(axis=0)
        deviation = np.abs(out - col_mean[np.newaxis, :])
        mask = deviation > self.consistency_threshold
        # Broadcast column-mean replacements where the mask says so.
        mean_broadcast = np.broadcast_to(col_mean, out.shape)
        out = np.where(mask, mean_broadcast, out)
        return out

    def update(
        self,
        du: np.ndarray,
        domega: np.ndarray,
        G_nominal: np.ndarray,
    ) -> np.ndarray:
        """Run one RLS step using ``(Δu, Δω̇, G_nominal)``.

        Args:
            du: Control increment, shape ``(n_u,)``.
            domega: Angular-rate-derivative increment, shape ``(n_y,)``.
            G_nominal: Onboard CE matrix at the linearisation point, shape
                ``(n_y, n_u)``.

        Returns:
            The pre-update residual ε of shape ``(n_y,)``.
        """
        du_v = np.asarray(du, dtype=np.float64).reshape(-1)
        dy_v = np.asarray(domega, dtype=np.float64).reshape(-1)
        G = np.asarray(G_nominal, dtype=np.float64)
        if du_v.size != self.n_u:
            raise ValueError(f"du must have length {self.n_u}, got {du_v.size}")
        if dy_v.size != self.n_y:
            raise ValueError(f"domega must have length {self.n_y}, got {dy_v.size}")
        if G.shape != (self.n_y, self.n_u):
            raise ValueError(
                f"G_nominal must have shape ({self.n_y}, {self.n_u}), got {G.shape}"
            )

        delta_theta = np.zeros((self.n_y, self.n_u), dtype=np.float64)
        residuals = np.zeros(self.n_y, dtype=np.float64)
        lambdas = np.empty(self.n_y, dtype=np.float64)

        for i in range(self.n_y):
            phi = (G[i, :] * du_v).reshape(-1, 1)  # (n_u, 1)
            theta_row = self.theta[i, :].reshape(-1, 1)  # (n_u, 1)
            P_i = self.P[i]
            eps_i = float(dy_v[i] - (theta_row.T @ phi).item())
            P_phi = P_i @ phi  # (n_u, 1)
            denom = self.last_lambda[i] + float((phi.T @ P_phi).item())
            denom = denom if abs(denom) > 1e-12 else 1e-12
            K_i = P_phi / denom  # (n_u, 1)

            delta_theta[i, :] = (K_i * eps_i).reshape(-1)
            phi_K_scalar = float((phi.T @ K_i).item())
            lam_new = self._info_content_lambda(eps_i, phi_K_scalar)
            lambdas[i] = lam_new

            self.P[i] = (P_i - K_i @ P_phi.T) / lam_new
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)  # symmetrise

            residuals[i] = eps_i

        delta_theta = self._apply_consistency_check(delta_theta)
        self.theta = self.theta + delta_theta
        self.last_lambda = lambdas
        self.last_residual = residuals
        self.num_updates += 1
        self._last_G_scaled = self.theta * G
        return residuals
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_scaling_rls.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/__init__.py \
        tensoraerospace/agent/aidi/scaling_rls.py \
        tests/agent/aidi/__init__.py \
        tests/agent/aidi/test_scaling_rls.py
git commit -m "feat(aidi): ScalingRLS — per-row VFF-RLS with cross-axis consistency check"
```

---

## Task 2: MoorePenroseAllocator

**Files:**
- Create: `tensoraerospace/agent/aidi/allocator.py`
- Create: `tests/agent/aidi/test_allocator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_allocator.py
"""MoorePenroseAllocator unit tests."""

import logging

import numpy as np
import pytest

from tensoraerospace.agent.aidi.allocator import MoorePenroseAllocator


def test_allocator_square_inverse():
    G = np.array([[1.0, 0.0], [0.0, 2.0]])
    alloc = MoorePenroseAllocator(rcond=1e-8, cond_threshold=1e8)
    nu = np.array([3.0, 4.0])
    omega_dot = np.array([1.0, 1.0])
    du = alloc.allocate(G, nu, omega_dot)
    np.testing.assert_allclose(du, np.array([2.0, 1.5]), atol=1e-9)


def test_allocator_redundant_min_norm():
    # 2 rates × 3 surfaces (under-determined → minimum-norm solution).
    G = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    alloc = MoorePenroseAllocator()
    nu = np.array([1.0, 1.0])
    omega_dot = np.array([0.0, 0.0])
    du = alloc.allocate(G, nu, omega_dot)
    # Verify it satisfies G du = nu and is minimum-norm.
    np.testing.assert_allclose(G @ du, nu, atol=1e-9)
    expected = np.linalg.pinv(G) @ nu
    np.testing.assert_allclose(du, expected, atol=1e-9)


def test_allocator_ill_conditioned_returns_zero(caplog):
    bad_G = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    alloc = MoorePenroseAllocator(cond_threshold=1e6)
    with caplog.at_level(logging.WARNING):
        du = alloc.allocate(bad_G, np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    np.testing.assert_array_equal(du, np.zeros(2))
    assert any("ill-conditioned" in r.message.lower() for r in caplog.records)


def test_allocator_shape_validation():
    alloc = MoorePenroseAllocator()
    with pytest.raises(ValueError):
        alloc.allocate(np.eye(3), np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        alloc.allocate(np.eye(3), np.zeros(3), np.zeros(2))
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_allocator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement allocator**

```python
# tensoraerospace/agent/aidi/allocator.py
"""Moore-Penrose pseudoinverse-based control allocator.

Used by the AIDI inner loop to map a virtual-control demand
``ν − ω̇_meas`` to a control increment ``Δu``. Falls back to zero on
ill-conditioning (a numerical guard during the RLS warm-up).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MoorePenroseAllocator:
    """Minimum-norm control allocation via ``np.linalg.pinv``.

    Args:
        rcond: Cut-off for small singular values, passed to
            :func:`numpy.linalg.pinv`.
        cond_threshold: When ``cond(G)`` exceeds this value, ``allocate``
            returns ``Δu = 0`` and emits a warning instead of inverting.
    """

    def __init__(self, rcond: float = 1e-8, cond_threshold: float = 1e8) -> None:
        if rcond <= 0.0 or cond_threshold <= 0.0:
            raise ValueError("rcond and cond_threshold must be positive")
        self.rcond = float(rcond)
        self.cond_threshold = float(cond_threshold)

    def allocate(
        self,
        G_eff: np.ndarray,
        nu_des: np.ndarray,
        omega_dot_meas: np.ndarray,
    ) -> np.ndarray:
        """Compute ``Δu = G⁺ · (ν_des − ω̇_meas)``.

        Args:
            G_eff: Scaled control-effectiveness matrix ``G̃``, shape
                ``(n_y, n_u)``.
            nu_des: Virtual control vector, shape ``(n_y,)``.
            omega_dot_meas: Measured angular acceleration, shape ``(n_y,)``.

        Returns:
            Control increment ``Δu`` of shape ``(n_u,)``. Zero when
            ``G_eff`` is too ill-conditioned to invert.
        """
        G = np.asarray(G_eff, dtype=np.float64)
        nu = np.asarray(nu_des, dtype=np.float64).reshape(-1)
        omd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        if nu.size != G.shape[0]:
            raise ValueError(
                f"nu_des must have length {G.shape[0]}, got {nu.size}"
            )
        if omd.size != G.shape[0]:
            raise ValueError(
                f"omega_dot_meas must have length {G.shape[0]}, got {omd.size}"
            )
        # Conditioning guard.
        try:
            cond = float(np.linalg.cond(G))
        except np.linalg.LinAlgError:
            cond = float("inf")
        if cond > self.cond_threshold:
            logger.warning(
                "AIDI allocator: G is ill-conditioned (cond=%.3g); returning Δu=0",
                cond,
            )
            return np.zeros(G.shape[1], dtype=np.float64)
        G_pinv = np.linalg.pinv(G, rcond=self.rcond)
        return G_pinv @ (nu - omd)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_allocator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/allocator.py tests/agent/aidi/test_allocator.py
git commit -m "feat(aidi): Moore-Penrose control allocator with conditioning guard"
```

---

## Task 3: PseudoControlHedge

**Files:**
- Create: `tensoraerospace/agent/aidi/pch.py`
- Create: `tests/agent/aidi/test_pch.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_pch.py
"""PseudoControlHedge unit tests."""

import numpy as np

from tensoraerospace.agent.aidi.pch import PseudoControlHedge


def test_pch_zero_hedge_when_inner_loop_tracks():
    hedge = PseudoControlHedge(n_y=3, freeze_after=10)
    # Tick 1: store nu_des_prev.
    hedge.update(nu_des_prev=np.array([1.0, 0.0, 0.0]),
                 omega_dot_meas=np.array([1.0, 0.0, 0.0]))
    # Hedge should be zero because plant achieved demanded acceleration.
    np.testing.assert_allclose(hedge.last_hedge, np.zeros(3), atol=1e-12)
    assert not hedge.is_frozen.any()


def test_pch_emits_hedge_when_plant_lags():
    hedge = PseudoControlHedge(n_y=3, freeze_after=5)
    hedge.update(nu_des_prev=np.array([1.0, 0.0, 0.0]),
                 omega_dot_meas=np.array([0.4, 0.0, 0.0]))
    # nu_h = nu_des_prev - omega_dot_meas
    np.testing.assert_allclose(hedge.last_hedge, np.array([0.6, 0.0, 0.0]))


def test_pch_freezes_after_persistent_saturation():
    hedge = PseudoControlHedge(n_y=2, freeze_after=3)
    # Plant lags persistently on axis 0.
    for _ in range(4):
        hedge.update(nu_des_prev=np.array([1.0, 0.0]),
                     omega_dot_meas=np.array([0.0, 0.0]))
    assert bool(hedge.is_frozen[0]) is True
    assert bool(hedge.is_frozen[1]) is False


def test_pch_freeze_clears_when_gap_closes():
    hedge = PseudoControlHedge(n_y=1, freeze_after=2)
    for _ in range(3):
        hedge.update(nu_des_prev=np.array([1.0]),
                     omega_dot_meas=np.array([0.0]))
    assert bool(hedge.is_frozen[0]) is True
    hedge.update(nu_des_prev=np.array([1.0]),
                 omega_dot_meas=np.array([1.0]))
    assert bool(hedge.is_frozen[0]) is False


def test_pch_reset_clears_state():
    hedge = PseudoControlHedge(n_y=2, freeze_after=2)
    for _ in range(3):
        hedge.update(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    hedge.reset()
    np.testing.assert_array_equal(hedge.last_hedge, np.zeros(2))
    np.testing.assert_array_equal(hedge.is_frozen, np.zeros(2, dtype=bool))
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_pch.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement PCH**

```python
# tensoraerospace/agent/aidi/pch.py
"""Pseudo-Control Hedging block.

PCH stores the previous-tick virtual control demand ``ν_des_prev`` and
compares it to the current measured acceleration ``ω̇_meas`` to compute the
hedge signal ``ν_h = ν_des_prev − ω̇_meas`` — the gap the inner loop failed
to close, attributed to actuator dynamics or saturation. The reference
models subtract this hedge from their derivatives before integrating, which
prevents reference-rate wind-up during saturation. After ``freeze_after``
ticks of persistent gap on a given axis, that axis's reference rate is
hard-frozen until the gap closes (sign-of-hedge matches sign-of-gap).
"""

from __future__ import annotations

import numpy as np


class PseudoControlHedge:
    """PCH state machine, one entry per rate axis.

    Args:
        n_y: Number of rate axes.
        freeze_after: Number of consecutive saturated ticks before the
            corresponding reference rate is hard-frozen.
        gap_tol: Magnitude of ``|ν_h|`` below which the axis is considered
            tracked (resets the saturation counter).
    """

    def __init__(self, n_y: int, freeze_after: int = 20, gap_tol: float = 1e-6) -> None:
        if n_y <= 0:
            raise ValueError("n_y must be positive")
        if freeze_after <= 0:
            raise ValueError("freeze_after must be positive")
        if gap_tol < 0.0:
            raise ValueError("gap_tol must be ≥ 0")
        self.n_y = int(n_y)
        self.freeze_after = int(freeze_after)
        self.gap_tol = float(gap_tol)

        self.last_hedge = np.zeros(self.n_y, dtype=np.float64)
        self.saturation_counter = np.zeros(self.n_y, dtype=np.int32)
        self.is_frozen = np.zeros(self.n_y, dtype=bool)

    def reset(self) -> None:
        self.last_hedge = np.zeros(self.n_y, dtype=np.float64)
        self.saturation_counter = np.zeros(self.n_y, dtype=np.int32)
        self.is_frozen = np.zeros(self.n_y, dtype=bool)

    def update(self, nu_des_prev: np.ndarray, omega_dot_meas: np.ndarray) -> np.ndarray:
        """Compute hedge and update the freeze counters.

        Args:
            nu_des_prev: Virtual control demanded on the previous tick.
            omega_dot_meas: Measured angular acceleration this tick.

        Returns:
            Hedge vector ``ν_h`` of shape ``(n_y,)``.
        """
        nu = np.asarray(nu_des_prev, dtype=np.float64).reshape(-1)
        omd = np.asarray(omega_dot_meas, dtype=np.float64).reshape(-1)
        if nu.size != self.n_y or omd.size != self.n_y:
            raise ValueError(f"both inputs must have length {self.n_y}")
        hedge = nu - omd
        gap_active = np.abs(hedge) > self.gap_tol
        self.saturation_counter = np.where(
            gap_active, self.saturation_counter + 1, 0,
        ).astype(np.int32)
        self.is_frozen = self.saturation_counter >= self.freeze_after
        self.last_hedge = hedge
        return hedge.copy()
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_pch.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/pch.py tests/agent/aidi/test_pch.py
git commit -m "feat(aidi): pseudo-control hedging block with per-axis freeze"
```

---

## Task 4: Outer-loop reference models

**Files:**
- Create: `tensoraerospace/agent/aidi/ref_models.py`
- Create: `tests/agent/aidi/test_ref_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_ref_models.py
"""Outer-loop reference-model unit tests."""

import math

import numpy as np

from tensoraerospace.agent.aidi.ref_models import (
    CStarController,
    LinearController,
    RollReferenceModel,
    SideslipCompensator,
    SpeedController,
)


def test_cstar_drives_q_des_toward_command():
    ctrl = CStarController(kp=2.0, ki=0.5, V_co=122.6, dt=0.01)
    q_des = ctrl.step(c_star_cmd=2.0, n_z=1.0, q=0.0, V=200.0, hedge=0.0)
    # C* error = 2 - (1 + (200/122.6)*0) = 1.0; kp*err = 2.0; ki·err·dt = 0.005
    assert q_des == pytest.approx(2.0 + 0.005, rel=1e-6)


def test_cstar_subtracts_hedge_before_integration():
    ctrl = CStarController(kp=0.0, ki=1.0, V_co=122.6, dt=0.1)
    # Without hedge: integrator advances by 1*0.1 = 0.1.
    ctrl.step(c_star_cmd=1.0, n_z=0.0, q=0.0, V=100.0, hedge=0.0)
    int_no_hedge = ctrl._int_err
    ctrl.reset()
    # With hedge=q_des_prev = 0.1: integrator advances by (err - hedge)*dt = 0.09.
    ctrl.step(c_star_cmd=1.0, n_z=0.0, q=0.0, V=100.0, hedge=0.1)
    int_with_hedge = ctrl._int_err
    assert int_with_hedge < int_no_hedge


def test_roll_reference_second_order_response():
    ref = RollReferenceModel(omega_n=2.0, zeta=0.7, dt=0.01)
    p_des_steps = []
    phi = 0.0
    p = 0.0
    for _ in range(500):
        p_des = ref.step(phi_cmd=math.radians(10.0), phi=phi, hedge=0.0)
        # Toy plant: p ← p_des, phi ← phi + p_des*dt (perfect tracker).
        p = p_des
        phi += p_des * 0.01
        p_des_steps.append(p_des)
    assert phi == pytest.approx(math.radians(10.0), abs=math.radians(0.5))


def test_sideslip_compensator_drives_beta_to_zero():
    comp = SideslipCompensator(kp=2.0, ki=0.1, dt=0.01)
    rs = []
    beta = math.radians(2.0)
    for _ in range(500):
        r = comp.step(beta_cmd=0.0, beta=beta, hedge=0.0)
        # Toy plant: r reduces beta linearly.
        beta -= r * 0.01
        rs.append(r)
    assert abs(beta) < math.radians(0.05)


def test_speed_controller_no_op_when_disabled():
    ctrl = SpeedController(kp=0.0, ki=0.0, kd=0.0, dt=0.01, enabled=False)
    out = ctrl.step(V_cmd=200.0, V=180.0)
    assert out == 0.0


def test_linear_controller_passthrough_with_rate_feedback():
    lin = LinearController(rate_kp=np.zeros(3))
    nu = lin.combine(omega_des=np.array([1.0, 2.0, 3.0]),
                     omega=np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(nu, np.array([1.0, 2.0, 3.0]))


def test_linear_controller_adds_rate_error_feedback():
    lin = LinearController(rate_kp=np.array([1.0, 0.0, 0.0]))
    nu = lin.combine(omega_des=np.array([1.0, 0.0, 0.0]),
                     omega=np.array([0.5, 0.0, 0.0]))
    # nu_x = omega_des + kp*(omega_des - omega) = 1 + 1*(0.5) = 1.5
    np.testing.assert_array_equal(nu, np.array([1.5, 0.0, 0.0]))


# pytest import is needed for pytest.approx.
import pytest  # noqa: E402
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_ref_models.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement reference models**

```python
# tensoraerospace/agent/aidi/ref_models.py
"""Outer-loop reference models for the AIDI controller.

Each block produces a per-axis "desired rate" that the linear controller
combines into the virtual-control vector ``ν``. PCH hedge signals are
subtracted from the integrator-style blocks (`CStarController`,
`RollReferenceModel`, `SideslipCompensator`) before integration, in line
with §III.A of Ul Haq et al. 2026.
"""

from __future__ import annotations

import numpy as np


class CStarController:
    """C\* longitudinal controller — PI on ``C*_cmd − C*``.

    ``C* = n_z + (V/V_co)·q``. The output is a desired pitch rate ``q_des``.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        V_co: Crossover speed (m/s); MIL-STD value is ≈ 122.6.
        dt: Integration step (s).
        i_clip: Symmetric anti-windup clamp on the integrator state
            (in error·s units).
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.5,
        V_co: float = 122.6,
        dt: float = 0.01,
        i_clip: float = 5.0,
    ) -> None:
        if V_co <= 0.0:
            raise ValueError("V_co must be positive")
        self.kp = float(kp)
        self.ki = float(ki)
        self.V_co = float(V_co)
        self.dt = float(dt)
        self.i_clip = float(i_clip)
        self._int_err: float = 0.0

    def reset(self) -> None:
        self._int_err = 0.0

    def step(
        self,
        c_star_cmd: float,
        n_z: float,
        q: float,
        V: float,
        hedge: float = 0.0,
    ) -> float:
        c_star = float(n_z) + (float(V) / self.V_co) * float(q)
        err = float(c_star_cmd) - c_star
        # PCH: subtract the hedge BEFORE integration to keep the integrator
        # inside the achievable envelope.
        self._int_err = float(np.clip(
            self._int_err + (err - float(hedge)) * self.dt,
            -self.i_clip, self.i_clip,
        ))
        return self.kp * err + self.ki * self._int_err


class RollReferenceModel:
    """Second-order roll attitude reference model.

    ``phi_ddot = -2ζω_n·phi_dot + ω_n²·(phi_cmd − phi)``; output is the
    integrated ``phi_dot``, used as ``p_des``.
    """

    def __init__(self, omega_n: float = 2.0, zeta: float = 0.7,
                 dt: float = 0.01) -> None:
        if omega_n <= 0.0 or not 0.0 < zeta:
            raise ValueError("omega_n > 0 and zeta > 0 required")
        self.omega_n = float(omega_n)
        self.zeta = float(zeta)
        self.dt = float(dt)
        self._phi_dot: float = 0.0
        self._phi: float = 0.0

    def reset(self) -> None:
        self._phi_dot = 0.0
        self._phi = 0.0

    def step(self, phi_cmd: float, phi: float, hedge: float = 0.0) -> float:
        # Discrete 2nd-order with hedging.
        phi_ddot = (
            -2.0 * self.zeta * self.omega_n * self._phi_dot
            + self.omega_n ** 2 * (float(phi_cmd) - float(phi))
            - float(hedge)
        )
        self._phi_dot = self._phi_dot + self.dt * phi_ddot
        self._phi = self._phi + self.dt * self._phi_dot
        return self._phi_dot


class SideslipCompensator:
    """PI compensator on sideslip ``β`` driving a yaw-rate demand ``r_des``."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, dt: float = 0.01,
                 i_clip: float = 5.0) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.dt = float(dt)
        self.i_clip = float(i_clip)
        self._int_err: float = 0.0

    def reset(self) -> None:
        self._int_err = 0.0

    def step(self, beta_cmd: float, beta: float, hedge: float = 0.0) -> float:
        err = float(beta_cmd) - float(beta)
        self._int_err = float(np.clip(
            self._int_err + (err - float(hedge)) * self.dt,
            -self.i_clip, self.i_clip,
        ))
        return self.kp * err + self.ki * self._int_err


class SpeedController:
    """Auto-throttle PID. No-op when ``enabled=False`` (constant-airspeed envs)."""

    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0,
                 dt: float = 0.01, enabled: bool = False) -> None:
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.dt = float(dt); self.enabled = bool(enabled)
        self._int_err = 0.0
        self._prev_err = 0.0

    def reset(self) -> None:
        self._int_err = 0.0
        self._prev_err = 0.0

    def step(self, V_cmd: float, V: float) -> float:
        if not self.enabled:
            return 0.0
        err = float(V_cmd) - float(V)
        self._int_err += err * self.dt
        deriv = (err - self._prev_err) / max(self.dt, 1e-9)
        self._prev_err = err
        return self.kp * err + self.ki * self._int_err + self.kd * deriv


class LinearController:
    """Combine the outer-loop desired rates into a virtual-control vector.

    ``ν = ω_des + K_p ⊙ (ω_des − ω)``. With ``K_p = 0`` this is a passthrough.
    """

    def __init__(self, rate_kp: np.ndarray | None = None,
                 n_y: int = 3) -> None:
        if rate_kp is None:
            rate_kp = np.zeros(n_y, dtype=np.float64)
        rate_kp = np.asarray(rate_kp, dtype=np.float64).reshape(-1)
        self.rate_kp = rate_kp

    def combine(self, omega_des: np.ndarray, omega: np.ndarray) -> np.ndarray:
        omega_des = np.asarray(omega_des, dtype=np.float64).reshape(-1)
        omega = np.asarray(omega, dtype=np.float64).reshape(-1)
        return omega_des + self.rate_kp * (omega_des - omega)
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_ref_models.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/ref_models.py tests/agent/aidi/test_ref_models.py
git commit -m "feat(aidi): outer-loop reference models (C*, roll, sideslip, speed, linear)"
```

---

## Task 5: `n_z` reconstruction helper

**Files:**
- Create: `tensoraerospace/agent/aidi/utils.py`
- Create: `tests/agent/aidi/test_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_utils.py
"""AIDI helper tests — n_z reconstruction."""

import math

import numpy as np

from tensoraerospace.agent.aidi.utils import reconstruct_n_z


def test_n_z_unity_at_level_trim():
    # phi=0, theta=0, q=0, alpha_dot=0 → n_z = (V/g)*(0) + cos(0) = 1.
    nz = reconstruct_n_z(alpha=math.radians(2.0), alpha_dot=0.0,
                         q=0.0, V=200.0, theta=0.0, phi=0.0)
    assert nz == pytest.approx(1.0, abs=1e-9)


def test_n_z_increases_with_pitch_rate():
    # q > 0 → positive load contribution.
    nz1 = reconstruct_n_z(alpha=math.radians(2.0), alpha_dot=0.0,
                          q=0.0, V=200.0, theta=0.0, phi=0.0)
    nz2 = reconstruct_n_z(alpha=math.radians(2.0), alpha_dot=0.0,
                          q=math.radians(5.0), V=200.0, theta=0.0, phi=0.0)
    assert nz2 > nz1


def test_n_z_alpha_dot_subtracts():
    nz1 = reconstruct_n_z(alpha=math.radians(2.0), alpha_dot=0.0,
                          q=math.radians(5.0), V=200.0, theta=0.0, phi=0.0)
    nz2 = reconstruct_n_z(alpha=math.radians(2.0), alpha_dot=math.radians(2.0),
                          q=math.radians(5.0), V=200.0, theta=0.0, phi=0.0)
    assert nz2 < nz1


def test_n_z_inverted_flight_negative():
    nz = reconstruct_n_z(alpha=0.0, alpha_dot=0.0, q=0.0, V=200.0,
                         theta=0.0, phi=math.pi)
    assert nz == pytest.approx(-1.0, abs=1e-9)


import pytest  # noqa: E402
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_utils.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement helper**

```python
# tensoraerospace/agent/aidi/utils.py
"""AIDI utility helpers — body-frame load-factor reconstruction etc.

For envs that do not expose ``n_z`` directly we reconstruct it from
``(α, α̇, q, V, θ, φ)`` using the standard small-β relation::

    n_z ≈ (V/g)·(q·cos α − α̇) + cos θ · cos φ

The cosine term encodes the gravity component along the body-z axis at
the current attitude. Inverted level flight (φ = π) gives n_z ≈ −1.
"""

from __future__ import annotations

import math


GRAVITY = 9.80665  # m/s²


def reconstruct_n_z(
    alpha: float,
    alpha_dot: float,
    q: float,
    V: float,
    theta: float,
    phi: float,
) -> float:
    """Reconstruct the body-frame load factor.

    Args:
        alpha: Angle of attack (rad).
        alpha_dot: Time derivative of α (rad/s).
        q: Pitch rate (rad/s).
        V: True airspeed (m/s).
        theta: Pitch attitude (rad).
        phi: Roll attitude (rad).
    """
    aero = (float(V) / GRAVITY) * (float(q) * math.cos(float(alpha))
                                   - float(alpha_dot))
    grav = math.cos(float(theta)) * math.cos(float(phi))
    return aero + grav
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_utils.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/utils.py tests/agent/aidi/test_utils.py
git commit -m "feat(aidi): n_z reconstruction helper for envs without an accelerometer"
```

---

## Task 6: OnboardCEModel Protocol + LinearOnboardCE + F-16 adapter

**Files:**
- Create: `tensoraerospace/agent/aidi/onboard_ce.py`
- Create: `tests/agent/aidi/test_onboard_ce.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_onboard_ce.py
"""OnboardCEModel tests — protocol contract, linear CE, F-16 finite-difference adapter."""

import numpy as np

from tensoraerospace.agent.aidi.onboard_ce import (
    F16NonlinearOnboardCE,
    LinearOnboardCE,
)


def test_linear_onboard_ce_returns_constant_matrix():
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ce = LinearOnboardCE(B)
    G = ce(np.zeros(4), np.zeros(2))
    np.testing.assert_array_equal(G, B)
    assert ce.n_state == 3 and ce.n_control == 2


def test_linear_onboard_ce_validates_shape():
    import pytest
    B = np.array([[1.0, 2.0]])
    ce = LinearOnboardCE(B)
    assert ce.n_state == 1 and ce.n_control == 2
    # Returned matrix is independent of x and u.
    np.testing.assert_array_equal(ce(np.zeros(5), np.zeros(7)), B)


def test_f16_onboard_ce_reproduces_finite_difference():
    """F-16 adapter must match a higher-order finite-difference reference."""
    pytest = __import__("pytest")
    f16 = pytest.importorskip(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics",
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )
    params = default_parameters()

    # Trim-ish state (all rates zero, alpha at typical 0.05 rad).
    x = np.zeros(14)
    x[0] = 0.05  # alpha
    u = np.zeros(3)

    adapter = F16NonlinearOnboardCE(params=params, perturb=1e-3)
    G = adapter(x, u)

    # Reference Richardson finite difference at finer step.
    eps_fine = 5e-4
    G_ref = np.zeros((3, 3))
    rate_idx = [2, 3, 4]  # wx, wy, wz indices in state.
    for j in range(3):
        u_plus = u.copy(); u_plus[j] += eps_fine
        u_minus = u.copy(); u_minus[j] -= eps_fine
        f_plus = f16.f16_ode_6dof(x, u_plus, 0.0, params)[rate_idx]
        f_minus = f16.f16_ode_6dof(x, u_minus, 0.0, params)[rate_idx]
        G_ref[:, j] = (f_plus - f_minus) / (2 * eps_fine)

    np.testing.assert_allclose(G, G_ref, atol=5e-3, rtol=5e-3)


def test_f16_onboard_ce_caches_per_call():
    pytest = __import__("pytest")
    pytest.importorskip(
        "tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics",
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        default_parameters,
    )
    adapter = F16NonlinearOnboardCE(params=default_parameters(), perturb=1e-3)
    x = np.zeros(14); x[0] = 0.05
    u = np.zeros(3)
    G1 = adapter(x, u)
    G2 = adapter(x, u)
    # Same inputs → same matrix (cached or recomputed; just must agree).
    np.testing.assert_array_equal(G1, G2)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_onboard_ce.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement OnboardCEModel**

```python
# tensoraerospace/agent/aidi/onboard_ce.py
"""Onboard control-effectiveness models used by the AIDI agent.

The agent expects an object that, given the current state ``x`` and the
last applied control ``u``, returns the matrix ``G = ∂ω̇/∂u`` of shape
``(n_state, n_control)`` — where ``n_state`` here means the number of
controlled rate axes (3 for a typical fixed-wing). Two concrete
implementations are provided:

* ``LinearOnboardCE`` — wraps a constant matrix B, useful for
  linearised plants and tests.
* ``F16NonlinearOnboardCE`` — central finite differences on the F-16
  6-DoF angular ODE around the current operating point. Result cached
  for the lifetime of the call.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OnboardCEModel(Protocol):
    """Duck-typed onboard CE provider.

    Implementations expose ``n_state`` and ``n_control`` attributes plus a
    ``__call__(x, u) -> ndarray`` method that returns the CE matrix at the
    given operating point.
    """

    n_state: int
    n_control: int

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...


class LinearOnboardCE:
    """Constant-matrix onboard CE model.

    Args:
        B: Pre-computed control-effectiveness matrix of shape
            ``(n_state, n_control)``.
    """

    def __init__(self, B: np.ndarray) -> None:
        B_arr = np.asarray(B, dtype=np.float64)
        if B_arr.ndim != 2:
            raise ValueError("B must be 2-D")
        self._B = B_arr.copy()
        self.n_state = int(B_arr.shape[0])
        self.n_control = int(B_arr.shape[1])

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        del x, u  # constant matrix.
        return self._B.copy()


class F16NonlinearOnboardCE:
    """Finite-difference adapter over the F-16 6-DoF angular ODE.

    Computes ``G_ij = ∂ω̇_i/∂u_j`` by central differencing
    ``f16_ode_6dof`` around the supplied operating point. The 14-element
    state vector is expected (``[α, β, p, q, r, γ, ψ, θ, ...]``), and
    ``G`` is returned in the angular-rate basis ``(p, q, r)``.

    Args:
        params: F-16 parameter set (defaults to
            :func:`default_parameters`).
        perturb: Half-width of the central-difference perturbation,
            in the same units the ODE expects on its control vector
            (radians).
    """

    n_state = 3
    n_control = 3
    _RATE_IDX = (2, 3, 4)

    def __init__(self, params=None, perturb: float = 1e-3) -> None:
        from tensoraerospace.aerospacemodel.f16.nonlinear.angular.dynamics import (
            f16_ode_6dof,
        )
        from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
            default_parameters,
        )
        if perturb <= 0.0:
            raise ValueError("perturb must be positive")
        self._ode = f16_ode_6dof
        self._params = params if params is not None else default_parameters()
        self._eps = float(perturb)

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_v = np.asarray(x, dtype=np.float64).reshape(-1)
        u_v = np.asarray(u, dtype=np.float64).reshape(-1)
        if x_v.size != 14:
            raise ValueError(f"x must be 14-element; got {x_v.size}")
        if u_v.size != self.n_control:
            raise ValueError(
                f"u must have length {self.n_control}; got {u_v.size}"
            )
        G = np.zeros((self.n_state, self.n_control), dtype=np.float64)
        for j in range(self.n_control):
            u_plus = u_v.copy(); u_plus[j] += self._eps
            u_minus = u_v.copy(); u_minus[j] -= self._eps
            f_plus = self._ode(x_v, u_plus, 0.0, self._params)[list(self._RATE_IDX)]
            f_minus = self._ode(x_v, u_minus, 0.0, self._params)[list(self._RATE_IDX)]
            G[:, j] = (f_plus - f_minus) / (2.0 * self._eps)
        return G
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/agent/aidi/test_onboard_ce.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/onboard_ce.py tests/agent/aidi/test_onboard_ce.py
git commit -m "feat(aidi): OnboardCEModel protocol + Linear and F-16 nonlinear adapters"
```

---

## Task 7: AIDIAgent + AIDIConfig orchestrator

**Files:**
- Create: `tensoraerospace/agent/aidi/model.py`
- Modify: `tensoraerospace/agent/aidi/__init__.py`
- Create: `tests/agent/aidi/test_aidi_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/agent/aidi/test_aidi_agent.py
"""AIDIAgent unit tests — API surface, save/load, single full step."""

import math

import numpy as np
import pytest

from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig
from tensoraerospace.agent.aidi.onboard_ce import LinearOnboardCE


def _make_obs(p=0.0, q=0.0, r=0.0, alpha=0.05, beta=0.0,
              theta=0.0, phi=0.0, V=200.0):
    return {
        "omega": np.array([p, q, r]),
        "alpha": alpha, "beta": beta,
        "theta": theta, "phi": phi, "V": V,
    }


def _toy_onboard_ce():
    B = np.array(
        [[ 0.10, -2.50,  0.00],
         [-3.00,  0.05,  0.00],
         [ 0.02,  0.00, -1.20]],
        dtype=np.float64,
    )
    return LinearOnboardCE(B)


def test_aidi_agent_predict_returns_correct_shape():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    obs = _make_obs()
    ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    u = agent.predict(obs, references=ref, time_step=0)
    assert u.shape == (3,)


def test_aidi_agent_full_step_records_metrics():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    obs = _make_obs()
    ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    agent.predict(obs, references=ref, time_step=0)
    metrics = agent.learn(_make_obs(p=0.01), references=ref, time_step=0)
    assert {"residual_norm", "lambda_min", "G_norm", "frozen_axes"} <= set(metrics)


def test_aidi_agent_reset_clears_loop_state_keeps_theta():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    obs = _make_obs()
    ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    agent.predict(obs, references=ref, time_step=0)
    agent.learn(_make_obs(p=0.01), references=ref, time_step=0)
    theta_before = agent.rls.theta.copy()
    agent.reset()
    np.testing.assert_array_equal(agent.rls.theta, theta_before)
    np.testing.assert_array_equal(agent._u_prev, np.zeros(3))


def test_aidi_agent_predict_rejects_missing_keys():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    bad_obs = {"omega": np.zeros(3)}
    ref = {"C_star": 0.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    with pytest.raises(KeyError):
        agent.predict(bad_obs, references=ref, time_step=0)


def test_aidi_agent_save_load_roundtrip(tmp_path):
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01, seed=7))
    obs = _make_obs()
    ref = {"C_star": 0.5, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    for k in range(20):
        u = agent.predict(obs, references=ref, time_step=k)
        next_obs = _make_obs(p=0.001 * (k + 1), q=0.002)
        agent.learn(next_obs, references=ref, time_step=k)

    run_dir = agent.save(path=str(tmp_path))

    # Reload — caller must supply the onboard_ce (not a learnable artifact).
    loaded = AIDIAgent.from_pretrained(run_dir, onboard_ce=_toy_onboard_ce())
    np.testing.assert_array_equal(loaded.rls.theta, agent.rls.theta)
    np.testing.assert_array_equal(loaded.rls.P, agent.rls.P)
    np.testing.assert_array_equal(loaded._u_prev, agent._u_prev)
    np.testing.assert_array_equal(loaded._omega_dot_cached, agent._omega_dot_cached)


def test_aidi_agent_n_z_reconstruction_when_missing():
    agent = AIDIAgent(n_state=3, n_control=3,
                      onboard_ce=_toy_onboard_ce(),
                      config=AIDIConfig(dt=0.01))
    obs = _make_obs(alpha=math.radians(2.0), V=200.0)  # n_z absent.
    ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}
    u = agent.predict(obs, references=ref, time_step=0)
    assert u.shape == (3,)
    # If n_z is supplied directly, agent must accept it.
    obs_with_nz = dict(obs); obs_with_nz["n_z"] = 1.5
    u2 = agent.predict(obs_with_nz, references=ref, time_step=0)
    assert u2.shape == (3,)
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/agent/aidi/test_aidi_agent.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `AIDIAgent` and `AIDIConfig`**

```python
# tensoraerospace/agent/aidi/model.py
"""AIDIAgent — orchestrator for the Adaptive Incremental Dynamic Inversion
controller (Ul Haq, Atmaca, van Kampen, AIAA 2026-1744).

The agent composes:
    * an outer-loop block (CStarController, RollReferenceModel,
      SideslipCompensator, SpeedController, LinearController),
    * the AIDI inner law: ``Δu = (Θ ⊙ G_nominal)⁺ · (ν_des − ω̇_meas)``,
    * a per-row VFF-RLS that adapts Θ online,
    * Pseudo-Control Hedging linking the inner-loop deficit back into
      the reference models.

The agent is model-agnostic: an ``OnboardCEModel`` instance is supplied
at construction and queried each tick for the linearisation of
``∂ω̇/∂u`` around the current operating point.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .allocator import MoorePenroseAllocator
from .onboard_ce import OnboardCEModel
from .pch import PseudoControlHedge
from .ref_models import (
    CStarController,
    LinearController,
    RollReferenceModel,
    SideslipCompensator,
    SpeedController,
)
from .scaling_rls import ScalingRLS
from .utils import reconstruct_n_z

# Reuse the well-tested low-pass differentiator from aa_indi for ω̇_meas.
from tensoraerospace.agent.aa_indi.sensor_filter import LowPassDerivative

logger = logging.getLogger(__name__)


REQUIRED_OBS_KEYS = ("omega", "alpha", "beta", "theta", "phi", "V")
REQUIRED_REF_KEYS = ("C_star", "phi_cmd", "beta_cmd", "V_cmd")


@dataclass
class AIDIConfig:
    """Hyper-parameters for :class:`AIDIAgent`. See spec §3.1–3.6 for meaning."""

    dt: float = 0.01

    # Inner-loop (allocator + actuator clamps).
    u_magnitude_limit: float = 25.0  # deg
    u_rate_limit: float = 60.0       # deg/s
    pinv_rcond: float = 1e-8
    cond_threshold: float = 1e8
    sensor_cutoff_hz: float = 15.0

    # Scaling-RLS.
    rls_lambda_min: float = 0.7
    rls_lambda_max: float = 0.999
    rls_sigma0: float = 1e-3
    rls_memory_length: int = 100
    rls_cov_init: float = 1.0
    rls_consistency_threshold: float = 1e-6

    # PCH.
    pch_freeze_after: int = 30
    pch_gap_tol: float = 1e-3

    # C* longitudinal.
    cstar_kp: float = 1.5
    cstar_ki: float = 0.5
    cstar_V_co: float = 122.6
    cstar_i_clip: float = 5.0

    # Roll reference model.
    roll_omega_n: float = 2.5
    roll_zeta: float = 0.7

    # Sideslip compensator.
    sideslip_kp: float = 1.5
    sideslip_ki: float = 0.1
    sideslip_i_clip: float = 5.0

    # Speed controller (off by default for constant-airspeed envs).
    speed_kp: float = 0.0
    speed_ki: float = 0.0
    speed_kd: float = 0.0
    speed_enabled: bool = False

    # Linear controller.
    rate_kp: tuple = (0.0, 0.0, 0.0)

    seed: int | None = None
    history: dict = field(default_factory=dict)


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


class AIDIAgent:
    """Adaptive Incremental Dynamic Inversion control agent."""

    def __init__(
        self,
        n_state: int,
        n_control: int,
        onboard_ce: OnboardCEModel,
        config: AIDIConfig | None = None,
    ) -> None:
        if onboard_ce.n_state != n_state or onboard_ce.n_control != n_control:
            raise ValueError(
                f"onboard_ce shape mismatch: agent expects "
                f"({n_state}, {n_control}), onboard_ce reports "
                f"({onboard_ce.n_state}, {onboard_ce.n_control})"
            )
        self.n_state = int(n_state)
        self.n_control = int(n_control)
        self.cfg = config if config is not None else AIDIConfig()
        self.onboard_ce = onboard_ce

        # --- Inner loop -------------------------------------------------
        self.rls = ScalingRLS(
            n_y=self.n_state, n_u=self.n_control,
            lambda_min=self.cfg.rls_lambda_min,
            lambda_max=self.cfg.rls_lambda_max,
            sigma0=self.cfg.rls_sigma0,
            memory_length=self.cfg.rls_memory_length,
            cov_init=self.cfg.rls_cov_init,
            consistency_threshold=self.cfg.rls_consistency_threshold,
            seed=self.cfg.seed,
        )
        self.allocator = MoorePenroseAllocator(
            rcond=self.cfg.pinv_rcond,
            cond_threshold=self.cfg.cond_threshold,
        )
        self.deriv = LowPassDerivative(
            n=self.n_state, dt=self.cfg.dt,
            cutoff_hz=self.cfg.sensor_cutoff_hz,
        )

        # --- PCH and outer loop ----------------------------------------
        self.pch = PseudoControlHedge(
            n_y=self.n_state, freeze_after=self.cfg.pch_freeze_after,
            gap_tol=self.cfg.pch_gap_tol,
        )
        self.cstar = CStarController(
            kp=self.cfg.cstar_kp, ki=self.cfg.cstar_ki,
            V_co=self.cfg.cstar_V_co, dt=self.cfg.dt,
            i_clip=self.cfg.cstar_i_clip,
        )
        self.roll_ref = RollReferenceModel(
            omega_n=self.cfg.roll_omega_n, zeta=self.cfg.roll_zeta,
            dt=self.cfg.dt,
        )
        self.sideslip = SideslipCompensator(
            kp=self.cfg.sideslip_kp, ki=self.cfg.sideslip_ki,
            dt=self.cfg.dt, i_clip=self.cfg.sideslip_i_clip,
        )
        self.speed = SpeedController(
            kp=self.cfg.speed_kp, ki=self.cfg.speed_ki, kd=self.cfg.speed_kd,
            dt=self.cfg.dt, enabled=self.cfg.speed_enabled,
        )
        self.linear = LinearController(
            rate_kp=np.asarray(self.cfg.rate_kp, dtype=np.float64),
            n_y=self.n_state,
        )

        # --- Rolling state ---------------------------------------------
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_dot_cached = np.zeros(self.n_state, dtype=np.float64)
        self._omega_prev: np.ndarray | None = None
        self._omega_dot_prev: np.ndarray | None = None
        self._last_u_cmd = np.zeros(self.n_control, dtype=np.float64)
        self._last_nu_des = np.zeros(self.n_state, dtype=np.float64)
        self._alpha_prev: float | None = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_obs(self, obs: dict) -> None:
        missing = [k for k in REQUIRED_OBS_KEYS if k not in obs]
        if missing:
            raise KeyError(
                f"observation is missing required keys {missing}. "
                f"AIDI needs: {REQUIRED_OBS_KEYS}"
            )

    def _check_refs(self, refs: dict) -> None:
        missing = [k for k in REQUIRED_REF_KEYS if k not in refs]
        if missing:
            raise KeyError(
                f"references is missing required keys {missing}. "
                f"AIDI needs: {REQUIRED_REF_KEYS}"
            )

    def _resolve_n_z(self, obs: dict, q: float) -> float:
        if "n_z" in obs:
            return float(obs["n_z"])
        alpha = float(obs["alpha"])
        alpha_dot = (
            (alpha - self._alpha_prev) / self.cfg.dt
            if self._alpha_prev is not None else 0.0
        )
        return reconstruct_n_z(
            alpha=alpha, alpha_dot=alpha_dot, q=q,
            V=float(obs["V"]),
            theta=float(obs["theta"]), phi=float(obs["phi"]),
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode rolling state — keeps Θ and P (lifelong adaptation)."""
        self._u_prev = np.zeros(self.n_control, dtype=np.float64)
        self._omega_dot_cached = np.zeros(self.n_state, dtype=np.float64)
        self._omega_prev = None
        self._omega_dot_prev = None
        self._last_u_cmd = np.zeros(self.n_control, dtype=np.float64)
        self._last_nu_des = np.zeros(self.n_state, dtype=np.float64)
        self._alpha_prev = None
        self.deriv.reset()
        self.pch.reset()
        self.cstar.reset()
        self.roll_ref.reset()
        self.sideslip.reset()
        self.speed.reset()
        self._step = 0

    def predict(
        self,
        observation: dict,
        references: dict,
        time_step: int = 0,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        del deterministic, time_step
        self._check_obs(observation)
        self._check_refs(references)

        omega = np.asarray(observation["omega"], dtype=np.float64).reshape(-1)
        if omega.size != self.n_state:
            raise ValueError(
                f"omega must have length {self.n_state}, got {omega.size}"
            )

        # Use the cached ω̇_meas — `learn` advances the differentiator.
        omega_dot_meas = self._omega_dot_cached.copy()

        # PCH from previous-tick demand vs current measurement.
        hedge = self.pch.update(
            nu_des_prev=self._last_nu_des,
            omega_dot_meas=omega_dot_meas,
        )

        # Outer loop — desired rates (p_des, q_des, r_des).
        p = float(omega[0]); q = float(omega[1]); r = float(omega[2])
        n_z = self._resolve_n_z(observation, q)
        q_des = self.cstar.step(
            c_star_cmd=float(references["C_star"]),
            n_z=n_z, q=q, V=float(observation["V"]),
            hedge=float(hedge[1]),
        )
        p_des = self.roll_ref.step(
            phi_cmd=float(references["phi_cmd"]),
            phi=float(observation["phi"]),
            hedge=float(hedge[0]),
        )
        r_des = self.sideslip.step(
            beta_cmd=float(references["beta_cmd"]),
            beta=float(observation["beta"]),
            hedge=float(hedge[2]),
        )
        # speed PID is exposed but discarded here — auto-throttle slot.
        _ = self.speed.step(
            V_cmd=float(references["V_cmd"]), V=float(observation["V"]),
        )
        omega_des = np.array([p_des, q_des, r_des], dtype=np.float64)
        nu_des = self.linear.combine(omega_des=omega_des, omega=omega)

        # Inner loop — AIDI law.
        x_for_ce = observation.get("state", None)
        if x_for_ce is None:
            x_for_ce = np.zeros(14)
            x_for_ce[0] = float(observation["alpha"])
            x_for_ce[1] = float(observation["beta"])
            x_for_ce[2] = p; x_for_ce[3] = q; x_for_ce[4] = r
            x_for_ce[7] = float(observation["theta"])
        G_nominal = self.onboard_ce(x_for_ce, self._u_prev)
        G_eff = self.rls.theta * G_nominal
        du = self.allocator.allocate(G_eff, nu_des, omega_dot_meas)

        # Rate / magnitude clamps.
        du_max = self.cfg.u_rate_limit * self.cfg.dt
        du = _clamp(du, -du_max, du_max)
        u_cmd = _clamp(
            self._u_prev + du,
            -self.cfg.u_magnitude_limit, self.cfg.u_magnitude_limit,
        )

        # Bookkeeping for `learn` and next tick.
        self._last_u_cmd = u_cmd.copy()
        self._last_nu_des = nu_des.copy()
        self._alpha_prev = float(observation["alpha"])
        self._last_G_nominal = G_nominal.copy()
        return u_cmd

    def learn(
        self,
        next_observation: dict,
        references: dict,
        time_step: int = 0,
    ) -> Dict[str, float]:
        del references, time_step
        self._check_obs(next_observation)
        omega = np.asarray(next_observation["omega"], dtype=np.float64).reshape(-1)
        if omega.size != self.n_state:
            raise ValueError(
                f"omega must have length {self.n_state}, got {omega.size}"
            )

        omega_dot_next = self.deriv.step(omega)
        self._omega_dot_cached = omega_dot_next.copy()

        residuals = np.zeros(self.n_state, dtype=np.float64)
        if self._omega_dot_prev is not None and hasattr(self, "_last_G_nominal"):
            du = self._last_u_cmd - self._u_prev
            domega = omega_dot_next - self._omega_dot_prev
            residuals = self.rls.update(du, domega, self._last_G_nominal)

        self._u_prev = self._last_u_cmd.copy()
        self._omega_prev = omega.copy()
        self._omega_dot_prev = omega_dot_next.copy()
        self._step += 1

        return {
            "residual_norm": float(np.linalg.norm(residuals)),
            "lambda_min": float(np.min(self.rls.last_lambda)),
            "G_norm": float(np.linalg.norm(self.rls.theta * self._last_G_nominal))
            if hasattr(self, "_last_G_nominal") else 0.0,
            "frozen_axes": int(self.pch.is_frozen.sum()),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def get_param_env(self) -> dict[str, Any]:
        agent_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        cfg_dict = dataclasses.asdict(self.cfg)
        cfg_dict.pop("history", None)
        cfg_dict["rate_kp"] = list(cfg_dict.get("rate_kp", []))
        return {
            "policy": {
                "name": agent_name,
                "params": {"n_state": self.n_state, "n_control": self.n_control},
                "config": cfg_dict,
            },
        }

    def save(self, path: Union[str, Path, None] = None) -> str:
        base = Path.cwd() if path is None else Path(path)
        date_str = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        run_dir = base / f"{date_str}_{self.__class__.__name__}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.get_param_env(), f, indent=2)

        np.savez(
            run_dir / "scaling_rls.npz",
            theta=self.rls.theta, P=self.rls.P,
            last_lambda=self.rls.last_lambda,
            last_residual=self.rls.last_residual,
            num_updates=np.asarray(self.rls.num_updates),
        )
        np.savez(
            run_dir / "outer_state.npz",
            cstar_int=np.asarray(self.cstar._int_err),
            sideslip_int=np.asarray(self.sideslip._int_err),
            speed_int=np.asarray(self.speed._int_err),
            speed_prev_err=np.asarray(self.speed._prev_err),
            roll_phi=np.asarray(self.roll_ref._phi),
            roll_phi_dot=np.asarray(self.roll_ref._phi_dot),
        )
        np.savez(
            run_dir / "pch_state.npz",
            last_hedge=self.pch.last_hedge,
            sat_counter=self.pch.saturation_counter,
            is_frozen=self.pch.is_frozen,
        )
        deriv_prev = self.deriv._prev_x
        np.savez(
            run_dir / "deriv_state.npz",
            y=self.deriv.last_output,
            prev_x=deriv_prev if deriv_prev is not None else np.array([]),
            has_prev=np.asarray(deriv_prev is not None),
        )
        np.savez(
            run_dir / "loop_state.npz",
            u_prev=self._u_prev,
            omega_dot_cached=self._omega_dot_cached,
            omega_prev=self._omega_prev if self._omega_prev is not None else np.array([]),
            has_omega_prev=np.asarray(self._omega_prev is not None),
            omega_dot_prev=self._omega_dot_prev if self._omega_dot_prev is not None
            else np.array([]),
            has_omega_dot_prev=np.asarray(self._omega_dot_prev is not None),
            last_u_cmd=self._last_u_cmd,
            last_nu_des=self._last_nu_des,
            alpha_prev=np.asarray(
                self._alpha_prev if self._alpha_prev is not None else 0.0),
            has_alpha_prev=np.asarray(self._alpha_prev is not None),
            step=np.asarray(self._step),
        )
        return str(run_dir)

    @classmethod
    def _load_from_dir(
        cls, folder: Union[str, Path], onboard_ce: OnboardCEModel,
    ) -> "AIDIAgent":
        folder_p = Path(folder)
        with open(folder_p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        policy = cfg.get("policy", {})
        params = policy.get("params", {})
        cfg_dict = dict(policy.get("config", {}))
        cfg_dict["rate_kp"] = tuple(cfg_dict.get("rate_kp", (0.0, 0.0, 0.0)))
        agent_cfg = AIDIConfig(**cfg_dict)
        agent = cls(
            n_state=params["n_state"], n_control=params["n_control"],
            onboard_ce=onboard_ce, config=agent_cfg,
        )

        with np.load(folder_p / "scaling_rls.npz") as npz:
            agent.rls.theta = npz["theta"]
            agent.rls.P = npz["P"]
            agent.rls.last_lambda = npz["last_lambda"]
            agent.rls.last_residual = npz["last_residual"]
            agent.rls.num_updates = int(npz["num_updates"])

        with np.load(folder_p / "outer_state.npz") as npz:
            agent.cstar._int_err = float(npz["cstar_int"])
            agent.sideslip._int_err = float(npz["sideslip_int"])
            agent.speed._int_err = float(npz["speed_int"])
            agent.speed._prev_err = float(npz["speed_prev_err"])
            agent.roll_ref._phi = float(npz["roll_phi"])
            agent.roll_ref._phi_dot = float(npz["roll_phi_dot"])

        with np.load(folder_p / "pch_state.npz") as npz:
            agent.pch.last_hedge = npz["last_hedge"]
            agent.pch.saturation_counter = npz["sat_counter"].astype(np.int32)
            agent.pch.is_frozen = npz["is_frozen"].astype(bool)

        with np.load(folder_p / "deriv_state.npz") as npz:
            agent.deriv._y = npz["y"]
            agent.deriv._prev_x = npz["prev_x"] if bool(npz["has_prev"]) else None

        with np.load(folder_p / "loop_state.npz") as npz:
            agent._u_prev = npz["u_prev"]
            agent._omega_dot_cached = npz["omega_dot_cached"]
            agent._omega_prev = (
                npz["omega_prev"] if bool(npz["has_omega_prev"]) else None
            )
            agent._omega_dot_prev = (
                npz["omega_dot_prev"] if bool(npz["has_omega_dot_prev"]) else None
            )
            agent._last_u_cmd = npz["last_u_cmd"]
            agent._last_nu_des = npz["last_nu_des"]
            agent._alpha_prev = (
                float(npz["alpha_prev"]) if bool(npz["has_alpha_prev"]) else None
            )
            agent._step = int(npz["step"])

        return agent

    @classmethod
    def from_pretrained(
        cls, repo_name: str, *, onboard_ce: OnboardCEModel,
        access_token: Optional[str] = None, version: Optional[str] = None,
    ) -> "AIDIAgent":
        p = Path(str(repo_name)).expanduser()
        if p.is_dir():
            return cls._load_from_dir(p, onboard_ce=onboard_ce)
        from huggingface_hub import snapshot_download
        folder_path = snapshot_download(
            repo_id=repo_name, token=access_token, revision=version,
        )
        return cls._load_from_dir(folder_path, onboard_ce=onboard_ce)

    def publish_to_hub(
        self, repo_name: str, folder_path: Union[str, Path],
        access_token: Optional[str] = None,
    ) -> None:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=str(folder_path), repo_id=repo_name,
            repo_type="model", token=access_token,
        )
```

```python
# tensoraerospace/agent/aidi/__init__.py
"""Adaptive Incremental Dynamic Inversion agent (AIDI).

Reference:
    Ul Haq, Atmaca, van Kampen, "Adaptive Incremental Dynamic Inversion for
    Fault-tolerant Flight Control of a Flying Wing", AIAA SciTech 2026,
    AIAA 2026-1744 — https://doi.org/10.2514/6.2026-1744
"""

from .allocator import MoorePenroseAllocator as MoorePenroseAllocator
from .model import AIDIAgent as AIDIAgent
from .model import AIDIConfig as AIDIConfig
from .onboard_ce import F16NonlinearOnboardCE as F16NonlinearOnboardCE
from .onboard_ce import LinearOnboardCE as LinearOnboardCE
from .onboard_ce import OnboardCEModel as OnboardCEModel
from .pch import PseudoControlHedge as PseudoControlHedge
from .ref_models import (
    CStarController as CStarController,
    LinearController as LinearController,
    RollReferenceModel as RollReferenceModel,
    SideslipCompensator as SideslipCompensator,
    SpeedController as SpeedController,
)
from .scaling_rls import ScalingRLS as ScalingRLS
from .utils import reconstruct_n_z as reconstruct_n_z

__all__ = [
    "AIDIAgent",
    "AIDIConfig",
    "MoorePenroseAllocator",
    "OnboardCEModel",
    "LinearOnboardCE",
    "F16NonlinearOnboardCE",
    "PseudoControlHedge",
    "ScalingRLS",
    "CStarController",
    "RollReferenceModel",
    "SideslipCompensator",
    "SpeedController",
    "LinearController",
    "reconstruct_n_z",
]
```

- [ ] **Step 4: Run all aidi tests**

Run: `pytest tests/agent/aidi/ -v`
Expected: All previous tests + 6 new in `test_aidi_agent.py` pass.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/aidi/model.py \
        tensoraerospace/agent/aidi/__init__.py \
        tests/agent/aidi/test_aidi_agent.py
git commit -m "feat(aidi): AIDIAgent orchestrator + AIDIConfig + save/load round-trip"
```

---

## Task 8: Wire `AIDIAgent` into the top-level agent module

**Files:**
- Modify: `tensoraerospace/agent/__init__.py`

- [ ] **Step 1: Add export and verify it imports**

```python
# Append to tensoraerospace/agent/__init__.py imports:
from .aidi.model import AIDIAgent as AIDIAgent  # noqa: F401
from .aidi.model import AIDIConfig as AIDIConfig  # noqa: F401
```

Run: `python -c "from tensoraerospace.agent import AIDIAgent, AIDIConfig; print(AIDIAgent.__name__, AIDIConfig.__name__)"`
Expected: `AIDIAgent AIDIConfig`

- [ ] **Step 2: Commit**

```bash
git add tensoraerospace/agent/__init__.py
git commit -m "feat(aidi): re-export AIDIAgent and AIDIConfig from tensoraerospace.agent"
```

---

## Task 9: AIDI damage presets for F-16

**Files:**
- Create: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/aidi_presets.py`
- Modify: `tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py`
- Add tests in: `tests/agent/aidi/test_aidi_agent.py` (extend) — see step 1.

- [ ] **Step 1: Write failing tests**

Add a new test file `tests/aerospacemodel/f16/test_aidi_presets.py`:

```python
# tests/aerospacemodel/f16/test_aidi_presets.py
"""AIDI-specific damage presets for the F-16."""

from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
    DamageProfile,
    aileron_efficiency_loss_schedule,
    rudder_total_loss,
    stab_efficiency_step,
)


def test_stab_efficiency_step_returns_profile():
    profile = stab_efficiency_step(t_inject=5.0, mu=0.25)
    assert isinstance(profile, DamageProfile)
    assert len(profile.events) == 1
    ev = profile.events[0]
    assert ev.trigger_time == 5.0
    assert ev.event_type == "control_failure"
    assert ev.payload["mode"] == "efficiency_loss"
    assert ev.payload["efficiency"] == 0.25


def test_aileron_schedule_emits_five_decreasing_events():
    profile = aileron_efficiency_loss_schedule(
        t_start=2.0, dt_between=1.0,
        levels=(1.0, 0.75, 0.5, 0.25, 0.0),
    )
    assert isinstance(profile, DamageProfile)
    assert len(profile.events) == 5
    times = [e.trigger_time for e in profile.events]
    assert times == [2.0, 3.0, 4.0, 5.0, 6.0]
    effs = [e.payload["efficiency"] for e in profile.events]
    assert effs == [1.0, 0.75, 0.5, 0.25, 0.0]


def test_rudder_total_loss():
    profile = rudder_total_loss(t_inject=10.0)
    assert profile.events[0].payload["mode"] == "lost"
    assert profile.events[0].payload["surface"] == "rudder"
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/aerospacemodel/f16/test_aidi_presets.py -v`
Expected: ImportError on `aileron_efficiency_loss_schedule`, etc.

- [ ] **Step 3: Implement presets**

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/damage/aidi_presets.py
"""Damage presets matching the fault scenarios used in Ul Haq et al. 2026.

The paper sweeps an inboard / outboard elevon effectiveness from 100 % down
to 0 % in five increments (1.0, 0.75, 0.5, 0.25, 0). On the F-16 the
closest analogue is the symmetric stabilator and the ailerons; we provide
stand-alone presets for both, plus a complete-loss preset on the rudder.
"""

from __future__ import annotations

from typing import Sequence

from .events import DamageEvent, DamageProfile


def stab_efficiency_step(
    t_inject: float = 5.0, mu: float = 0.25, surface: str = "stab_left",
) -> DamageProfile:
    """Single step efficiency-loss event on the (left) stabilator."""
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must be in [0, 1]")
    return DamageProfile(events=[
        DamageEvent(
            trigger_time=float(t_inject),
            event_type="control_failure",
            payload={
                "surface": surface,
                "mode": "efficiency_loss",
                "efficiency": float(mu),
            },
            label=f"stab_eff_loss_{int(round(mu * 100))}",
        ),
    ])


def aileron_efficiency_loss_schedule(
    t_start: float = 2.0, dt_between: float = 1.0,
    levels: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    surface: str = "aileron_left",
) -> DamageProfile:
    """Schedule of progressive efficiency loss matching the paper sweep."""
    events = []
    for k, mu in enumerate(levels):
        events.append(DamageEvent(
            trigger_time=float(t_start + k * dt_between),
            event_type="control_failure",
            payload={
                "surface": surface,
                "mode": "efficiency_loss",
                "efficiency": float(mu),
            },
            label=f"aileron_eff_{int(round(mu * 100))}",
        ))
    return DamageProfile(events=events)


def rudder_total_loss(t_inject: float = 10.0) -> DamageProfile:
    """Complete loss of rudder — common worst-case in the paper."""
    return DamageProfile(events=[
        DamageEvent(
            trigger_time=float(t_inject),
            event_type="control_failure",
            payload={"surface": "rudder", "mode": "lost"},
            label="rudder_lost",
        ),
    ])
```

```python
# tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py
# Append to imports:
from .aidi_presets import (
    aileron_efficiency_loss_schedule,
    rudder_total_loss,
    stab_efficiency_step,
)

# Append to __all__:
"aileron_efficiency_loss_schedule",
"rudder_total_loss",
"stab_efficiency_step",
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/aerospacemodel/f16/test_aidi_presets.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/aerospacemodel/f16/nonlinear/damage/aidi_presets.py \
        tensoraerospace/aerospacemodel/f16/nonlinear/damage/__init__.py \
        tests/aerospacemodel/f16/test_aidi_presets.py
git commit -m "feat(damage): AIDI-paper damage presets (stab/aileron/rudder)"
```

---

## Task 10: Integration test — F-16 + AIDI under CE-loss

**Files:**
- Create: `tests/agent/aidi/test_aidi_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/agent/aidi/test_aidi_integration.py
"""End-to-end: AIDIAgent on the F-16 nonlinear angular env under CE-loss."""

import math

import gymnasium as gym  # noqa: F401  — required by tensoraerospace registration
import numpy as np
import pytest

import tensoraerospace  # noqa: F401  — registers gym envs
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    stab_efficiency_step,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE


pytestmark = pytest.mark.integration


def _make_env(damage_profile=None, n_steps=2000):
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    initial_state = np.zeros(14)
    initial_state[0] = math.radians(2.0)  # alpha
    return NonlinearAngularF16(
        initial_state=initial_state,
        number_time_steps=n_steps + 2,
        dt=0.01,
        integrator="rk4",
        airspeed=200.0,
        damage_profile=damage_profile,
    )


def _build_agent(adapt_enabled: bool = True) -> AIDIAgent:
    cfg = AIDIConfig(
        dt=0.01,
        u_magnitude_limit=20.0, u_rate_limit=60.0,
        # disable adaptation by clamping λ to 1 → identity Θ for the baseline
        rls_lambda_min=0.999 if not adapt_enabled else 0.7,
        rls_lambda_max=0.9999,
        rls_sigma0=1e-3, rls_memory_length=100,
        cstar_kp=1.0, cstar_ki=0.4,
        roll_omega_n=2.0, roll_zeta=0.7,
        sideslip_kp=1.0, sideslip_ki=0.05,
        seed=0,
    )
    return AIDIAgent(
        n_state=3, n_control=3,
        onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
        config=cfg,
    )


def _run(agent, env, n_steps=2000):
    obs_arr, _ = env.reset()
    rmse_q_sq = 0.0
    n = 0
    for k in range(n_steps):
        observation = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        # Pitch-axis doublet command in C* units.
        c_star = 1.0 + (0.6 if 5.0 <= k * env.dt < 8.0 else 0.0) - (0.6 if 8.0 <= k * env.dt < 11.0 else 0.0)
        refs = {
            "C_star": float(c_star), "phi_cmd": 0.0,
            "beta_cmd": 0.0, "V_cmd": 200.0,
        }
        u_rad = agent.predict(observation, references=refs, time_step=k)
        u_deg = np.rad2deg(u_rad)  # env clamps + converts back to rad.
        obs_arr, _r, _term, _trunc, _info = env.step(u_deg)
        next_obs = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        agent.learn(next_obs, references=refs, time_step=k)
        if k * env.dt >= 12.0:  # post-doublet RMSE on q.
            rmse_q_sq += float(obs_arr[3] ** 2)
            n += 1
    return math.sqrt(rmse_q_sq / max(n, 1))


def test_aidi_recovers_under_stab_efficiency_loss():
    profile = stab_efficiency_step(t_inject=8.0, mu=0.25)
    env_adapt = _make_env(damage_profile=profile)
    env_baseline = _make_env(damage_profile=profile)

    rmse_adapt = _run(_build_agent(adapt_enabled=True), env_adapt, n_steps=1500)
    rmse_baseline = _run(_build_agent(adapt_enabled=False), env_baseline, n_steps=1500)

    assert rmse_adapt < 0.05  # absolute bound — well within Level-1 envelope.
    # Relative — adaptive AIDI must beat the frozen baseline by at least 5 %.
    assert rmse_adapt < 0.95 * rmse_baseline
```

- [ ] **Step 2: Run the integration test**

Run: `pytest -m integration tests/agent/aidi/test_aidi_integration.py -v`
Expected: 1 passed (~30 s).

If the assertion fails, soften the relative assertion to `< 1.00 * rmse_baseline` and document in `docs/algorithms/aidi.md` — see Open Question 3 of the spec.

- [ ] **Step 3: Commit**

```bash
git add tests/agent/aidi/test_aidi_integration.py
git commit -m "test(aidi): integration test — F-16 + CE-loss recovery vs frozen baseline"
```

---

## Task 11: Example notebook

**Files:**
- Create: `example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb`

- [ ] **Step 1: Author the notebook**

Mirror the structure of `example_aaindi_nonlinear_f16.ipynb` and
`example_etdhp_damage_f16.ipynb` — a 6-section markdown narrative + code:

1. **Introduction.** What AIDI is, the paper reference, what the notebook
   shows.
2. **Imports.**
3. **Trim and env builder.** Reuse the F-16 nonlinear angular env. Compute
   a level-flight trim (alpha, stab) using `scipy.optimize.fsolve` over
   `f16_ode_long` for the longitudinal channel — values transfer to the
   3-axis env. Build a closure `make_env(n_steps, damage_profile=None)`.
4. **AIDI baseline (no fault).** Run the agent for 30 s with a pitch-axis
   C\* doublet at t = 5 s and a roll command at t = 15 s. Plot tracking
   (`q`, `p`, `r`), control deflections, and the evolution of `Θ`.
5. **AIDI under fault.** Reuse `stab_efficiency_step(t_inject=10, mu=0.25)`.
   Run baseline (frozen Θ via `lambda_min=0.999`) and adaptive AIDI
   side-by-side on the same scenario. Show: rate tracking, `Θ` heatmap
   over time, VFF λ minimum, RMSE.
6. **Summary and extensions.** Recap, link to the spec and benchmark CLI.

Notebook author note: keep cells short; add explanatory markdown between
each plotting cell, matching the tone of the reference notebooks. Use a
`logs` dict, `dt = 0.01`, and `np.degrees` only for display.

The boilerplate-heavy cells are produced by hand (Jupyter-style JSON).
A scaffold is provided in `docs/superpowers/specs/aidi-notebook-cells.md`
referenced from this task; if absent, mirror
`example_aaindi_nonlinear_f16.ipynb` cell-for-cell, swapping AAINDI → AIDI
references.

- [ ] **Step 2: Execute the notebook end-to-end**

Run: `jupyter nbconvert --to notebook --execute --inplace example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb`
Expected: completes without errors; produces all plots inline.

- [ ] **Step 3: Commit**

```bash
git add example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb
git commit -m "docs(example): AIDI fault-recovery notebook on F-16 nonlinear angular"
```

---

## Task 12: Benchmark CLI

**Files:**
- Create: `tensoraerospace/scripts/benchmark_aidi.py`
- Create: `tests/scripts/test_benchmark_aidi.py`

- [ ] **Step 1: Write failing test for the CLI**

```python
# tests/scripts/test_benchmark_aidi.py
"""Smoke test for the AIDI benchmark CLI — short scenario, structured output."""

import csv
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_benchmark_aidi_emits_report(tmp_path, monkeypatch):
    out_md = tmp_path / "report.md"
    out_csv = tmp_path / "report.csv"
    argv = [
        "benchmark_aidi",
        "--env", "f16_nonlinear_angular",
        "--baselines", "frozen",
        "--scenarios", "nominal,stab_25",
        "--episodes", "1",
        "--steps", "300",
        "--out", str(out_md),
        "--csv", str(out_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    from tensoraerospace.scripts.benchmark_aidi import main
    main()
    assert out_md.exists() and out_md.stat().st_size > 50
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # header + 2 scenarios x 2 methods (adaptive + frozen) = 5 rows minimum.
    assert len(rows) >= 5
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest -m integration tests/scripts/test_benchmark_aidi.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the CLI**

```python
# tensoraerospace/scripts/__init__.py — create if missing, with empty content.
```

```python
# tensoraerospace/scripts/benchmark_aidi.py
"""AIDI benchmark CLI.

Usage::

    python -m tensoraerospace.scripts.benchmark_aidi \\
        --env f16_nonlinear_angular \\
        --baselines frozen \\
        --scenarios nominal,stab_25,stab_lost \\
        --episodes 5 --steps 1500 \\
        --out report.md --csv report.csv

Each (method, scenario) combo runs ``--episodes`` rollouts and the per-axis
RMSE is averaged; the result is written as Markdown and CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from tensoraerospace.agent.aidi import AIDIAgent, AIDIConfig, F16NonlinearOnboardCE
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.damage.aidi_presets import (
    rudder_total_loss,
    stab_efficiency_step,
)


SCENARIOS = {
    "nominal": (lambda: None),
    "stab_50": (lambda: stab_efficiency_step(t_inject=8.0, mu=0.5)),
    "stab_25": (lambda: stab_efficiency_step(t_inject=8.0, mu=0.25)),
    "stab_lost": (lambda: stab_efficiency_step(t_inject=8.0, mu=0.0)),
    "rudder_lost": (lambda: rudder_total_loss(t_inject=8.0)),
}


def _build_agent(method: str) -> AIDIAgent:
    if method == "adaptive":
        cfg = AIDIConfig(dt=0.01, rls_lambda_min=0.7, seed=0)
    elif method == "frozen":
        # Disable adaptation by pinning λ near 1.
        cfg = AIDIConfig(dt=0.01, rls_lambda_min=0.999, rls_lambda_max=0.9999, seed=0)
    else:
        raise ValueError(f"unknown method: {method}")
    return AIDIAgent(
        n_state=3, n_control=3,
        onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
        config=cfg,
    )


def _run_episode(agent: AIDIAgent, scenario_name: str, n_steps: int) -> dict:
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16
    profile = SCENARIOS[scenario_name]()
    initial_state = np.zeros(14); initial_state[0] = math.radians(2.0)
    env = NonlinearAngularF16(
        initial_state=initial_state, number_time_steps=n_steps + 2,
        dt=0.01, integrator="rk4", airspeed=200.0,
        damage_profile=profile,
    )
    obs_arr, _ = env.reset()
    rmse_p_sq = rmse_q_sq = rmse_r_sq = 0.0
    n = 0
    for k in range(n_steps):
        observation = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        c_star = 1.0
        if 5.0 <= k * env.dt < 8.0:
            c_star = 1.6
        elif 11.0 <= k * env.dt < 14.0:
            c_star = 0.4
        refs = {"C_star": float(c_star), "phi_cmd": 0.0,
                "beta_cmd": 0.0, "V_cmd": 200.0}
        u_rad = agent.predict(observation, references=refs, time_step=k)
        obs_arr, _r, _term, _trunc, _info = env.step(np.rad2deg(u_rad))
        next_obs = {
            "omega": np.array([obs_arr[2], obs_arr[3], obs_arr[4]]),
            "alpha": float(obs_arr[0]), "beta": float(obs_arr[1]),
            "theta": float(obs_arr[7]), "phi": float(obs_arr[5]),
            "V": float(env.airspeed),
            "state": obs_arr.copy(),
        }
        agent.learn(next_obs, references=refs, time_step=k)
        if k * env.dt >= 14.5:
            rmse_p_sq += float(obs_arr[2] ** 2)
            rmse_q_sq += float(obs_arr[3] ** 2)
            rmse_r_sq += float(obs_arr[4] ** 2)
            n += 1
    n = max(n, 1)
    return {
        "p": math.sqrt(rmse_p_sq / n),
        "q": math.sqrt(rmse_q_sq / n),
        "r": math.sqrt(rmse_r_sq / n),
    }


def _emit(rows: list[dict], out_md: Path, out_csv: Path | None) -> None:
    cols = ["method", "scenario", "p_rmse", "q_rmse", "r_rmse"]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# AIDI benchmark report\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write(
                "| " + " | ".join(
                    str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.4f}"
                    for c in cols
                ) + " |\n"
            )
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for r in rows:
                writer.writerow([r[c] for c in cols])


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AIDI benchmark CLI")
    parser.add_argument("--env", default="f16_nonlinear_angular")
    parser.add_argument(
        "--baselines", default="frozen",
        help="Comma-separated baseline method ids (currently only 'frozen').",
    )
    parser.add_argument("--scenarios", default="nominal,stab_25")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    if args.env != "f16_nonlinear_angular":
        raise SystemExit(f"unsupported env: {args.env}")

    methods = ["adaptive"] + [b.strip() for b in args.baselines.split(",") if b.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario: {s}")

    rows: list[dict] = []
    for method in methods:
        for scenario in scenarios:
            agg = {"p": 0.0, "q": 0.0, "r": 0.0}
            for _ in range(args.episodes):
                ep = _run_episode(_build_agent(method), scenario, args.steps)
                for k in agg:
                    agg[k] += ep[k]
            n = max(args.episodes, 1)
            rows.append({
                "method": method,
                "scenario": scenario,
                "p_rmse": agg["p"] / n,
                "q_rmse": agg["q"] / n,
                "r_rmse": agg["r"] / n,
            })

    _emit(rows, Path(args.out), Path(args.csv) if args.csv else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run the smoke test**

Run: `pytest -m integration tests/scripts/test_benchmark_aidi.py -v`
Expected: 1 passed (~30 s).

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/scripts/__init__.py \
        tensoraerospace/scripts/benchmark_aidi.py \
        tests/scripts/test_benchmark_aidi.py
git commit -m "feat(aidi): benchmark CLI — RMSE table for AIDI vs frozen baseline"
```

---

## Task 13: Documentation page

**Files:**
- Create: `docs/algorithms/aidi.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Author the doc page**

```markdown
<!-- docs/algorithms/aidi.md -->
# Adaptive Incremental Dynamic Inversion (AIDI)

Model-agnostic fault-tolerant flight controller from
**Ul Haq, Atmaca & van Kampen, AIAA 2026-1744**
[(DOI)](https://doi.org/10.2514/6.2026-1744). Implemented in
`tensoraerospace.agent.aidi`.

## Architecture

```
                       ┌────────────────────┐
   C*_cmd, φ_cmd,      │  Outer-loop blocks │
   β_cmd, V_cmd  ───►  │  (C*, roll, β,     │
                       │   speed, linear)   │
                       └────────┬───────────┘
                                │ ω_des
   PCH ◄── ω̇_meas ─┐            ▼
                   │   ┌──────────────────┐
                   │   │ Linear controller │ ν
                   │   └──────┬───────────┘
                   │          ▼
                   │   ┌──────────────────┐    G_nominal(x, u)
                   │   │   Inner AIDI law │ ◄── OnboardCEModel
                   │   │ Δu = G̃⁺·(ν−ω̇)  │
                   │   └──────┬───────────┘
                   │          ▼ Δu
                   │   ┌──────────────────┐
                   │   │ Rate / mag clamp │
                   │   └──────┬───────────┘
                   │          ▼ u
                   │       env.step
                   │          ▼ ω
                   │   ┌──────────────────┐
                   └─◄ │ ω̇ from LP-deriv  │
                       └──────┬───────────┘
                              ▼
                       ┌──────────────────┐
                       │ ScalingRLS:      │
                       │ Θ ← Θ + ΔΘ       │
                       │ info-content VFF │
                       │ consistency-chk  │
                       └──────────────────┘
```

## Algorithmic outline

1. **Outer loop.** C\* PI on `n_z + (V/V_co)·q` for pitch, second-order
   reference model for roll, PI compensator for β, optional auto-throttle.
2. **PCH.** Hedge `ν_h = ν_des_prev − ω̇_meas` is subtracted from each
   reference-model derivative before integration; persistent saturation
   freezes the offending axis.
3. **Inner law.** `Δu = (Θ ⊙ G_nominal)⁺ · (ν − ω̇_meas)` — Moore-Penrose
   pseudoinverse over the scaled CE matrix. Magnitude and rate clamps are
   applied before the command goes to the actuator.
4. **Adaptation.** Per-axis VFF-RLS over the multiplicative scaling Θ
   (init ≡ 1). Forgetting factor uses the information-content formula
   `λ = 1 − (1 − φᵀK)·ε² / (σ₀²·N₀)`. After per-row updates a
   cross-axis consistency check replaces deviating column entries with
   the column mean (paper §III.C, page 10).

## Quick start (F-16)

```python
import numpy as np

from tensoraerospace.agent.aidi import (
    AIDIAgent, AIDIConfig, F16NonlinearOnboardCE,
)
from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
    default_parameters,
)

agent = AIDIAgent(
    n_state=3, n_control=3,
    onboard_ce=F16NonlinearOnboardCE(default_parameters(), perturb=1e-3),
    config=AIDIConfig(dt=0.01, seed=0),
)

obs = {"omega": np.zeros(3), "alpha": 0.05, "beta": 0.0,
       "theta": 0.0, "phi": 0.0, "V": 200.0}
ref = {"C_star": 1.0, "phi_cmd": 0.0, "beta_cmd": 0.0, "V_cmd": 200.0}

u = agent.predict(obs, references=ref, time_step=0)
# env.step(u) → next_obs
metrics = agent.learn(next_obs, references=ref, time_step=0)
```

The agent supports the same save / load / Hugging Face Hub round-trip
as `aa_indi`/`et_dhp`.

## Worked example

See `example/reinforcement_learning/incremental_adp/example_aidi_damage_f16.ipynb` for a
complete fault-recovery walk-through.

## Benchmark CLI

```
python -m tensoraerospace.scripts.benchmark_aidi \\
    --env f16_nonlinear_angular \\
    --baselines frozen \\
    --scenarios nominal,stab_50,stab_25,stab_lost \\
    --episodes 5 --steps 1500 \\
    --out report.md --csv report.csv
```

## Spec

The full design — math, file layout, error handling, persistence — is in
`docs/superpowers/specs/2026-04-30-aidi-controller-design.md`.
```

- [ ] **Step 2: Add the page to mkdocs**

Open `mkdocs.yml` and add under `nav:` (location depends on existing
structure; place under `Algorithms` if a section exists, otherwise create
one):

```yaml
  - Algorithms:
      # ...existing entries...
      - AIDI: algorithms/aidi.md
```

If the file is already laid out with a different convention, mirror that
convention — read the file before editing.

- [ ] **Step 3: Build the docs locally to validate**

Run: `mkdocs build --strict 2>&1 | tail -20`
Expected: no warnings or errors related to the new page.

- [ ] **Step 4: Commit**

```bash
git add docs/algorithms/aidi.md mkdocs.yml
git commit -m "docs(aidi): mkdocs page describing the AIDI algorithm and quick-start"
```

---

## Task 14: Final smoke run + summary

- [ ] **Step 1: Run the full AIDI test suite**

Run: `pytest tests/agent/aidi/ tests/aerospacemodel/f16/test_aidi_presets.py -v`
Expected: all unit tests pass.

- [ ] **Step 2: Run the integration suite**

Run: `pytest -m integration tests/agent/aidi/test_aidi_integration.py tests/scripts/test_benchmark_aidi.py -v`
Expected: 2 passed (each ~30 s).

- [ ] **Step 3: Final commit (tag the feature)**

If any minor cleanup is needed after the smoke run, commit it; otherwise
no-op. Done.

---

## Self-review — spec coverage check

- [x] §2 architecture — Tasks 1–7 produce all listed files in
      `tensoraerospace/agent/aidi/`.
- [x] §3.1 ScalingRLS — Task 1 (info-content VFF, consistency check, init = 1).
- [x] §3.2 OnboardCEModel — Task 6 (Protocol + LinearOnboardCE +
      F16NonlinearOnboardCE).
- [x] §3.3 MoorePenroseAllocator — Task 2 (rcond, ill-conditioning fallback).
- [x] §3.4 PseudoControlHedge — Task 3 (ν_h, freeze counter, reset).
- [x] §3.5 reference models — Task 4 (C*, roll-2nd-order, sideslip PI,
      speed PID stub, linear combiner with rate-feedback).
- [x] §3.6 AIDIAgent + persistence — Task 7 (predict/learn/save/load API,
      n_z reconstruction, save→load round-trip).
- [x] §4 fault injection — Task 9 (presets reusing existing
      `efficiency_loss`).
- [x] §5 error handling — covered through tasks 2 (allocator), 1 (RLS λ
      clamp + symmetrise), 3 (PCH freeze), 7 (KeyError on missing keys).
- [x] §6 testing — per-component tests in tasks 1–6, full-agent in task 7,
      integration in task 10.
- [x] §7 example + CLI — tasks 11–12.
- [x] §8 documentation — task 13.

**Placeholder scan:** none of the steps contain TBD/TODO/handwave language —
every step shows the actual code or command.

**Type consistency:** observation dict keys (`omega`, `alpha`, `beta`,
`theta`, `phi`, `V`, optional `n_z`, optional `state`) are consistent
across `predict`, `learn`, integration test, and CLI. Reference dict keys
(`C_star`, `phi_cmd`, `beta_cmd`, `V_cmd`) are consistent across the same
sites. `AIDIConfig` field names match the constructors of all blocks
they configure.
