# Unified TensorBoard Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate every RL agent in TensorAeroSpace to a single canonical TensorBoard metric schema (mandatory minimum + per-algorithm extras), enforced by a strict whitelist in `MetricWriter`, with `global_env_step` as the only X-axis.

**Architecture:** Replace the single-file `tensoraerospace/agent/metrics.py` with a package containing `schema.py` (named constants + `REGISTRY`), `writer.py` (`MetricWriter` with `strict=True` whitelist and a required `env_step` argument), and `contract.py` (mandatory-set assertion helper). Remove the alias map / `normalize_tag()`. Rename every metric tag in every agent's source code to the canonical constant.

**Tech Stack:** Python 3.10+, `torch.utils.tensorboard.SummaryWriter` (existing dep), pytest, `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` (for verification in smoke tests).

**Spec:** `docs/superpowers/specs/2026-04-19-unified-tensorboard-metrics-design.md`

---

## File Structure

**Created:**
- `tensoraerospace/agent/metrics/__init__.py` — public re-exports
- `tensoraerospace/agent/metrics/schema.py` — canonical names + `REGISTRY`
- `tensoraerospace/agent/metrics/writer.py` — `MetricWriter`, fallback writer, factory
- `tensoraerospace/agent/metrics/contract.py` — `MANDATORY_METRICS` + assertion helper
- `tests/agents/metrics_schema_test.py`
- `tests/agents/metrics_writer_test.py`
- `tests/agents/metrics_contract_smoke_test.py`
- `docs/en/source/api/metrics.md` — user-facing schema reference

**Deleted:**
- `tensoraerospace/agent/metrics.py` (replaced by package)

**Modified (agent migrations):**
- `tensoraerospace/agent/a2c/model.py`
- `tensoraerospace/agent/a2c/narx.py`
- `tensoraerospace/agent/a3c/pytorch.py`
- `tensoraerospace/agent/a3c/utils.py`
- `tensoraerospace/agent/adhdp/model.py`
- `tensoraerospace/agent/adp/adp.py`
- `tensoraerospace/agent/ddpg/model.py`
- `tensoraerospace/agent/dqn/model.py`
- `tensoraerospace/agent/dsac/dsac_flight.py`
- `tensoraerospace/agent/et_dhp/model.py` (add logging from scratch)
- `tensoraerospace/agent/gail/` (add logging from scratch — file path identified during Task 17)
- `tensoraerospace/agent/ppo/model.py`
- `tensoraerospace/agent/sac/sac.py`

---

## Task 1: Create the metrics package skeleton

**Files:**
- Create: `tensoraerospace/agent/metrics/__init__.py`
- Create: `tensoraerospace/agent/metrics/schema.py`
- Create: `tensoraerospace/agent/metrics/writer.py`
- Create: `tensoraerospace/agent/metrics/contract.py`
- Delete: `tensoraerospace/agent/metrics.py` (deferred to Task 5 once all submodules exist)

- [ ] **Step 1: Create `schema.py` with all canonical constants**

Create `tensoraerospace/agent/metrics/schema.py`:

```python
"""Canonical TensorBoard metric names for TensorAeroSpace RL agents.

Every metric written through ``MetricWriter`` (with ``strict=True``)
must use one of the constants defined here. Group prefixes are stable:

    rollout/      — per-episode environment statistics
    loss/         — training losses
    policy/       — policy / action statistics
    value/        — value-function statistics
    diagnostics/  — algorithm-specific diagnostics
    train/        — training progress counters
    eval/         — evaluation episode statistics
    weights/      — network weight histograms
    grads/        — gradient histograms
"""

from __future__ import annotations

import re
from typing import FrozenSet

# ---------------------------------------------------------------------------
# Tier 1 — Mandatory minimum (every RL agent must log)
# ---------------------------------------------------------------------------

ROLLOUT_EPISODE_REWARD = "rollout/episode_reward"
ROLLOUT_EPISODE_LENGTH = "rollout/episode_length"
ROLLOUT_TOTAL_STEPS = "rollout/total_steps"
TRAIN_UPDATES = "train/updates"
TRAIN_LR = "train/lr"

# ---------------------------------------------------------------------------
# Tier 2 — Common (logged when applicable)
# ---------------------------------------------------------------------------

# loss/*
LOSS_ACTOR = "loss/actor"
LOSS_CRITIC = "loss/critic"
LOSS_ENTROPY = "loss/entropy"
LOSS_VALUE = "loss/value"

# policy/*
POLICY_ENTROPY = "policy/entropy"
POLICY_ACTION_STD = "policy/action_std"
POLICY_ACTION_ABS_MEAN = "policy/action_abs_mean"

# value/*
VALUE_MEAN = "value/mean"
VALUE_TD_TARGET = "value/td_target_mean"
VALUE_TD_ERROR_MEAN = "value/td_error_mean"
VALUE_TD_ERROR_MAX = "value/td_error_max"
VALUE_TD_ERROR_MIN = "value/td_error_min"

# diagnostics/*
DIAG_TERMINATED_COUNT = "diagnostics/terminated_count"
DIAG_TRUNCATED_COUNT = "diagnostics/truncated_count"

# eval/*
EVAL_EPISODE_REWARD = "eval/episode_reward"
EVAL_EPISODE_LENGTH = "eval/episode_length"


# ---------------------------------------------------------------------------
# Tier 3 — Per-algorithm extras
# ---------------------------------------------------------------------------

class PPO:
    APPROX_KL = "diagnostics/approx_kl"
    CLIP_FRACTION = "diagnostics/clip_fraction"
    EXPLAINED_VARIANCE = "diagnostics/explained_variance"
    REWARD_MEDIAN = "rollout/episode_reward_median"
    REWARD_P10 = "rollout/episode_reward_p10"
    REWARD_P90 = "rollout/episode_reward_p90"


class SAC:
    LOSS_Q1 = "loss/q1"
    LOSS_Q2 = "loss/q2"
    LOSS_POLICY = "loss/policy"
    LOSS_ALPHA = "loss/alpha"
    ALPHA_VALUE = "policy/alpha"
    Q_MEAN = "value/q_mean"
    LOG_PI_MEAN = "policy/log_pi_mean"
    REPLAY_SIZE = "train/replay_size"


class DSAC:
    # DSAC reuses SAC.* names for the standard losses (loss/q1, loss/q2,
    # loss/policy, loss/alpha, policy/alpha, train/replay_size). It only
    # adds its CAPS regularization terms here.
    CAPS_SPATIAL = "loss/caps_spatial"
    CAPS_TEMPORAL = "loss/caps_temporal"


class DQN:
    LOSS_Q = "loss/q"
    Q_PRED_SA_MEAN = "value/q_pred_mean"
    Q_TARGET_SA_MEAN = "value/q_target_mean"
    EPSILON = "train/epsilon"
    PER_BETA = "train/per_beta"
    REPLAY_SIZE = "train/replay_size"
    TARGET_UPDATE = "train/target_update"


class DDPG:
    LOSS_POLICY = "loss/policy"
    LOSS_VALUE = "loss/value"
    REPLAY_SIZE = "train/replay_size"


class A2C:
    ADVANTAGE_MEAN = "value/advantage_mean"
    ADVANTAGE_STD = "value/advantage_std"
    ADVANTAGE_NORMALIZED_MEAN = "value/advantage_normalized_mean"
    VALUE_BEFORE_UPDATE = "value/before_update_mean"
    ENTROPY_BETA = "policy/entropy_beta"


class ADP:
    DHP_PHASE_EPISODE = "train/dhp_phase_episode"
    LOSS_ACTOR_HDP = "loss/actor_hdp"
    LOSS_ACTOR_GDHP = "loss/actor_adgdhp"
    LOSS_CRITIC_HDP = "loss/critic_hdp"
    LOSS_CRITIC_GDHP = "loss/critic_gdhp"
    LOSS_CRITIC_LAMBDA = "loss/critic_lambda"


class ADHDP:
    DO_CRITIC = "train/do_critic"
    DO_ACTOR = "train/do_actor"
    ACTION_SAT_FRAC = "policy/action_sat_frac"


class GAIL:
    LOSS_DISCRIMINATOR = "loss/discriminator"
    LOSS_GENERATOR = "loss/generator"
    EXPERT_ACCURACY = "diagnostics/expert_accuracy"
    POLICY_ACCURACY = "diagnostics/policy_accuracy"


# ---------------------------------------------------------------------------
# Histogram / multi-worker prefix conventions
# ---------------------------------------------------------------------------

# Allowed histogram top-level groups.
HISTOGRAM_GROUPS: FrozenSet[str] = frozenset({"weights", "grads"})

# Allowed second-level groups inside weights/<group>/<param> and grads/<group>/<param>.
HISTOGRAM_SUBGROUPS: FrozenSet[str] = frozenset({
    "actor", "critic", "policy", "value", "q1", "q2", "discriminator",
})

# Multi-worker suffix pattern: any registered scalar tag may be suffixed with
# "/worker_<N>" where N is a non-negative integer.
_WORKER_SUFFIX_RE = re.compile(r"/worker_\d+$")


def strip_worker_suffix(tag: str) -> str:
    """Return ``tag`` with any trailing ``/worker_<N>`` stripped."""
    return _WORKER_SUFFIX_RE.sub("", tag)


def _collect_constants(namespace) -> FrozenSet[str]:
    """Return all UPPER_SNAKE_CASE string attributes from a module or class."""
    if isinstance(namespace, type):
        items = vars(namespace).items()
    else:
        items = vars(namespace).items()
    return frozenset(
        v
        for k, v in items
        if k.isupper() and isinstance(v, str) and not k.startswith("_")
    )


def _build_registry() -> FrozenSet[str]:
    import sys
    module = sys.modules[__name__]
    parts = [_collect_constants(module)]
    for cls in (PPO, SAC, DSAC, DQN, DDPG, A2C, ADP, ADHDP, GAIL):
        parts.append(_collect_constants(cls))
    out: set[str] = set()
    for p in parts:
        out |= p
    return frozenset(out)


REGISTRY: FrozenSet[str] = _build_registry()


def is_registered_scalar(tag: str) -> bool:
    """Return True if ``tag`` (after stripping multi-worker suffix) is in REGISTRY."""
    return strip_worker_suffix(tag) in REGISTRY


def is_registered_histogram(tag: str) -> bool:
    """Return True if ``tag`` matches ``<top>/<sub>/<param...>`` with allowed groups."""
    parts = tag.split("/")
    if len(parts) < 3:
        return False
    top, sub = parts[0], parts[1]
    return top in HISTOGRAM_GROUPS and sub in HISTOGRAM_SUBGROUPS
```

- [ ] **Step 2: Create `contract.py`**

Create `tensoraerospace/agent/metrics/contract.py`:

```python
"""Mandatory metric contract for RL training agents."""

from __future__ import annotations

from typing import Iterable, Tuple

from . import schema

MANDATORY_METRICS: Tuple[str, ...] = (
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
)


class MetricsContractError(AssertionError):
    """Raised when an agent's training run did not log the mandatory minimum."""


def check_contract(written_tags: Iterable[str], required: Iterable[str]) -> None:
    """Raise MetricsContractError if any required tag was never written."""
    written = set(written_tags)
    missing = [t for t in required if t not in written]
    if missing:
        raise MetricsContractError(
            f"Mandatory metrics never written: {missing}. "
            "Make sure the agent's train() loop calls writer.log_episode(...) "
            "and writes train/updates and train/lr."
        )
```

- [ ] **Step 3: Create `writer.py` (preserves the fallback + lazy import from old metrics.py)**

Create `tensoraerospace/agent/metrics/writer.py`:

```python
"""MetricWriter — strict-whitelist wrapper around torch.utils.tensorboard.SummaryWriter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Set, Union

from . import schema
from .contract import MANDATORY_METRICS, check_contract


class _FallbackSummaryWriter:
    """No-op SummaryWriter used when tensorboard is unavailable."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def add_histogram(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _get_summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter
    except Exception:
        return _FallbackSummaryWriter


class _LazyTorchSummaryWriter:
    def __call__(self, *args, **kwargs):
        cls = _get_summary_writer_class()
        return cls(*args, **kwargs)


TorchSummaryWriter = _LazyTorchSummaryWriter()


class MetricWriter:
    """SummaryWriter wrapper that enforces the canonical metric schema.

    Parameters
    ----------
    log_dir
        TensorBoard log directory.
    strict
        If True, ``add_scalar``/``add_histogram`` raise ``ValueError`` for tags
        not in ``schema.REGISTRY`` (after stripping multi-worker suffix) or not
        matching the histogram prefix rule.
    required
        Tuple of tags that must be written at least once during the writer's
        lifetime. Checked by ``assert_contract_satisfied``.
    algo
        Optional algorithm label, included in error messages.
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        *,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: Optional[str] = None,
    ) -> None:
        log_path = str(log_dir) if log_dir is not None else None
        self._writer = (
            TorchSummaryWriter(log_dir=log_path)
            if log_path is not None
            else TorchSummaryWriter()
        )
        self._strict = strict
        self._required = tuple(required)
        self._algo = algo
        self._written: Set[str] = set()

    # -- core api ----------------------------------------------------------

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        if self._strict and not schema.is_registered_scalar(tag):
            raise ValueError(
                f"Unknown metric tag {tag!r}"
                + (f" (algo={self._algo})" if self._algo else "")
                + ". Register it in tensoraerospace.agent.metrics.schema "
                "or construct MetricWriter(strict=False)."
            )
        self._written.add(schema.strip_worker_suffix(tag))
        self._writer.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        if self._strict and not schema.is_registered_histogram(tag):
            raise ValueError(
                f"Unknown histogram tag {tag!r}. "
                "Histograms must match weights/<group>/<param> or "
                "grads/<group>/<param> with <group> in "
                f"{sorted(schema.HISTOGRAM_SUBGROUPS)}."
            )
        self._writer.add_histogram(tag, values, env_step)

    # -- sugar -------------------------------------------------------------

    def log_episode(
        self,
        *,
        reward: float,
        length: int,
        env_step: int,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        """Write the mandatory rollout/* tier for one finished episode."""
        self.add_scalar(schema.ROLLOUT_EPISODE_REWARD, float(reward), env_step)
        self.add_scalar(schema.ROLLOUT_EPISODE_LENGTH, int(length), env_step)
        self.add_scalar(schema.ROLLOUT_TOTAL_STEPS, int(env_step), env_step)
        if terminated is not None:
            self.add_scalar(schema.DIAG_TERMINATED_COUNT, int(bool(terminated)), env_step)
        if truncated is not None:
            self.add_scalar(schema.DIAG_TRUNCATED_COUNT, int(bool(truncated)), env_step)

    def assert_contract_satisfied(self) -> None:
        """Raise MetricsContractError if any required metric was never written."""
        check_contract(self._written, self._required)

    # -- lifecycle ---------------------------------------------------------

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def create_metric_writer(
    log_dir: Optional[Union[str, Path]] = None,
    *,
    strict: bool = True,
    algo: Optional[str] = None,
) -> MetricWriter:
    """Factory used by agents — keeps call sites short."""
    return MetricWriter(log_dir=log_dir, strict=strict, algo=algo)
```

- [ ] **Step 4: Create `__init__.py`**

Create `tensoraerospace/agent/metrics/__init__.py`:

```python
"""Unified TensorBoard metrics for TensorAeroSpace RL agents.

See ``docs/superpowers/specs/2026-04-19-unified-tensorboard-metrics-design.md``
for the full schema.
"""

from . import schema
from .contract import MANDATORY_METRICS, MetricsContractError, check_contract
from .writer import MetricWriter, TorchSummaryWriter, create_metric_writer

__all__ = [
    "schema",
    "MetricWriter",
    "TorchSummaryWriter",
    "create_metric_writer",
    "MANDATORY_METRICS",
    "MetricsContractError",
    "check_contract",
]
```

- [ ] **Step 5: Delete the old single-file `metrics.py`**

Run:
```bash
git rm tensoraerospace/agent/metrics.py
```

This breaks the old `from tensoraerospace.agent.metrics import normalize_tag, ensure_metric_writer` callers — every agent migration in later tasks updates them.

- [ ] **Step 6: Verify the package imports cleanly**

Run:
```bash
python -c "from tensoraerospace.agent.metrics import schema, MetricWriter, MANDATORY_METRICS; print(sorted(schema.REGISTRY)[:5]); print(MANDATORY_METRICS)"
```

Expected: prints first 5 registered tags (sorted) and the mandatory tuple. No exceptions.

- [ ] **Step 7: Commit**

```bash
git add tensoraerospace/agent/metrics/
git rm tensoraerospace/agent/metrics.py
git commit -m "refactor(metrics): create metrics package with strict whitelist schema

Replaces tensoraerospace/agent/metrics.py with a package:
  schema.py    — canonical names + REGISTRY
  writer.py    — MetricWriter with strict whitelist + required env_step
  contract.py  — mandatory minimum + assertion helper
  __init__.py  — public re-exports

Agent migrations follow in subsequent commits."
```

---

## Task 2: Schema unit tests

**Files:**
- Create: `tests/agents/metrics_schema_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/metrics_schema_test.py`:

```python
"""Schema invariants for tensoraerospace.agent.metrics.schema."""

import re

import pytest

from tensoraerospace.agent.metrics import schema


_VALID_TAG = re.compile(r"^[a-z][a-z0-9_]*(/[a-z0-9_]+)+$")


def _all_constants():
    """Yield (qualified_name, value) for every UPPER_SNAKE_CASE string constant."""
    for name in dir(schema):
        if name.startswith("_"):
            continue
        obj = getattr(schema, name)
        if isinstance(obj, str) and name.isupper():
            yield name, obj
        elif isinstance(obj, type) and obj.__module__ == schema.__name__:
            for sub in dir(obj):
                if sub.startswith("_"):
                    continue
                val = getattr(obj, sub)
                if isinstance(val, str) and sub.isupper():
                    yield f"{name}.{sub}", val


def test_all_tags_match_canonical_pattern():
    bad = [(n, v) for n, v in _all_constants() if not _VALID_TAG.match(v)]
    assert not bad, f"Tags violating canonical pattern: {bad}"


def test_no_duplicate_tag_values():
    seen: dict[str, str] = {}
    dups: list[tuple[str, str, str]] = []
    for name, value in _all_constants():
        if value in seen and seen[value] != name:
            dups.append((value, seen[value], name))
        else:
            seen[value] = name
    assert not dups, f"Duplicate tag values across constants: {dups}"


def test_registry_contains_every_constant():
    expected = {v for _, v in _all_constants()}
    missing = expected - schema.REGISTRY
    assert not missing, f"Constants missing from REGISTRY: {missing}"


def test_strip_worker_suffix_removes_trailing_worker_id():
    assert (
        schema.strip_worker_suffix("rollout/episode_reward/worker_0")
        == "rollout/episode_reward"
    )
    assert (
        schema.strip_worker_suffix("loss/actor/worker_42")
        == "loss/actor"
    )
    assert schema.strip_worker_suffix("loss/actor") == "loss/actor"


def test_is_registered_scalar_accepts_worker_suffix():
    assert schema.is_registered_scalar("rollout/episode_reward/worker_3")
    assert schema.is_registered_scalar(schema.LOSS_ACTOR)
    assert not schema.is_registered_scalar("Performance/Reward")
    assert not schema.is_registered_scalar("loss/actor/wrkr_3")  # bad suffix


def test_is_registered_histogram_validates_two_level_prefix():
    assert schema.is_registered_histogram("weights/actor/fc1.weight")
    assert schema.is_registered_histogram("grads/critic/conv.bias")
    assert schema.is_registered_histogram("weights/q1/layer.0.weight")
    assert not schema.is_registered_histogram("weights/unknown/x")
    assert not schema.is_registered_histogram("foo/actor/x")
    assert not schema.is_registered_histogram("weights/actor")  # too short


def test_mandatory_subset_of_registry():
    from tensoraerospace.agent.metrics import MANDATORY_METRICS
    assert set(MANDATORY_METRICS).issubset(schema.REGISTRY)
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest tests/agents/metrics_schema_test.py -v`
Expected: 7 PASSED. If any fail, fix the schema (most likely a typo in a constant value), not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/metrics_schema_test.py
git commit -m "test(metrics): add schema invariant tests"
```

---

## Task 3: MetricWriter unit tests

**Files:**
- Create: `tests/agents/metrics_writer_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/metrics_writer_test.py`:

```python
"""Behaviour of tensoraerospace.agent.metrics.writer.MetricWriter."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoraerospace.agent.metrics import (
    MANDATORY_METRICS,
    MetricWriter,
    MetricsContractError,
    schema,
)


@pytest.fixture
def writer(tmp_path: Path) -> MetricWriter:
    return MetricWriter(log_dir=str(tmp_path / "tb"), algo="test")


def test_add_scalar_accepts_registered_tag(writer: MetricWriter):
    writer.add_scalar(schema.LOSS_ACTOR, 0.123, env_step=10)


def test_add_scalar_rejects_unregistered_tag(writer: MetricWriter):
    with pytest.raises(ValueError, match="Unknown metric tag 'Performance/Reward'"):
        writer.add_scalar("Performance/Reward", 1.0, env_step=1)


def test_add_scalar_accepts_worker_suffix(writer: MetricWriter):
    writer.add_scalar(f"{schema.LOSS_ACTOR}/worker_2", 0.5, env_step=5)


def test_add_scalar_rejects_bad_worker_suffix(writer: MetricWriter):
    with pytest.raises(ValueError):
        writer.add_scalar(f"{schema.LOSS_ACTOR}/wrkr_2", 0.5, env_step=5)


def test_add_scalar_strict_false_allows_anything(tmp_path: Path):
    w = MetricWriter(log_dir=str(tmp_path / "tb"), strict=False)
    w.add_scalar("anything/at/all", 1.0, env_step=1)


def test_add_scalar_requires_env_step(writer: MetricWriter):
    with pytest.raises(TypeError):
        writer.add_scalar(schema.LOSS_ACTOR, 0.5)  # type: ignore[call-arg]


def test_add_histogram_accepts_valid_prefix(writer: MetricWriter):
    import numpy as np
    writer.add_histogram("weights/actor/fc1.weight", np.zeros(8), env_step=1)


def test_add_histogram_rejects_invalid_prefix(writer: MetricWriter):
    import numpy as np
    with pytest.raises(ValueError, match="Unknown histogram tag"):
        writer.add_histogram("parameters/actor", np.zeros(8), env_step=1)


def test_log_episode_writes_mandatory_rollout_tier(writer: MetricWriter):
    writer.log_episode(reward=12.5, length=200, env_step=200,
                       terminated=False, truncated=True)
    # all mandatory rollout tags should now be marked written
    written = writer._written  # noqa: SLF001 — internal but stable for tests
    assert schema.ROLLOUT_EPISODE_REWARD in written
    assert schema.ROLLOUT_EPISODE_LENGTH in written
    assert schema.ROLLOUT_TOTAL_STEPS in written
    assert schema.DIAG_TRUNCATED_COUNT in written
    assert schema.DIAG_TERMINATED_COUNT in written


def test_assert_contract_raises_when_missing(writer: MetricWriter):
    # Only log rollout tier, miss train/updates and train/lr
    writer.log_episode(reward=1.0, length=10, env_step=10)
    with pytest.raises(MetricsContractError, match="train/updates"):
        writer.assert_contract_satisfied()


def test_assert_contract_passes_when_complete(writer: MetricWriter):
    writer.log_episode(reward=1.0, length=10, env_step=10)
    writer.add_scalar(schema.TRAIN_UPDATES, 5, env_step=10)
    writer.add_scalar(schema.TRAIN_LR, 1e-3, env_step=10)
    writer.assert_contract_satisfied()


def test_close_does_not_raise(writer: MetricWriter):
    writer.close()
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `pytest tests/agents/metrics_writer_test.py -v`
Expected: 12 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/metrics_writer_test.py
git commit -m "test(metrics): add MetricWriter behaviour tests"
```

---

## Task 4: Per-agent smoke-test helper

**Files:**
- Create: `tests/agents/metrics_contract_smoke_test.py`

This file is a single utility used by agent migrations: read TensorBoard event files in a directory and return the set of scalar tags written.

- [ ] **Step 1: Write the helper module**

Create `tests/agents/metrics_contract_smoke_test.py`:

```python
"""Helpers for verifying that an agent wrote canonical TB tags during train()."""

from __future__ import annotations

import glob
import os
from typing import Set

import pytest


def read_event_scalar_tags(log_dir: str) -> Set[str]:
    """Return the set of scalar tags found across all event files under log_dir."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        pytest.skip("tensorboard package not installed")

    if not os.path.isdir(log_dir):
        return set()
    tags: Set[str] = set()
    for ev in glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"),
                        recursive=True):
        ea = EventAccumulator(os.path.dirname(ev))
        ea.Reload()
        tags.update(ea.Tags().get("scalars", []))
    return tags


def assert_tags_present(log_dir: str, required: set[str]) -> None:
    """Raise AssertionError if any of ``required`` is missing from the event files."""
    found = read_event_scalar_tags(log_dir)
    missing = required - found
    assert not missing, f"Missing tags in {log_dir}: sorted={sorted(missing)} found={sorted(found)}"


def test_helper_reads_empty_dir(tmp_path):
    """Smoke check that the helper handles a directory with no event files."""
    assert read_event_scalar_tags(str(tmp_path)) == set()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/agents/metrics_contract_smoke_test.py -v`
Expected: 1 PASSED (or SKIPPED if tensorboard is not installed in the environment).

- [ ] **Step 3: Commit**

```bash
git add tests/agents/metrics_contract_smoke_test.py
git commit -m "test(metrics): add event-file scalar-tag reader helper"
```

---

## Task 5: Migrate DDPG

**Files:**
- Modify: `tensoraerospace/agent/ddpg/model.py` (logging block around line 752, `Performance/Reward` ~963)
- Test: `tests/agents/ddpg_metrics_smoke_test.py` (new)

DDPG currently logs:
- `"Performance/Reward"`
- `"loss/policy"`, `"loss/value"`
- `"policy/mean_action"`

Target canonical names:
- `schema.ROLLOUT_EPISODE_REWARD`
- `schema.DDPG.LOSS_POLICY`, `schema.DDPG.LOSS_VALUE`
- `schema.POLICY_ACTION_ABS_MEAN` (rename `mean_action` → `action_abs_mean`)
- Add: `schema.TRAIN_UPDATES`, `schema.TRAIN_LR`, `schema.ROLLOUT_EPISODE_LENGTH`, `schema.ROLLOUT_TOTAL_STEPS` (use `writer.log_episode(...)`)

- [ ] **Step 1: Inspect current logging code**

Run: `grep -n "add_scalar\|writer\." tensoraerospace/agent/ddpg/model.py | head -40`

This identifies the exact lines and current tag strings.

- [ ] **Step 2: Write a smoke test that asserts canonical tags exist after a tiny train run**

Create `tests/agents/ddpg_metrics_smoke_test.py`:

```python
"""Smoke test: DDPG.train() writes canonical TensorBoard tags."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present


REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.DDPG.LOSS_POLICY,
    schema.DDPG.LOSS_VALUE,
}


@pytest.mark.timeout(60)
def test_ddpg_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")

    import gymnasium as gym
    from tensoraerospace.agent.ddpg.model import DDPG

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"
    agent = DDPG(env=env, log_dir=str(log_dir))  # adjust ctor kwargs to match
    agent.train(num_episodes=2, max_steps=8, verbose=False)
    agent.writer.close()

    assert_tags_present(str(log_dir), REQUIRED)
```

Note: the constructor kwargs may differ — match the agent's actual API. If
DDPG cannot accept a stub env in 2 episodes × 8 steps, increase to the
minimum that works while still completing in <60 s.

- [ ] **Step 3: Run the smoke test to verify it fails (old tag names)**

Run: `pytest tests/agents/ddpg_metrics_smoke_test.py -v`
Expected: FAIL with "Missing tags in <tmp>: sorted=['rollout/...']" — old names like `Performance/Reward` were used.

- [ ] **Step 4: Migrate the DDPG source**

Modify `tensoraerospace/agent/ddpg/model.py`:

a) Replace the import:
```python
# old
from tensoraerospace.agent.metrics import create_metric_writer
# new
from tensoraerospace.agent.metrics import create_metric_writer, schema
```

b) Construct the writer with `algo="ddpg"`:
```python
self.writer = create_metric_writer(self.log_dir, algo="ddpg")
```

c) Replace each `add_scalar` call with the constant + `env_step=...`:
```python
# old
self.writer.add_scalar("loss/policy", policy_loss.item(), self.update_count)
# new
self.writer.add_scalar(schema.DDPG.LOSS_POLICY, policy_loss.item(),
                       env_step=self.global_env_step)
```
Repeat for:
- `"loss/value"` → `schema.DDPG.LOSS_VALUE`
- `"policy/mean_action"` → `schema.POLICY_ACTION_ABS_MEAN` (and ensure value is `abs(action).mean()`)
- `"Performance/Reward"` → use `self.writer.log_episode(reward=ep_reward, length=ep_length, env_step=self.global_env_step, terminated=terminated, truncated=truncated)` at episode end

d) Add training-progress logging once per update:
```python
self.writer.add_scalar(schema.TRAIN_UPDATES, self.update_count,
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.TRAIN_LR,
                       self.actor_optimizer.param_groups[0]["lr"],
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.DDPG.REPLAY_SIZE, len(self.replay_buffer),
                       env_step=self.global_env_step)
```

e) Add `self.global_env_step` initialization (`self.global_env_step = 0`) and increment it on every `env.step(...)` call inside the rollout loop.

f) At the end of `train()`:
```python
self.writer.assert_contract_satisfied()
self.writer.close()
```

- [ ] **Step 5: Run the smoke test to verify it now passes**

Run: `pytest tests/agents/ddpg_metrics_smoke_test.py -v`
Expected: PASSED.

- [ ] **Step 6: Run the existing DDPG tests**

Run: `pytest tests/agents/test_ddpg_*.py -v`
Expected: all PASSED. If any failure references an old metric name, update the test to the canonical name.

- [ ] **Step 7: Commit**

```bash
git add tensoraerospace/agent/ddpg/model.py tests/agents/ddpg_metrics_smoke_test.py
git commit -m "refactor(ddpg): adopt unified TB metrics schema"
```

---

## Task 6: Migrate DQN

**Files:**
- Modify: `tensoraerospace/agent/dqn/model.py`
- Test: `tests/agents/dqn_metrics_smoke_test.py`

Current DQN tags → canonical:
| Old | New |
|---|---|
| `"Loss/DQN"` | `schema.DQN.LOSS_Q` |
| `"Q/PredSA/Mean"` | `schema.DQN.Q_PRED_SA_MEAN` |
| `"Q/TargetSA/Mean"` | `schema.DQN.Q_TARGET_SA_MEAN` |
| `"TD-Error/Mean"` | `schema.VALUE_TD_ERROR_MEAN` |
| `"TD-Error/Max"` | `schema.VALUE_TD_ERROR_MAX` |
| `"TD-Error/Min"` | `schema.VALUE_TD_ERROR_MIN` |
| `"PER/Beta"` | `schema.DQN.PER_BETA` |
| `f"DQN/{name}"` | `f"weights/q1/{name}"` |
| `"Performance/Reward"` | `writer.log_episode(...)` |
| `"Target/Update"` | `schema.DQN.TARGET_UPDATE` |
| `"Exploration/Epsilon"` | `schema.DQN.EPSILON` |
| `"Performance/EvalReward"` | `schema.EVAL_EPISODE_REWARD` |

- [ ] **Step 1: Write smoke test**

Create `tests/agents/dqn_metrics_smoke_test.py`:

```python
"""Smoke test: DQN.train() writes canonical TensorBoard tags."""

from pathlib import Path
import pytest

from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present

REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.DQN.LOSS_Q,
    schema.DQN.EPSILON,
}


@pytest.mark.timeout(60)
def test_dqn_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")

    import gymnasium as gym
    from tensoraerospace.agent.dqn.model import DQNAgent

    env = gym.make("CartPole-v1")
    log_dir = tmp_path / "tb"
    agent = DQNAgent(env=env, log_dir=str(log_dir))
    agent.train(num_episodes=2, max_steps=8, verbose=False)
    agent.writer.close()

    assert_tags_present(str(log_dir), REQUIRED)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/agents/dqn_metrics_smoke_test.py -v`
Expected: FAIL.

- [ ] **Step 3: Migrate `tensoraerospace/agent/dqn/model.py`**

Apply the rename table above to every `writer.add_scalar(...)` and
`writer.add_histogram(...)` call. For histograms use the `weights/q1/<name>`
form (DQN's online network is conceptually `q1`).

Add at episode end:
```python
self.writer.log_episode(
    reward=ep_reward,
    length=ep_length,
    env_step=self.global_env_step,
    terminated=terminated,
    truncated=truncated,
)
```

Add per-update:
```python
self.writer.add_scalar(schema.TRAIN_UPDATES, self.update_count,
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.TRAIN_LR,
                       self.optimizer.param_groups[0]["lr"],
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.DQN.REPLAY_SIZE, len(self.replay_buffer),
                       env_step=self.global_env_step)
```

End of `train()`:
```python
self.writer.assert_contract_satisfied()
self.writer.close()
```

- [ ] **Step 4: Run smoke test, expect PASS**

Run: `pytest tests/agents/dqn_metrics_smoke_test.py -v`
Expected: PASSED.

- [ ] **Step 5: Run existing DQN tests, fix any tag-string assertions**

Run: `pytest tests/agents/test_dqn_*.py -v`
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/dqn/model.py tests/agents/dqn_metrics_smoke_test.py
git commit -m "refactor(dqn): adopt unified TB metrics schema"
```

---

## Task 7: Migrate SAC

**Files:**
- Modify: `tensoraerospace/agent/sac/sac.py`
- Test: `tests/agents/sac_metrics_smoke_test.py`

Current SAC tags → canonical:
| Old | New |
|---|---|
| `"Loss/QF1"` | `schema.SAC.LOSS_Q1` |
| `"Loss/QF2"` | `schema.SAC.LOSS_Q2` |
| `"Loss/Policy"` | `schema.SAC.LOSS_POLICY` |
| `"Loss/Alpha"` | `schema.SAC.LOSS_ALPHA` |
| `"Alpha/value"` | `schema.SAC.ALPHA_VALUE` |
| `"Performance/Reward"` and `"Performance/EpisodeReward"` | `writer.log_episode(...)` |
| `"Performance/EpisodeLength"` | (covered by `log_episode`) |
| `"Performance/BestReward"` | drop (computable in postprocessing) |
| `"Performance/BestMeanReward"` | drop (same) |
| `"Train/ReplaySize"` | `schema.SAC.REPLAY_SIZE` |
| `"Train/Updates"` | `schema.TRAIN_UPDATES` |
| `"Train/TotalSteps"` | drop (covered by `ROLLOUT_TOTAL_STEPS`) |
| `f"Critic/{name}"` | `f"weights/critic/{name}"` |
| `f"Policy/{name}"` | `f"weights/policy/{name}"` |

- [ ] **Step 1: Write smoke test**

Create `tests/agents/sac_metrics_smoke_test.py`:

```python
from pathlib import Path
import pytest
from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present

REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.SAC.LOSS_Q1,
    schema.SAC.LOSS_Q2,
    schema.SAC.LOSS_POLICY,
    schema.SAC.LOSS_ALPHA,
    schema.SAC.REPLAY_SIZE,
}


@pytest.mark.timeout(120)
def test_sac_train_writes_canonical_tags(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    from tensoraerospace.agent.sac.sac import SAC

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"
    agent = SAC(env=env, log_dir=str(log_dir))
    agent.train(num_episodes=2, max_steps=16, verbose=False)
    agent.writer.close()

    assert_tags_present(str(log_dir), REQUIRED)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/agents/sac_metrics_smoke_test.py -v`

- [ ] **Step 3: Migrate `tensoraerospace/agent/sac/sac.py`**

Apply the rename table. At episode end, replace the manual reward/length
writes with `writer.log_episode(...)`. Drop best-reward metrics (postprocess).
Add `TRAIN_LR` write per update.

- [ ] **Step 4: Run smoke test, expect PASS**

- [ ] **Step 5: Run existing SAC tests**

Run: `pytest tests/agents/sac_*.py tests/agents/sac_train_smoke_test.py -v`
Expected: all PASSED. Update any test asserting old tag strings.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/sac/sac.py tests/agents/sac_metrics_smoke_test.py
git commit -m "refactor(sac): adopt unified TB metrics schema"
```

---

## Task 8: Migrate DSAC

**Files:**
- Modify: `tensoraerospace/agent/dsac/dsac_flight.py`
- Test: `tests/agents/dsac_metrics_smoke_test.py`

DSAC reuses `schema.SAC.*` for standard SAC metrics (since the rename table
matches: `Loss/Z1` → `loss/q1`, etc.) and adds `schema.DSAC.CAPS_SPATIAL`,
`schema.DSAC.CAPS_TEMPORAL`.

Mapping:
| Old | New |
|---|---|
| `"Loss/Z1"` | `schema.SAC.LOSS_Q1` |
| `"Loss/Z2"` | `schema.SAC.LOSS_Q2` |
| `"Loss/Policy"` | `schema.SAC.LOSS_POLICY` |
| `"Loss/Alpha"` | `schema.SAC.LOSS_ALPHA` |
| `"Alpha/value"` | `schema.SAC.ALPHA_VALUE` |
| `"Train/Q_mean"` | `schema.SAC.Q_MEAN` |
| `"Train/LogPi_mean"` | `schema.SAC.LOG_PI_MEAN` |
| `"Train/CAPS_spatial"` | `schema.DSAC.CAPS_SPATIAL` |
| `"Train/CAPS_temporal"` | `schema.DSAC.CAPS_TEMPORAL` |
| `"Train/ActionAbsMean"` | `schema.POLICY_ACTION_ABS_MEAN` |
| `"Performance/Reward"` / `"Performance/EpisodeReward"` | `writer.log_episode(...)` |
| `"Performance/EpisodeLength"` | (covered by `log_episode`) |
| `"Performance/BestReward"` / `"Performance/BestMeanReward"` | drop |
| `"Train/ReplaySize"` | `schema.SAC.REPLAY_SIZE` |
| `"Train/Updates"` | `schema.TRAIN_UPDATES` |
| `"Train/TotalSteps"` | drop |
| `"Diagnostics/TerminatedCount"` / `"Diagnostics/TruncatedCount"` | covered by `log_episode(terminated=, truncated=)` |

- [ ] **Step 1: Write smoke test**

Create `tests/agents/dsac_metrics_smoke_test.py` mirroring `sac_metrics_smoke_test.py` but additionally requiring `schema.DSAC.CAPS_SPATIAL` and `schema.DSAC.CAPS_TEMPORAL`.

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Migrate `tensoraerospace/agent/dsac/dsac_flight.py`** — apply table; both `train()` and `train_vector()` paths must be updated.

- [ ] **Step 4: Run smoke test, expect PASS**

- [ ] **Step 5: Run existing DSAC tests**

Run: `pytest tests/agents/dsac_*.py -v`

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/dsac/dsac_flight.py tests/agents/dsac_metrics_smoke_test.py
git commit -m "refactor(dsac): adopt unified TB metrics schema"
```

---

## Task 9: Migrate PPO

**Files:**
- Modify: `tensoraerospace/agent/ppo/model.py`
- Test: `tests/agents/ppo_metrics_smoke_test.py`

Mapping:
| Old | New |
|---|---|
| `"Loss/Actor"` | `schema.LOSS_ACTOR` |
| `"Loss/Critic"` | `schema.LOSS_CRITIC` |
| `"Performance/Reward"` | `writer.log_episode(...)` |
| `"Performance/RewardMedian"` | `schema.PPO.REWARD_MEDIAN` |
| `"Performance/RewardP10"` | `schema.PPO.REWARD_P10` |
| `"Performance/RewardP90"` | `schema.PPO.REWARD_P90` |
| `"Performance/Entropy"` | `schema.POLICY_ENTROPY` |
| `"Performance/Episode Length"` | (covered by `log_episode`) |
| `"Diagnostics/Approx KL"` | `schema.PPO.APPROX_KL` |
| `"Diagnostics/Clip Fraction"` | `schema.PPO.CLIP_FRACTION` |
| `"Diagnostics/Explained Variance"` | `schema.PPO.EXPLAINED_VARIANCE` |
| `"Diagnostics/TerminatedCount"` / `"Diagnostics/TruncatedCount"` | via `log_episode` |
| `"Evaluation/Reward"` | `schema.EVAL_EPISODE_REWARD` |

X-axis: PPO currently uses episode index in many places — replace every
step argument with `self.global_env_step` (incremented on every env.step).

- [ ] **Step 1–6:** Same TDD pattern as Tasks 5–8. Smoke test required:

Required tags for smoke:
```python
REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.LOSS_ACTOR,
    schema.LOSS_CRITIC,
    schema.PPO.APPROX_KL,
    schema.PPO.CLIP_FRACTION,
    schema.POLICY_ENTROPY,
}
```

- [ ] **Final step: Commit**

```bash
git add tensoraerospace/agent/ppo/model.py tests/agents/ppo_metrics_smoke_test.py
git commit -m "refactor(ppo): adopt unified TB metrics schema"
```

---

## Task 10: Migrate A2C (main model)

**Files:**
- Modify: `tensoraerospace/agent/a2c/model.py`
- Test: `tests/agents/a2c_metrics_smoke_test.py`

Mapping (covering the 17 tags in the audit):
| Old | New |
|---|---|
| `"Performance/Episode_Reward"` | `writer.log_episode(...)` |
| `"Performance/Episode_Reward_Avg_10"` | drop (postprocess) |
| `"Performance/Episode_Reward_Avg_100"` | drop |
| `"Loss/Log_probs"` | drop (covered by entropy/actor — keep only if A2C tests rely on it; if so add `schema.A2C` constant) |
| `"Loss/Entropy"` | `schema.LOSS_ENTROPY` |
| `"Loss/Entropy_beta"` | `schema.A2C.ENTROPY_BETA` |
| `"Loss/Actor"` | `schema.LOSS_ACTOR` |
| `"Loss/Critic"` | `schema.LOSS_CRITIC` |
| `"Advantage/Raw_Mean"` | `schema.A2C.ADVANTAGE_MEAN` |
| `"Advantage/Raw_Std"` | `schema.A2C.ADVANTAGE_STD` |
| `"Advantage/Normalized_Mean"` | `schema.A2C.ADVANTAGE_NORMALIZED_MEAN` |
| `"Value/Mean"` | `schema.VALUE_MEAN` |
| `"Value/TD_Target_Mean"` | `schema.VALUE_TD_TARGET` |
| `"Value/Value_Before_Update"` | `schema.A2C.VALUE_BEFORE_UPDATE` |
| `"Policy/Action_Std"` | `schema.POLICY_ACTION_STD` |
| `"Advantage/Mean"` (line 1021) | `schema.A2C.ADVANTAGE_MEAN` (deduplicate) |

Add per-update `TRAIN_LR` and `TRAIN_UPDATES` writes.
Change all `step` arguments to `self.global_env_step`.

- [ ] **Steps 1–6:** Same TDD pattern. Required tags:
```python
REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD, schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS, schema.TRAIN_UPDATES, schema.TRAIN_LR,
    schema.LOSS_ACTOR, schema.LOSS_CRITIC, schema.LOSS_ENTROPY,
    schema.A2C.ADVANTAGE_MEAN,
}
```

- [ ] **Commit:**

```bash
git add tensoraerospace/agent/a2c/model.py tests/agents/a2c_metrics_smoke_test.py
git commit -m "refactor(a2c): adopt unified TB metrics schema"
```

---

## Task 11: Migrate A2C-NARX

**Files:**
- Modify: `tensoraerospace/agent/a2c/narx.py`
- Test: `tests/agents/a2c_narx_metrics_smoke_test.py`

Currently bypasses MetricWriter — uses raw `SummaryWriter` and
`losses/*`, `parameters/*`, `gradients/*`, `episode_reward` tags.

Mapping:
| Old | New |
|---|---|
| `"parameters/actor"` | `f"weights/actor/{name}"` per param |
| `"parameters/critic"` | `f"weights/critic/{name}"` per param |
| `"gradients/actor"` | `f"grads/actor/{name}"` per param |
| `"gradients/critic"` | `f"grads/critic/{name}"` per param |
| `"losses/log_probs"` | drop or add to `schema.A2C` (keep dropped initially) |
| `"losses/entropy"` | `schema.LOSS_ENTROPY` |
| `"losses/entropy_beta"` | `schema.A2C.ENTROPY_BETA` |
| `"losses/actor"` | `schema.LOSS_ACTOR` |
| `"losses/advantage"` | `schema.A2C.ADVANTAGE_MEAN` |
| `"losses/critic"` | `schema.LOSS_CRITIC` |
| `"episode_reward"` | `writer.log_episode(...)` |

Switch construction from raw `SummaryWriter(log_dir)` to
`create_metric_writer(log_dir, algo="a2c-narx")`.

- [ ] **Steps 1–6:** Same TDD pattern.

- [ ] **Commit:**

```bash
git add tensoraerospace/agent/a2c/narx.py tests/agents/a2c_narx_metrics_smoke_test.py
git commit -m "refactor(a2c-narx): adopt unified TB metrics schema"
```

---

## Task 12: Migrate A3C (multi-worker)

**Files:**
- Modify: `tensoraerospace/agent/a3c/pytorch.py`
- Modify: `tensoraerospace/agent/a3c/utils.py` (`record()` function)
- Test: `tests/agents/a3c_metrics_smoke_test.py`

A3C is the only agent using `f"Loss/{name}/total"`,
`f"Performance/{name}/episode_reward"`, etc., where `name` is a per-worker
identifier. Move the worker id from the **second** position to the **last**
position so group prefix stays canonical.

Mapping:
| Old | New |
|---|---|
| `f"Loss/{name}/total"` | drop (composite of value+policy+entropy) |
| `f"Loss/{name}/value"` | `f"{schema.LOSS_CRITIC}/worker_{wid}"` |
| `f"Loss/{name}/policy"` | `f"{schema.LOSS_ACTOR}/worker_{wid}"` |
| `f"Loss/{name}/entropy"` | `f"{schema.LOSS_ENTROPY}/worker_{wid}"` |
| `f"Performance/{name}/episode_reward"` | `f"{schema.ROLLOUT_EPISODE_REWARD}/worker_{wid}"` |
| `f"Performance/{name}/moving_avg_reward"` | drop (postprocess) |

- [ ] **Step 1: Add a shared global env-step counter to the worker pool**

Modify `tensoraerospace/agent/a3c/pytorch.py`:

a) Where the worker pool is constructed, add:
```python
import torch.multiprocessing as mp
self.global_env_step = mp.Value("l", 0)  # 'l' = signed long
```

b) Pass `global_env_step` into each `Worker.__init__(...)` and store as `self.global_env_step`.

c) Inside the worker's env loop, after every `env.step(...)`:
```python
with self.global_env_step.get_lock():
    self.global_env_step.value += 1
current_env_step = self.global_env_step.value
```

d) Pass `current_env_step` as `env_step=` for every `add_scalar` call.

- [ ] **Step 2: Replace `SummaryWriter` with `MetricWriter`**

Each worker creates `self.writer = create_metric_writer(self.log_dir, algo=f"a3c-worker-{self.worker_id}")` (or share a single writer in the master — design choice; see existing structure).

If a single writer is shared, only the master worker writes — workers send
metrics via a queue. Keep whichever pattern the current code uses; only
the **tag names and step axis** change.

- [ ] **Step 3: Apply the rename table.**

- [ ] **Step 4: Write smoke test** that asserts `f"{schema.ROLLOUT_EPISODE_REWARD}/worker_0"` exists in event files.

- [ ] **Steps 5–6:** Run smoke test, run existing A3C tests, commit.

```bash
git add tensoraerospace/agent/a3c/ tests/agents/a3c_metrics_smoke_test.py
git commit -m "refactor(a3c): adopt unified TB metrics schema with /worker_<id> suffix"
```

---

## Task 13: Migrate ADHDP

**Files:**
- Modify: `tensoraerospace/agent/adhdp/model.py`
- Test: `tests/agents/adhdp_metrics_smoke_test.py`

Mapping:
| Old | New |
|---|---|
| `"loss/critic"` | `schema.LOSS_CRITIC` (already lowercase — only constant import changes) |
| `"loss/actor"` | `schema.LOSS_ACTOR` |
| `"train/do_critic"` | `schema.ADHDP.DO_CRITIC` |
| `"train/do_actor"` | `schema.ADHDP.DO_ACTOR` |
| `"action/mean_abs"` | `schema.POLICY_ACTION_ABS_MEAN` |
| `"action/sat_frac"` | `schema.ADHDP.ACTION_SAT_FRAC` |
| `"performance/episode_reward"` | `writer.log_episode(...)` |
| `"performance/episode_length"` | covered by `log_episode` |
| `"train/total_steps"` | covered by `log_episode` (`ROLLOUT_TOTAL_STEPS`) |

Add `TRAIN_LR` write per update.

- [ ] **Steps 1–6:** Same TDD pattern.
- [ ] **Commit:**

```bash
git add tensoraerospace/agent/adhdp/model.py tests/agents/adhdp_metrics_smoke_test.py
git commit -m "refactor(adhdp): adopt unified TB metrics schema"
```

---

## Task 14: Migrate ADP

**Files:**
- Modify: `tensoraerospace/agent/adp/adp.py`
- Test: `tests/agents/adp_metrics_smoke_test.py`

Mapping:
| Old | New |
|---|---|
| `"loss/critic"` | `schema.LOSS_CRITIC` |
| `"loss/actor"` | `schema.LOSS_ACTOR` |
| `"loss/actor_hdp"` | `schema.ADP.LOSS_ACTOR_HDP` |
| `"loss/actor_adgdhp"` | `schema.ADP.LOSS_ACTOR_GDHP` |
| `"loss/critic_hdp"` | `schema.ADP.LOSS_CRITIC_HDP` |
| `"loss/critic_gdhp"` | `schema.ADP.LOSS_CRITIC_GDHP` |
| `"loss/critic_adgdhp"` | `schema.ADP.LOSS_CRITIC_GDHP` (same as above — dedupe) |
| `"loss/critic_lambda"` | `schema.ADP.LOSS_CRITIC_LAMBDA` |
| `"performance/episode_reward"` | `writer.log_episode(...)` |
| `"performance/episode_length"` | via `log_episode` |
| `"train/total_steps"` | via `log_episode` |
| `"train/dhp_phase"` / `"train/dhp_phase_episode"` | `schema.ADP.DHP_PHASE_EPISODE` |

- [ ] **Steps 1–6:** Same TDD pattern.

```bash
git add tensoraerospace/agent/adp/adp.py tests/agents/adp_metrics_smoke_test.py
git commit -m "refactor(adp): adopt unified TB metrics schema"
```

---

## Task 15: Add logging to ET-DHP

**Files:**
- Modify: `tensoraerospace/agent/et_dhp/model.py`
- Test: `tests/agents/etdhp_metrics_smoke_test.py`

ET-DHP currently has no TensorBoard logging. Add the mandatory minimum
plus actor/critic losses (reuse `schema.LOSS_ACTOR`, `schema.LOSS_CRITIC`).
No new per-algo constants required initially.

- [ ] **Step 1: Inspect the agent**

Run: `grep -n "def train\|def __init__\|loss" tensoraerospace/agent/et_dhp/model.py | head -30`

Identify the `train()` loop, the optimizer(s), and where actor/critic losses are computed.

- [ ] **Step 2: Add writer construction in `__init__`**

```python
from tensoraerospace.agent.metrics import create_metric_writer, schema

# in __init__:
self.log_dir = log_dir or "runs/etdhp"
self.writer = create_metric_writer(self.log_dir, algo="etdhp")
self.global_env_step = 0
self.update_count = 0
```

- [ ] **Step 3: Write losses inside the update step**

```python
self.update_count += 1
self.writer.add_scalar(schema.LOSS_ACTOR, float(actor_loss),
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.LOSS_CRITIC, float(critic_loss),
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.TRAIN_UPDATES, self.update_count,
                       env_step=self.global_env_step)
self.writer.add_scalar(schema.TRAIN_LR,
                       self.actor_optimizer.param_groups[0]["lr"],
                       env_step=self.global_env_step)
```

- [ ] **Step 4: Increment `global_env_step` after each env.step, log episode at end**

```python
# inside rollout loop:
self.global_env_step += 1

# at episode end:
self.writer.log_episode(reward=ep_reward, length=ep_length,
                        env_step=self.global_env_step,
                        terminated=terminated, truncated=truncated)
```

- [ ] **Step 5: At end of `train()`**

```python
self.writer.assert_contract_satisfied()
self.writer.close()
```

- [ ] **Step 6: Smoke test** (mandatory minimum + LOSS_ACTOR + LOSS_CRITIC).

- [ ] **Step 7: Run existing tests** (`tests/agents/et_dhp_test.py`) — should still pass.

- [ ] **Step 8: Commit**

```bash
git add tensoraerospace/agent/et_dhp/model.py tests/agents/etdhp_metrics_smoke_test.py
git commit -m "feat(etdhp): add TensorBoard logging using unified schema"
```

---

## Task 16: Add logging to GAIL

**Files:**
- Modify: GAIL training file (path identified in Step 1)
- Test: `tests/agents/gail_metrics_smoke_test.py`

GAIL has no TensorBoard logging. Add mandatory minimum plus discriminator/
generator losses and accuracies (`schema.GAIL.*`).

- [ ] **Step 1: Locate the GAIL training file**

Run: `grep -rln "class GAIL\|def train" tensoraerospace/agent/gail/`

Identify the file containing the training loop.

- [ ] **Step 2: Add writer + log losses + log episode + train/lr + train/updates**

Same pattern as Task 15, but use:
- `schema.GAIL.LOSS_DISCRIMINATOR` for discriminator loss
- `schema.GAIL.LOSS_GENERATOR` (or `schema.LOSS_ACTOR` if generator == policy actor) for generator
- `schema.GAIL.EXPERT_ACCURACY` and `schema.GAIL.POLICY_ACCURACY` for discriminator accuracies

- [ ] **Step 3: Smoke test** with required:
```python
REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD, schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS, schema.TRAIN_UPDATES, schema.TRAIN_LR,
    schema.GAIL.LOSS_DISCRIMINATOR, schema.GAIL.EXPERT_ACCURACY,
}
```

- [ ] **Step 4: Run existing GAIL tests** (`tests/agents/gail_model_test.py`).

- [ ] **Step 5: Commit**

```bash
git add tensoraerospace/agent/gail/ tests/agents/gail_metrics_smoke_test.py
git commit -m "feat(gail): add TensorBoard logging using unified schema"
```

---

## Task 17: Documentation

**Files:**
- Create: `docs/en/source/api/metrics.md`
- Create: `docs/ru/source/api/metrics.md` (optional — copy + translate)

- [ ] **Step 1: Write the user-facing metrics reference**

Create `docs/en/source/api/metrics.md`:

```markdown
# Unified TensorBoard Metrics

All RL agents in TensorAeroSpace log to TensorBoard using the canonical
schema defined in `tensoraerospace.agent.metrics.schema`. Group prefixes:

- `rollout/` — per-episode environment statistics (mandatory)
- `loss/` — training losses
- `policy/` — policy / action statistics
- `value/` — value-function statistics
- `diagnostics/` — algorithm-specific diagnostics
- `train/` — training progress counters (mandatory)
- `eval/` — evaluation episode statistics
- `weights/` — network weight histograms
- `grads/` — gradient histograms

X-axis: every metric is logged against the cumulative environment step
(`global_env_step`). Per-episode metrics are written at episode end at
the env step reached at that moment.

## Mandatory minimum (every RL agent)

| Tag | Description |
|---|---|
| `rollout/episode_reward` | Sum of rewards in a finished episode |
| `rollout/episode_length` | Steps in a finished episode |
| `rollout/total_steps` | Cumulative environment interactions |
| `train/updates` | Cumulative gradient updates |
| `train/lr` | Current learning rate |

## Common (logged when applicable)

[List the Tier-2 constants and a one-line description for each.]

## Per-algorithm extras

[For each algorithm class — PPO, SAC, DSAC, DQN, DDPG, A2C, ADP, ADHDP,
GAIL — list the constants and a one-line description.]

## Adding a new algorithm

1. Add a class to `tensoraerospace/agent/metrics/schema.py` with
   `UPPER_SNAKE_CASE` string constants.
2. Reuse common-tier constants (`LOSS_ACTOR`, `LOSS_CRITIC`, etc.) where
   the metric semantics match.
3. The class is automatically picked up by `_build_registry()` if added to
   the `_build_registry` function's tuple of classes.
```

- [ ] **Step 2: Wire the page into the docs nav** (if mkdocs / sphinx config requires explicit listing — check `docs/en/source/conf.py` or `mkdocs.yml`).

- [ ] **Step 3: Commit**

```bash
git add docs/en/source/api/metrics.md
git commit -m "docs: add unified metrics schema reference"
```

---

## Task 18: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -x --timeout=300 -q`

Expected: all tests pass. If any test still references an old metric name,
update it to the canonical name.

- [ ] **Step 2: Search the codebase for stragglers**

Run: `grep -rn "Performance/\|Loss/QF\|Loss/Z\|Loss/DQN\|Performance/Reward\|Performance/Episode\|losses/\|parameters/actor\|gradients/actor\|Train/TotalSteps\|Train/ReplaySize\|Train/Updates\|Diagnostics/Approx KL\|Diagnostics/Clip Fraction\|TD-Error/\|Q/PredSA\|Q/TargetSA\|PER/Beta\|Exploration/Epsilon\|episode_reward.*=" tensoraerospace/`

Expected: no matches in agent files (matches in docs/specs are OK).

- [ ] **Step 3: Confirm no `normalize_tag` / `_METRIC_ALIASES` references remain**

Run: `grep -rn "normalize_tag\|_METRIC_ALIASES\|_GROUP_ALIASES\|ensure_metric_writer" tensoraerospace/ tests/`

Expected: no matches.

- [ ] **Step 4: Final commit (if anything stray was fixed)**

```bash
git status
# if clean: nothing to commit
# else:
git add -A
git commit -m "chore: clean up remaining metric-name stragglers"
```

- [ ] **Step 5: Open PR (when ready)**

Title: `Unify TensorBoard metric naming across all RL agents`
Body: link the spec at `docs/superpowers/specs/2026-04-19-unified-tensorboard-metrics-design.md`.
