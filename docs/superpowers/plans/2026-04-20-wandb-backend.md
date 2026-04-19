# Wandb Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Weights & Biases as a second sink for `MetricWriter` alongside the existing TensorBoard sink, with auto-detection via `WANDB_API_KEY` and per-backend kwargs override.

**Architecture:** Refactor `MetricWriter` into a façade holding `0..N` private `_Sink` objects. `_TensorBoardSink` wraps the existing `SummaryWriter` path; new `_WandbSink` wraps `wandb.init()` + `wandb.log(...)`. Strict-whitelist validation stays in the façade — sinks never validate. Factory `create_metric_writer(...)` auto-activates wandb when `WANDB_API_KEY` is set, or when `wandb_project` is passed explicitly. Each of the 12 RL agents gains five wandb kwargs in `__init__` and forwards them.

**Tech Stack:** Python 3.10+, `wandb` (~0.18+), `torch.utils.tensorboard.SummaryWriter`, `pytest`, `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-04-20-wandb-backend-design.md`

---

## File Structure

**Modified:**
- `pyproject.toml` — add `wandb` to dependencies
- `tensoraerospace/agent/metrics/writer.py` — refactor to sinks; add `_TensorBoardSink`, `_WandbSink`, factory auto-detection
- `tensoraerospace/agent/metrics/__init__.py` — re-export unchanged surface
- `tensoraerospace/agent/a2c/model.py`, `a2c/narx.py`, `a3c/pytorch.py`, `adhdp/model.py`, `adp/adp.py`, `ddpg/model.py`, `dqn/model.py`, `dsac/dsac_flight.py`, `et_dhp/model.py`, `gail/model.py`, `ppo/model.py`, `sac/sac.py` — extend `__init__` with 5 wandb kwargs and forward to `create_metric_writer`
- `tests/agents/metrics_writer_test.py` — rename `log_dir=` → `tb_log_dir=`
- `docs/en/guide/metrics.md` — add "Wandb backend" section

**Created:**
- `tests/agents/metrics_wandb_sink_test.py` — `_WandbSink` unit tests with mocked wandb
- `tests/agents/metrics_factory_test.py` — auto-activation table tests
- `tests/agents/metrics_dual_sink_smoke_test.py` — dual-sink end-to-end
- `tests/agents/a3c_wandb_parent_only_test.py` — multi-worker exclusion test

---

## Task 1: Add wandb to project dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect current dependencies block**

Run: `grep -n -A 30 '\[tool\.poetry\.dependencies\]\|\[project\]' pyproject.toml | head -60`

Identify the dependencies table (poetry-style `[tool.poetry.dependencies]` or PEP 621 `[project] dependencies = [...]`).

- [ ] **Step 2: Add wandb dependency**

If poetry-style:
```toml
[tool.poetry.dependencies]
# ... existing entries ...
wandb = "^0.18"
```

If PEP 621-style:
```toml
[project]
dependencies = [
    # ... existing entries ...
    "wandb>=0.18,<1.0",
]
```

Use whichever style the file already uses; do not migrate styles.

- [ ] **Step 3: Install the dependency**

Run (whichever the project uses):
```bash
.venv/bin/pip install "wandb>=0.18,<1.0"
```
or
```bash
poetry install
```

Verify:
```bash
.venv/bin/python -c "import wandb; print(wandb.__version__)"
```
Expected: prints a version `0.18.x` or `0.19.x` (no error).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
# also commit the lockfile if your project uses one (poetry.lock / uv.lock / etc.)
git add poetry.lock 2>/dev/null || true
git commit -m "deps: add wandb for second metrics sink"
```

---

## Task 2: Extract `_TensorBoardSink` from `MetricWriter` (behavior-preserving refactor)

**Files:**
- Modify: `tensoraerospace/agent/metrics/writer.py`

This task ONLY extracts the existing TB writing code into a `_TensorBoardSink` class and routes `MetricWriter` through it. Behavior is identical. No tests should change.

- [ ] **Step 1: Read current `writer.py`**

Run: `cat tensoraerospace/agent/metrics/writer.py`

Identify:
- `_FallbackSummaryWriter` (no-op summary writer)
- `_get_summary_writer_class()`, `_LazyTorchSummaryWriter`, `TorchSummaryWriter` (lazy import proxy)
- `MetricWriter.__init__` constructs `self._writer` via `TorchSummaryWriter(log_dir=...)`
- `MetricWriter.add_scalar/add_histogram` calls `self._writer.add_scalar/add_histogram`
- `MetricWriter.flush/close` forwards to `self._writer`

- [ ] **Step 2: Add `_TensorBoardSink` class** (insert right above `class MetricWriter:`)

```python
class _TensorBoardSink:
    """Sink that forwards metrics to a torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, log_dir: Optional[Union[str, Path]]) -> None:
        log_path = str(log_dir) if log_dir is not None else None
        self._writer = (
            TorchSummaryWriter(log_dir=log_path)
            if log_path is not None
            else TorchSummaryWriter()
        )

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        self._writer.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        self._writer.add_histogram(tag, values, env_step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
```

- [ ] **Step 3: Refactor `MetricWriter` to hold a list of sinks**

Replace the existing `__init__`, `add_scalar`, `add_histogram`, `flush`, `close` with:

```python
class MetricWriter:
    """SummaryWriter wrapper that enforces the canonical metric schema."""

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        *,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: Optional[str] = None,
    ) -> None:
        self._sinks: list = [_TensorBoardSink(log_dir)]
        self._strict = strict
        self._required = tuple(required)
        self._algo = algo
        self._written: Set[str] = set()

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        if self._strict and not schema.is_registered_scalar(tag):
            raise ValueError(
                f"Unknown metric tag {tag!r}"
                + (f" (algo={self._algo})" if self._algo else "")
                + ". Register it in tensoraerospace.agent.metrics.schema "
                "or construct MetricWriter(strict=False)."
            )
        self._written.add(schema.strip_worker_suffix(tag))
        for sink in self._sinks:
            sink.add_scalar(tag, value, env_step)

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        if self._strict and not schema.is_registered_histogram(tag):
            raise ValueError(
                f"Unknown histogram tag {tag!r}. "
                "Histograms must match weights/<group>/<param> or "
                "grads/<group>/<param> with <group> in "
                f"{sorted(schema.HISTOGRAM_SUBGROUPS)}."
            )
        for sink in self._sinks:
            sink.add_histogram(tag, values, env_step)

    def log_episode(
        self,
        *,
        reward: float,
        length: int,
        env_step: int,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> None:
        self.add_scalar(schema.ROLLOUT_EPISODE_REWARD, float(reward), env_step)
        self.add_scalar(schema.ROLLOUT_EPISODE_LENGTH, int(length), env_step)
        self.add_scalar(schema.ROLLOUT_TOTAL_STEPS, int(env_step), env_step)
        if terminated is not None:
            self.add_scalar(schema.DIAG_TERMINATED_COUNT, int(bool(terminated)), env_step)
        if truncated is not None:
            self.add_scalar(schema.DIAG_TRUNCATED_COUNT, int(bool(truncated)), env_step)

    def assert_contract_satisfied(self) -> None:
        check_contract(self._written, self._required)

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
```

- [ ] **Step 4: Run existing test suite to confirm no behavior change**

```bash
.venv/bin/python -m pytest tests/agents/metrics_schema_test.py tests/agents/metrics_writer_test.py tests/agents/metrics_contract_smoke_test.py -v
```
Expected: 7 + 12 + 1 = 20 PASSED.

- [ ] **Step 5: Run full agent suite (sanity)**

```bash
.venv/bin/python -m pytest tests/agents/ --tb=short -q
```
Expected: 985 PASSED.

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/metrics/writer.py
git commit -m "refactor(metrics): extract _TensorBoardSink from MetricWriter

No behavior change. Prepares for dual-sink architecture in subsequent
commits."
```

---

## Task 3: Add `_WandbSink` class (TDD)

**Files:**
- Modify: `tensoraerospace/agent/metrics/writer.py`
- Create: `tests/agents/metrics_wandb_sink_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/metrics_wandb_sink_test.py`:

```python
"""Behaviour of tensoraerospace.agent.metrics.writer._WandbSink."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tensoraerospace.agent.metrics import writer as writer_mod
from tensoraerospace.agent.metrics.writer import _WandbSink


@pytest.fixture
def mock_wandb(monkeypatch):
    """Provide a fresh mock for the `wandb` module with `Settings`, `init`,
    `login`, `Histogram`, `finish`."""
    mock = MagicMock()
    mock.Histogram = MagicMock(side_effect=lambda v: ("HISTOGRAM", v))
    mock.Settings = MagicMock(return_value="SETTINGS_OBJ")
    mock_run = MagicMock()
    mock.init = MagicMock(return_value=mock_run)
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock, mock_run


def test_init_calls_wandb_init_with_kwargs(mock_wandb, monkeypatch):
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    sink = _WandbSink(
        project="proj-1",
        entity="ent-1",
        run_name="run-1",
        tags=["sac", "pendulum"],
        config={"lr": 1e-3},
    )

    mock.login.assert_not_called()
    mock.init.assert_called_once()
    kwargs = mock.init.call_args.kwargs
    assert kwargs["project"] == "proj-1"
    assert kwargs["entity"] == "ent-1"
    assert kwargs["name"] == "run-1"
    assert kwargs["tags"] == ["sac", "pendulum"]
    assert kwargs["config"] == {"lr": 1e-3}
    assert kwargs["reinit"] is True
    assert kwargs["settings"] == "SETTINGS_OBJ"


def test_init_calls_login_when_api_key_missing(mock_wandb, monkeypatch):
    mock, _ = mock_wandb
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    mock.login.assert_called_once()
    mock.init.assert_called_once()


def test_add_scalar_calls_wandb_log(mock_wandb, monkeypatch):
    mock, _ = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)
    mock.log.reset_mock()

    sink.add_scalar("loss/actor", 0.123, env_step=10)

    mock.log.assert_called_once_with({"loss/actor": 0.123}, step=10)


def test_add_histogram_wraps_in_wandb_histogram(mock_wandb, monkeypatch):
    mock, _ = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)
    mock.log.reset_mock()

    values = np.zeros(8)
    sink.add_histogram("weights/actor/fc1", values, env_step=5)

    mock.Histogram.assert_called_once_with(values)
    mock.log.assert_called_once_with(
        {"weights/actor/fc1": ("HISTOGRAM", values)}, step=5
    )


def test_close_calls_finish_once(mock_wandb, monkeypatch):
    mock, mock_run = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    sink.close()
    sink.close()  # second call must be a no-op

    mock_run.finish.assert_called_once()


def test_flush_is_noop(mock_wandb, monkeypatch):
    mock, _ = mock_wandb
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    sink = _WandbSink(project="p", entity=None, run_name=None, tags=None, config=None)

    # Should not raise; nothing to assert on the mock besides no exception.
    sink.flush()
```

- [ ] **Step 2: Run the tests, expect collection error (no `_WandbSink` yet)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_wandb_sink_test.py -v
```
Expected: ImportError or collection failure.

- [ ] **Step 3: Add `_WandbSink` and the `wandb` module-level import to `writer.py`**

At the top of `tensoraerospace/agent/metrics/writer.py`, add (alongside existing imports):

```python
import os
import wandb  # type: ignore[import-untyped]
```

Add the class right after `_TensorBoardSink`:

```python
class _WandbSink:
    """Sink that forwards metrics to Weights & Biases."""

    def __init__(
        self,
        *,
        project: str,
        entity: Optional[str],
        run_name: Optional[str],
        tags: Optional[Sequence[str]],
        config: Optional[Mapping[str, Any]],
    ) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            wandb.login()
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags) if tags else None,
            config=dict(config) if config else None,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

    def add_scalar(self, tag: str, value: float, env_step: int) -> None:
        wandb.log({tag: float(value)}, step=int(env_step))

    def add_histogram(self, tag: str, values, env_step: int) -> None:
        wandb.log({tag: wandb.Histogram(values)}, step=int(env_step))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
```

Update the `from typing import` line to include `Any, Mapping, Sequence` if not already present.

- [ ] **Step 4: Run the tests, expect PASS**

```bash
.venv/bin/python -m pytest tests/agents/metrics_wandb_sink_test.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Run full metrics test suite (sanity)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_*.py -v
```
Expected: all PASS (the new 6 + previously-passing 20 = 26).

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/metrics/writer.py tests/agents/metrics_wandb_sink_test.py
git commit -m "feat(metrics): add _WandbSink for wandb backend

Mirrors _TensorBoardSink interface. Calls wandb.login() interactively
when WANDB_API_KEY is unset. Uses thread start_method to stay safe under
multiprocessing.fork (relevant for A3C). Histograms wrapped via
wandb.Histogram."
```

---

## Task 4: Add wandb kwargs to `MetricWriter.__init__` (TDD)

**Files:**
- Modify: `tensoraerospace/agent/metrics/writer.py`
- Modify: `tests/agents/metrics_writer_test.py` (rename `log_dir=` → `tb_log_dir=`, add new tests)

- [ ] **Step 1: Update `metrics_writer_test.py` to use the renamed kwarg**

In `tests/agents/metrics_writer_test.py`, find the fixture:

```python
@pytest.fixture
def writer(tmp_path: Path) -> MetricWriter:
    return MetricWriter(log_dir=str(tmp_path / "tb"), algo="test")
```

and change to:

```python
@pytest.fixture
def writer(tmp_path: Path) -> MetricWriter:
    return MetricWriter(tb_log_dir=str(tmp_path / "tb"), algo="test")
```

Also update `test_add_scalar_strict_false_allows_anything`:
```python
def test_add_scalar_strict_false_allows_anything(tmp_path: Path):
    w = MetricWriter(tb_log_dir=str(tmp_path / "tb"), strict=False)
    w.add_scalar("anything/at/all", 1.0, env_step=1)
```

- [ ] **Step 2: Add new tests for wandb kwargs in the same file**

Append to `tests/agents/metrics_writer_test.py`:

```python
def test_metricwriter_with_wandb_project_creates_two_sinks(
    tmp_path: Path, monkeypatch
):
    """When both tb_log_dir and wandb_project are passed, both sinks are active."""
    from unittest.mock import MagicMock
    from tensoraerospace.agent.metrics import writer as writer_mod

    mock_wandb = MagicMock()
    mock_wandb.init = MagicMock(return_value=MagicMock())
    mock_wandb.Settings = MagicMock(return_value="S")
    monkeypatch.setattr(writer_mod, "wandb", mock_wandb, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    w = MetricWriter(
        tb_log_dir=str(tmp_path / "tb"),
        wandb_project="exp",
        algo="test",
    )
    assert len(w._sinks) == 2  # noqa: SLF001 — internal but stable for tests


def test_metricwriter_no_kwargs_creates_zero_sinks(monkeypatch):
    """No tb_log_dir and no wandb activation => zero sinks (writer no-ops)."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = MetricWriter(algo="test")
    assert len(w._sinks) == 0
    # whitelist still enforced
    w.add_scalar(schema.LOSS_ACTOR, 0.5, env_step=1)
    with pytest.raises(ValueError):
        w.add_scalar("not/in/schema", 0.5, env_step=1)
```

- [ ] **Step 3: Run tests, expect failures (writer doesn't accept new kwargs yet)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_writer_test.py -v
```
Expected: at least the two new tests fail (TypeError on `wandb_project`); existing 12 may also fail because `tb_log_dir=` is unknown.

- [ ] **Step 4: Update `MetricWriter.__init__` signature and sink construction**

In `tensoraerospace/agent/metrics/writer.py`, replace `MetricWriter.__init__` with:

```python
    def __init__(
        self,
        tb_log_dir: Optional[Union[str, Path]] = None,
        *,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: Optional[str] = None,
    ) -> None:
        self._sinks: list = []
        if tb_log_dir is not None:
            self._sinks.append(_TensorBoardSink(tb_log_dir))
        if wandb_project is not None:
            self._sinks.append(
                _WandbSink(
                    project=wandb_project,
                    entity=wandb_entity,
                    run_name=wandb_run_name,
                    tags=wandb_tags,
                    config=wandb_config,
                )
            )
        self._strict = strict
        self._required = tuple(required)
        self._algo = algo
        self._written: Set[str] = set()
```

Note: the auto-detection (env-var) logic lives in the **factory**, not in `MetricWriter.__init__` — `__init__` is explicit-only. Auto-detection is added in Task 5.

- [ ] **Step 5: Run tests, expect PASS**

```bash
.venv/bin/python -m pytest tests/agents/metrics_writer_test.py -v
```
Expected: 14 PASSED (12 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/metrics/writer.py tests/agents/metrics_writer_test.py
git commit -m "feat(metrics): MetricWriter accepts wandb kwargs and builds sinks list

Renames first positional kwarg log_dir -> tb_log_dir. Adds five wandb
kwargs (project/entity/run_name/tags/config). When tb_log_dir or
wandb_project is passed, the corresponding sink is added; otherwise the
writer no-ops on logging while still enforcing the strict whitelist."
```

---

## Task 5: Update `create_metric_writer` factory with auto-detection (TDD)

**Files:**
- Modify: `tensoraerospace/agent/metrics/writer.py`
- Create: `tests/agents/metrics_factory_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/metrics_factory_test.py`:

```python
"""Auto-activation behaviour of create_metric_writer."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensoraerospace.agent.metrics import writer as writer_mod
from tensoraerospace.agent.metrics import (
    MetricWriter,
    create_metric_writer,
    schema,
)


@pytest.fixture
def mock_wandb(monkeypatch):
    mock = MagicMock()
    mock.init = MagicMock(return_value=MagicMock())
    mock.Settings = MagicMock(return_value="S")
    mock.Histogram = MagicMock(side_effect=lambda v: ("H", v))
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock


def test_tb_only_when_only_tb_log_dir(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = create_metric_writer(tb_log_dir=str(tmp_path / "tb"), algo="sac")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_not_called()


def test_wandb_only_when_only_api_key(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(algo="sac")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_called_once()
    # default project is the algo
    assert mock_wandb.init.call_args.kwargs["project"] == "sac"


def test_both_when_tb_log_dir_and_api_key(tmp_path: Path, monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(tb_log_dir=str(tmp_path / "tb"), algo="ppo")
    assert len(w._sinks) == 2
    mock_wandb.init.assert_called_once()


def test_zero_sinks_when_nothing(monkeypatch, mock_wandb):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    w = create_metric_writer(algo="sac")
    assert len(w._sinks) == 0
    # whitelist still enforced
    with pytest.raises(ValueError):
        w.add_scalar("not/in/schema", 1.0, env_step=1)


def test_explicit_wandb_project_calls_login_when_api_key_missing(
    monkeypatch, mock_wandb
):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    create_metric_writer(wandb_project="my-exp", algo="sac")
    mock_wandb.login.assert_called_once()
    mock_wandb.init.assert_called_once()
    assert mock_wandb.init.call_args.kwargs["project"] == "my-exp"


def test_default_project_uses_algo_when_only_api_key(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="ddpg")
    assert mock_wandb.init.call_args.kwargs["project"] == "ddpg"


def test_default_project_falls_back_when_no_algo(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer()
    assert mock_wandb.init.call_args.kwargs["project"] == "tensoraerospace"


def test_default_run_name_includes_algo_and_timestamp(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="sac")
    name = mock_wandb.init.call_args.kwargs["name"]
    assert name.startswith("sac-")
    assert len(name) > len("sac-")


def test_default_tags_contain_algo(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(algo="dqn")
    tags = mock_wandb.init.call_args.kwargs["tags"]
    assert "dqn" in tags


def test_explicit_kwargs_override_defaults(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    create_metric_writer(
        algo="sac",
        wandb_project="p",
        wandb_entity="e",
        wandb_run_name="r",
        wandb_tags=["t1", "t2"],
        wandb_config={"k": "v"},
    )
    kw = mock_wandb.init.call_args.kwargs
    assert kw["project"] == "p"
    assert kw["entity"] == "e"
    assert kw["name"] == "r"
    assert kw["tags"] == ["t1", "t2"]
    assert kw["config"] == {"k": "v"}
```

- [ ] **Step 2: Run tests, expect failures**

```bash
.venv/bin/python -m pytest tests/agents/metrics_factory_test.py -v
```
Expected: most fail (factory doesn't auto-detect yet).

- [ ] **Step 3: Update `create_metric_writer` factory**

In `tensoraerospace/agent/metrics/writer.py`, replace `create_metric_writer` with:

```python
import datetime as _dt


def create_metric_writer(
    tb_log_dir: Optional[Union[str, Path]] = None,
    *,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_config: Optional[Mapping[str, Any]] = None,
    strict: bool = True,
    algo: Optional[str] = None,
) -> MetricWriter:
    """Construct a MetricWriter with auto-detection of the wandb backend.

    Activation rules:
      * tb_log_dir set                          -> TB sink active.
      * wandb_project set                       -> wandb sink active (calls
                                                   wandb.login() if no key).
      * WANDB_API_KEY in env, no wandb_project  -> wandb sink active with
                                                   project = algo (or
                                                   "tensoraerospace").

    In multi-worker (forked) processes, wandb is skipped — see
    docs/superpowers/specs/2026-04-20-wandb-backend-design.md.
    """
    import multiprocessing

    is_main_process = multiprocessing.current_process().name == "MainProcess"

    # Resolve wandb activation
    wandb_enabled = is_main_process and (
        wandb_project is not None or os.environ.get("WANDB_API_KEY") is not None
    )

    if wandb_enabled:
        resolved_project = wandb_project or algo or "tensoraerospace"
        if wandb_run_name is None:
            ts = _dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            wandb_run_name = f"{algo or 'run'}-{ts}"
        if wandb_tags is None and algo is not None:
            wandb_tags = [algo]
    else:
        resolved_project = None

    return MetricWriter(
        tb_log_dir=tb_log_dir,
        wandb_project=resolved_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_config=wandb_config,
        strict=strict,
        algo=algo,
    )
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
.venv/bin/python -m pytest tests/agents/metrics_factory_test.py -v
```
Expected: 10 PASSED.

- [ ] **Step 5: Run full metrics test suite (sanity)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_*.py -v
```
Expected: 36 PASSED (existing 26 + new 10).

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/metrics/writer.py tests/agents/metrics_factory_test.py
git commit -m "feat(metrics): add wandb auto-detection to create_metric_writer

WANDB_API_KEY in env auto-activates wandb sink (project = algo).
wandb_project= forces wandb on regardless of env. Workers (non-main
process) skip wandb. Default run name = algo + timestamp; default tags
= [algo]."
```

---

## Task 6: Add A3C parent-only behaviour test

**Files:**
- Create: `tests/agents/a3c_wandb_parent_only_test.py`

The factory's parent-only logic was added in Task 5; this test pins that behaviour.

- [ ] **Step 1: Write the test**

Create `tests/agents/a3c_wandb_parent_only_test.py`:

```python
"""Multi-worker guard: workers must not initialize wandb."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tensoraerospace.agent.metrics import create_metric_writer
from tensoraerospace.agent.metrics import writer as writer_mod


@pytest.fixture
def mock_wandb(monkeypatch):
    mock = MagicMock()
    mock.init = MagicMock(return_value=MagicMock())
    mock.Settings = MagicMock(return_value="S")
    monkeypatch.setattr(writer_mod, "wandb", mock, raising=False)
    return mock


def test_in_main_process_wandb_init_runs(monkeypatch, mock_wandb):
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    w = create_metric_writer(algo="a3c")
    assert len(w._sinks) == 1
    mock_wandb.init.assert_called_once()


def test_in_worker_process_wandb_skipped(monkeypatch, mock_wandb):
    """When not running as MainProcess, wandb sink is omitted."""
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    fake_proc = MagicMock()
    fake_proc.name = "Worker-1"
    with patch("multiprocessing.current_process", return_value=fake_proc):
        w = create_metric_writer(algo="a3c")

    assert len(w._sinks) == 0
    mock_wandb.init.assert_not_called()


def test_in_worker_process_explicit_wandb_project_still_skipped(
    monkeypatch, mock_wandb
):
    """Even an explicit wandb_project doesn't override the worker guard."""
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    fake_proc = MagicMock()
    fake_proc.name = "Worker-2"
    with patch("multiprocessing.current_process", return_value=fake_proc):
        w = create_metric_writer(wandb_project="exp", algo="a3c")
    assert len(w._sinks) == 0
    mock_wandb.init.assert_not_called()
```

- [ ] **Step 2: Run the test**

```bash
.venv/bin/python -m pytest tests/agents/a3c_wandb_parent_only_test.py -v
```
Expected: 3 PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/a3c_wandb_parent_only_test.py
git commit -m "test(metrics): pin parent-only wandb behaviour for A3C workers"
```

---

## Task 7: Add dual-sink end-to-end smoke test

**Files:**
- Create: `tests/agents/metrics_dual_sink_smoke_test.py`

- [ ] **Step 1: Write the test**

Create `tests/agents/metrics_dual_sink_smoke_test.py`:

```python
"""Dual-sink end-to-end: SAC.train() writes to both TB and (mocked) wandb."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tensoraerospace.agent.metrics import writer as writer_mod
from tensoraerospace.agent.metrics import schema
from tests.agents.metrics_contract_smoke_test import assert_tags_present


REQUIRED = {
    schema.ROLLOUT_EPISODE_REWARD,
    schema.ROLLOUT_EPISODE_LENGTH,
    schema.ROLLOUT_TOTAL_STEPS,
    schema.TRAIN_UPDATES,
    schema.TRAIN_LR,
    schema.TRAIN_REPLAY_SIZE,
    schema.SAC.LOSS_Q1,
    schema.LOSS_POLICY,
}


@pytest.mark.timeout(120)
def test_sac_writes_to_both_tb_and_wandb(tmp_path: Path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    pytest.importorskip("tensorboard")

    import gymnasium as gym
    from tensoraerospace.agent.sac.sac import SAC

    # Mock wandb so the test doesn't hit the network
    mock_wandb = MagicMock()
    mock_wandb.init = MagicMock(return_value=MagicMock())
    mock_wandb.Settings = MagicMock(return_value="S")
    mock_wandb.Histogram = MagicMock(side_effect=lambda v: ("H", v))
    monkeypatch.setattr(writer_mod, "wandb", mock_wandb, raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "fake")

    env = gym.make("Pendulum-v1")
    log_dir = tmp_path / "tb"

    agent = SAC(
        env=env,
        batch_size=16,
        memory_capacity=2_000,
        log_dir=str(log_dir),
        wandb_project="dual-sink-test",
        device="cpu",
    )
    agent.train(num_episodes=1, max_steps=64, verbose=False)
    agent.writer.flush()

    # TB side: tags appear in event files
    assert_tags_present(str(log_dir), REQUIRED)

    # Wandb side: the same tags appear in mocked wandb.log calls
    logged_keys = set()
    for call in mock_wandb.log.call_args_list:
        payload = call.args[0] if call.args else call.kwargs.get("data", {})
        logged_keys.update(payload.keys())
    missing = REQUIRED - logged_keys
    assert not missing, (
        f"Tags missing from wandb.log calls: {sorted(missing)}; "
        f"got {sorted(logged_keys)}"
    )
```

- [ ] **Step 2: Run the test, expect failure (SAC doesn't accept wandb_project yet)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_dual_sink_smoke_test.py -v
```
Expected: TypeError on `wandb_project=` in `SAC(...)`.

This will be fixed in Task 8 (when SAC ctor is extended). Leave the test as-is.

- [ ] **Step 3: Commit the test (red state — it will go green after Task 8)**

```bash
git add tests/agents/metrics_dual_sink_smoke_test.py
git commit -m "test(metrics): add dual-sink smoke test for SAC

Currently red — turns green when SAC.__init__ gains wandb_project=
in Task 8."
```

---

## Task 8: Extend all 12 RL agent constructors with wandb kwargs

**Files (modify):**
- `tensoraerospace/agent/a2c/model.py`
- `tensoraerospace/agent/a2c/narx.py`
- `tensoraerospace/agent/a3c/pytorch.py`
- `tensoraerospace/agent/adhdp/model.py`
- `tensoraerospace/agent/adp/adp.py`
- `tensoraerospace/agent/ddpg/model.py`
- `tensoraerospace/agent/dqn/model.py` (both `DQNAgent` and `PERNARXAgent`)
- `tensoraerospace/agent/dsac/dsac_flight.py`
- `tensoraerospace/agent/et_dhp/model.py`
- `tensoraerospace/agent/gail/model.py`
- `tensoraerospace/agent/ppo/model.py`
- `tensoraerospace/agent/sac/sac.py`

The change pattern is identical for each: add five kwargs to `__init__`, store as instance attributes, forward to `create_metric_writer`. Apply mechanically to every file.

- [ ] **Step 1: Add a typing import where missing**

In each agent file, ensure these are importable:
```python
from typing import Any, Mapping, Optional, Sequence
```
(Most files already import some of these. Add only what's missing — do not duplicate.)

- [ ] **Step 2: Add five kwargs to each agent's `__init__`**

For every agent listed above, find the `__init__` signature and add these five keyword-only parameters at the end (or alongside the existing `log_dir` if any):

```python
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_tags: Optional[Sequence[str]] = None,
        wandb_config: Optional[Mapping[str, Any]] = None,
```

**Per-agent guidance for the `create_metric_writer` call site:**

For each agent, find the existing call:
```python
self.writer = create_metric_writer(<positional log_dir>, algo="<name>")
```
(or `create_metric_writer(self.log_dir, algo="<name>")`).

Replace with:
```python
self.writer = create_metric_writer(
    tb_log_dir=<existing log_dir argument>,
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
    wandb_run_name=wandb_run_name,
    wandb_tags=wandb_tags,
    wandb_config=wandb_config,
    algo="<name>",
)
```

**Specific files and call sites (verified):**

- `tensoraerospace/agent/a2c/model.py:359` — `create_metric_writer(log_dir, algo="a2c")` → `create_metric_writer(tb_log_dir=log_dir, ...)`
- `tensoraerospace/agent/a2c/narx.py:252` — `create_metric_writer(log_dir, algo="a2c-narx")` → forward kwargs
- `tensoraerospace/agent/a3c/pytorch.py:499` — `create_metric_writer(log_dir, algo="a3c")` (inside `Agent.__init__`)
- `tensoraerospace/agent/adhdp/model.py:453` — `create_metric_writer(self.log_dir, algo="adhdp")`
- `tensoraerospace/agent/adp/adp.py:476` — `create_metric_writer(self.log_dir, algo="adp")`
- `tensoraerospace/agent/ddpg/model.py:899` — `create_metric_writer(logdir, algo="ddpg")` (inside `learn()` lazy init; the kwargs need to come from `self.<wandb_*>` set in `__init__`)
- `tensoraerospace/agent/dqn/model.py:292` (DQNAgent) and another at line 928 (PERNARXAgent) — both `create_metric_writer(self.log_dir, algo="dqn"|"dqn-narx")`
- `tensoraerospace/agent/dsac/dsac_flight.py:131` — `create_metric_writer(self.log_dir, algo="dsac")`
- `tensoraerospace/agent/et_dhp/model.py` — `create_metric_writer(self.log_dir, algo="etdhp")` (gated `if log_dir is not None`)
- `tensoraerospace/agent/gail/model.py` — `create_metric_writer(self.log_dir, algo="gail")` (gated)
- `tensoraerospace/agent/ppo/model.py` — `create_metric_writer(self.log_dir, algo="ppo")` (around line 622)
- `tensoraerospace/agent/sac/sac.py:104` — `create_metric_writer(self.log_dir, algo="sac")`

For each agent, the pattern after the change in `__init__`:
```python
self.wandb_project = wandb_project
self.wandb_entity = wandb_entity
self.wandb_run_name = wandb_run_name
self.wandb_tags = wandb_tags
self.wandb_config = wandb_config
self.writer = create_metric_writer(
    tb_log_dir=self.log_dir,  # or whatever local var holds the path
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
    wandb_run_name=wandb_run_name,
    wandb_tags=wandb_tags,
    wandb_config=wandb_config,
    algo="<name>",
)
```

**Special cases:**

- **ET-DHP / GAIL:** writer is gated `if log_dir is not None`. Change the gate to `if log_dir is not None or wandb_project is not None or os.environ.get("WANDB_API_KEY"):`. (Add `import os` if missing.)
- **DDPG:** writer is lazily created inside `learn()`. Move the wandb kwargs into instance attrs in `__init__` (`self.wandb_project = wandb_project`, etc.), then in `learn()` use `self.wandb_project` etc. when constructing the writer.

- [ ] **Step 3: Run the dual-sink smoke test (now expected to pass)**

```bash
.venv/bin/python -m pytest tests/agents/metrics_dual_sink_smoke_test.py -v
```
Expected: PASSED.

- [ ] **Step 4: Run all 12 per-agent smoke tests (no regression)**

```bash
.venv/bin/python -m pytest tests/agents/*_metrics_smoke_test.py -v
```
Expected: 13 PASSED (12 existing + 1 dual-sink).

- [ ] **Step 5: Run full agent suite**

```bash
.venv/bin/python -m pytest tests/agents/ --tb=short -q
```
Expected: previous count + new metrics tests (was 985; now ≈ 985 + 19 new metrics tests).

- [ ] **Step 6: Commit**

```bash
git add tensoraerospace/agent/
git commit -m "feat(agents): forward wandb kwargs from agent ctor to create_metric_writer

All 12 RL agents (A2C, A2C-NARX, A3C, ADHDP, ADP, DDPG, DQN x2, DSAC,
ET-DHP, GAIL, PPO, SAC) now accept wandb_project/entity/run_name/tags/
config in __init__ and forward them to the metrics factory.

ET-DHP and GAIL gate writer construction on either log_dir or wandb
activation. DDPG stores wandb kwargs as instance attrs because writer
is lazily created in learn()."
```

---

## Task 9: Update the metrics docs page

**Files:**
- Modify: `docs/en/guide/metrics.md`

- [ ] **Step 1: Read the existing page**

Run: `cat docs/en/guide/metrics.md | head -80`

Identify the right place to insert a new section. A good location is right after the "Histogram convention" section, or as a top-level "Backends" section near the end, before "Adding a new algorithm".

- [ ] **Step 2: Add a "Wandb backend" section**

Append (or insert before "Adding a new algorithm"):

```markdown
## Wandb backend

`MetricWriter` supports Weights & Biases as a second sink alongside
TensorBoard. Both can be active at the same time — every `add_scalar` /
`add_histogram` / `log_episode` call fans out to whichever sinks are
enabled.

### When wandb is enabled

| `tb_log_dir` | `wandb_project` | `WANDB_API_KEY` | TB | wandb |
|---|---|---|---|---|
| set | — | — | ✅ | — |
| set | — | set | ✅ | ✅ (project = `algo`) |
| — | — | set | — | ✅ (project = `algo`) |
| set | set | * | ✅ | ✅ (project as given) |
| — | set | unset | — | ✅ (calls `wandb.login()` interactively) |
| — | — | unset | — | — |

### Example

```python
from tensoraerospace.agent.sac.sac import SAC

# TensorBoard only (default)
agent = SAC(env=env, log_dir="runs/sac")

# Wandb only — set WANDB_API_KEY in env, then:
agent = SAC(env=env, wandb_project="my-experiment", wandb_tags=["sac", "pendulum"])

# Both backends in parallel
agent = SAC(
    env=env,
    log_dir="runs/sac",
    wandb_project="my-experiment",
    wandb_config={"lr": 3e-4, "tau": 5e-3},
)
```

### A3C limitation

A3C runs workers in forked processes. The wandb sink is initialized only
in the main process — workers continue to share the parent's TensorBoard
event file via `/worker_<id>` suffix. To get per-worker wandb runs, use
the launch-N-jobs pattern (one wandb run per worker process started
externally) instead of `Agent.train()`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/en/guide/metrics.md
git commit -m "docs: add Wandb backend section to metrics reference"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ --tb=short
```
Expected: all tests pass (≈ 1724 + new metrics tests).

- [ ] **Step 2: Verify no stragglers**

Run: `grep -rn "log_dir=" tensoraerospace/agent/metrics/` should return no matches inside the metrics package (only `tb_log_dir=` remains).

Run: `grep -rn "create_metric_writer(" tensoraerospace/agent/ --include="*.py"` should show every call site uses keyword args, not bare positional.

- [ ] **Step 3: Manual smoke test against real wandb (optional, only if `WANDB_API_KEY` is available)**

```bash
export WANDB_API_KEY=<your-key>
.venv/bin/python -c "
import gymnasium as gym
from tensoraerospace.agent.sac.sac import SAC

env = gym.make('Pendulum-v1')
agent = SAC(
    env=env,
    log_dir='runs/sac-wandb-manual',
    wandb_project='tensoraero-smoke',
    batch_size=16,
    memory_capacity=2000,
    device='cpu',
)
agent.train(num_episodes=1, max_steps=32, verbose=False)
agent.writer.close()
print('Run uploaded — check https://wandb.ai/<entity>/tensoraero-smoke')
"
```

If a wandb key is not available, skip this step.

- [ ] **Step 4: Final commit (if anything was fixed in Step 2/3)**

```bash
git status
# if clean: nothing to do
# else:
git add -A
git commit -m "chore: cleanup stragglers from wandb backend integration"
```

- [ ] **Step 5: Open PR**

```bash
git push -u origin feature/wandb
gh pr create --base develop --title "Add Wandb backend support for MetricWriter" --body "$(cat <<'EOF'
## Summary

Adds Weights & Biases as a second sink for `MetricWriter` alongside TensorBoard. Both can run in parallel.

## How it activates

- `WANDB_API_KEY` in env → wandb sink turns on automatically (project = `algo`).
- `wandb_project=` kwarg → wandb sink turns on regardless; calls `wandb.login()` interactively if no key is set.
- Per-agent kwargs: `wandb_project`, `wandb_entity`, `wandb_run_name`, `wandb_tags`, `wandb_config` on every RL agent's `__init__`.

## Closes

#176

## Test plan

- 6 new `_WandbSink` unit tests (mocked wandb).
- 10 new factory auto-detection tests.
- 3 new A3C parent-only tests.
- 1 new dual-sink end-to-end smoke test.
- All existing 985 agent tests continue to pass.
EOF
)"
```

---

## Self-review (planner notes)

**Spec coverage:** every requirement in `2026-04-20-wandb-backend-design.md` maps to a task:
- Sink protocol → Tasks 2, 3
- `create_metric_writer` auto-detection → Task 5
- A3C parent-only → Task 5 (factory) + Task 6 (test)
- 12-agent ctor extension → Task 8
- `pyproject.toml` dep → Task 1
- Documentation → Task 9
- Tests (4 new files + 1 updated) → Tasks 3, 5, 6, 7

**No placeholders.** All code blocks are complete; commands are exact.

**Type consistency:** `_WandbSink.__init__` signature `(*, project, entity, run_name, tags, config)` matches what `MetricWriter.__init__` passes (Task 4) and what factory passes (Task 5). `MetricWriter.__init__` keyword names (`wandb_project`, `wandb_entity`, etc.) match the factory's kwargs (Task 5) match the agent ctors' kwargs (Task 8).
