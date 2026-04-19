# Wandb Backend for MetricWriter — Design

**Date:** 2026-04-20
**Status:** Approved (brainstorming)
**Scope:** Add Weights & Biases (`wandb`) as a second sink for `MetricWriter`, alongside the existing TensorBoard sink.
**Related issue:** #176 (kept out of scope of #215; this spec closes it.)

---

## Problem

The unified `MetricWriter` (introduced in #215) writes scalars, histograms,
and episode summaries to TensorBoard event files via `SummaryWriter`. Many
users want the same metrics in Weights & Biases at the same time — for
team-shared experiment tracking and cross-run dashboards. Today there is no
wandb integration; users would have to wrap every `add_scalar` call manually.

## Goal

Let any agent write metrics to wandb in addition to (or instead of)
TensorBoard, with the same canonical schema, the same `env_step` X-axis,
and the same strict-whitelist guarantees, by changing only the writer
construction call.

## Non-goals

- Other backends (MLflow, CSV, console). The `_Sink` protocol leaves the
  door open, but no implementations beyond TB and wandb in this spec.
- Sweeps / hyperparameter optimization (orthogonal — wandb-sweep can
  consume wandb runs without further work).
- wandb Artifacts (model checkpoints). Out of scope; agent `save()` paths
  remain unchanged.
- Running wandb in *offline* mode by default. Users who need offline mode
  set `WANDB_MODE=offline` themselves; we don't add a switch.

---

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Strategy | Dual-sink. TB and wandb run in parallel; either may be off. |
| Auto-detection | If `WANDB_API_KEY` is set in the environment, wandb-sink turns on automatically (and uses `algo` as the project name when none is supplied). |
| Explicit override | Per-backend kwargs on `create_metric_writer(...)`: `tb_log_dir=...`, `wandb_project=...`, etc. Passing `tb_log_dir` forces TB on; passing `wandb_project` forces wandb on. |
| Missing `WANDB_API_KEY` when wandb is requested | Call `wandb.login()` (interactive prompt). In CI, the wandb library itself raises a clear error. |
| Wandb config surface | Basic set: `project, entity, run_name, tags, config` + `algo` (already exists). |
| Architecture | `MetricWriter` as a façade with `0..N` private `_Sink` objects. Strict-whitelist validation stays in the façade — sinks never validate. |
| A3C multi-worker | wandb-sink is parent-only. Workers (forked) skip wandb-init even if `WANDB_API_KEY` is set; they keep writing to the parent's TB writer (already shared via fork). |
| Optional dep | `wandb` is added as a regular dependency in `pyproject.toml` (no `[wandb]` extra). Plain `ImportError` surfaces if it is somehow missing. |

---

## Architecture

### Sink protocol

```python
class _Sink(Protocol):
    def add_scalar(self, tag: str, value: float, env_step: int) -> None: ...
    def add_histogram(self, tag: str, values, env_step: int) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...
```

`MetricWriter` holds `self._sinks: list[_Sink]`. Each public method does:

1. Strict-whitelist check (as today).
2. Track in `self._written` (as today).
3. Fan-out: `for sink in self._sinks: sink.<method>(...)`.

The public surface of `MetricWriter` does not change in shape — only the
constructor signature gains wandb kwargs and `log_dir` is renamed to
`tb_log_dir` (with a positional-compat shim so existing call sites work).

### Module layout

```
tensoraerospace/agent/metrics/
├── __init__.py        # public re-exports (unchanged surface)
├── schema.py          # unchanged
├── contract.py        # unchanged
├── writer.py          # MetricWriter façade + _TensorBoardSink + _WandbSink
└── _sinks.py          # if writer.py grows past ~350 lines, split here
```

### MetricWriter constructor

```python
class MetricWriter:
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
        ...
```

`create_metric_writer(...)` mirrors the same kwargs and is the agent-facing
factory.

### Auto-activation rules

| `tb_log_dir` | `wandb_project` | `WANDB_API_KEY` | TB | wandb |
|---|---|---|---|---|
| set | — | — | ✅ | — |
| set | — | set | ✅ | ✅ (project = `algo` or `"tensoraerospace"`) |
| — | — | set | — | ✅ (project = `algo` or `"tensoraerospace"`) |
| set | set | * | ✅ | ✅ (project as given) |
| — | set | unset | — | ✅ (calls `wandb.login()` interactively) |
| — | — | unset | — | — (writer is no-op) |

### Default values when wandb-sink turns on

- `wandb_run_name` → `f"{algo or 'run'}-{YYYY-MM-DD-HH-MM-SS}"`
- `wandb_tags` → `[algo]` (extendable per agent)
- `wandb_project` → `algo` if `WANDB_API_KEY` present and project not given
- `wandb_config` → `None` (agents may pass their hyperparameters)

---

## `_WandbSink` behaviour

```python
class _WandbSink:
    def __init__(self, *, project, entity, run_name, tags, config):
        import wandb
        self._wandb = wandb
        if not os.environ.get("WANDB_API_KEY"):
            wandb.login()  # interactive prompt
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=list(tags) if tags else None,
            config=dict(config) if config else None,
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

    def add_scalar(self, tag, value, env_step):
        self._wandb.log({tag: float(value)}, step=int(env_step))

    def add_histogram(self, tag, values, env_step):
        self._wandb.log({tag: self._wandb.Histogram(values)}, step=int(env_step))

    def flush(self):
        pass  # wandb buffers internally

    def close(self):
        if self._run is not None:
            self._run.finish()
            self._run = None
```

Key technical notes:

- **`reinit=True`** allows multiple `MetricWriter` instances per process
  (e.g., notebook re-runs).
- **`settings.start_method="thread"`** avoids fork-vs-grpc deadlocks under
  `multiprocessing.fork()` (relevant for A3C).
- **`flush()` is intentionally a no-op.** wandb maintains its own buffer
  and flushes asynchronously; there is no public API to force a flush.
- **Histogram**: any numpy/torch tensor passed through `wandb.Histogram(...)`.
- **Step axis**: every `wandb.log(...)` call passes `step=env_step`, so
  charts share the X-axis with TB.

---

## Multi-worker A3C handling

Wandb runs do not survive `fork()`: the gRPC channel and uploader thread
hold parent-process state that does not transfer to children. Calling
`wandb.init()` from each worker would either crash or create separate
spurious runs.

**Rule:** `_WandbSink.__init__` runs only in the main process. In a worker
process (detected via `multiprocessing.current_process().name != "MainProcess"`,
or explicitly via a `_skip_wandb_init` agent-level flag), `create_metric_writer`
constructs the writer with the wandb-sink **omitted** even if the env var is
set.

A3C workers continue writing to the parent's TB-sink (already shared via
fork file-descriptor inheritance) under the `/worker_<id>` suffix
convention.

---

## Migration impact

### `MetricWriter` / `create_metric_writer`

- Rename first positional kwarg `log_dir` → `tb_log_dir`. Both the class
  and the factory keep it as the first positional, so existing
  `create_metric_writer("runs/sac", algo="sac")` calls continue to work.
- Add the five wandb kwargs.
- Add the env-var auto-detection block at top of factory.
- All existing `MetricWriter(log_dir=...)` keyword call sites in tests get
  renamed to `tb_log_dir=...` (search-and-replace).

### Each of the 12 RL agents

Add five wandb kwargs to `__init__` (existing `log_dir` stays):

```python
def __init__(
    self,
    ...,
    log_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    wandb_config: Optional[Mapping[str, Any]] = None,
):
    ...
    self.writer = create_metric_writer(
        tb_log_dir=log_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_config=wandb_config,
        algo="<agent>",
    )
```

For agents whose ctor does not accept `log_dir` today (A2C — added in
#215), add the same block. ET-DHP and GAIL already accept `log_dir`; same
treatment.

### `pyproject.toml`

Add `wandb` to `[tool.poetry.dependencies]` (or whatever dependency table
the project uses). Pin to a recent stable major (e.g., `^0.18` or `^0.19`).

---

## Testing strategy

1. **Existing schema/writer unit tests** (`metrics_schema_test.py`,
   `metrics_writer_test.py`) — update to the renamed `tb_log_dir` kwarg;
   no semantic change in coverage.

2. **`tests/agents/metrics_wandb_sink_test.py`** — patch
   `tensoraerospace.agent.metrics.writer.wandb`:
   - `add_scalar` triggers `wandb.log({tag: value}, step=env_step)` exactly once.
   - `add_histogram` wraps values via `wandb.Histogram`.
   - `__init__` calls `wandb.login()` iff `WANDB_API_KEY` is unset.
   - `__init__` calls `wandb.init(...)` with the right kwargs (project,
     entity, run name, tags, config, reinit, settings).
   - `close()` calls `_run.finish()`.
   - Reading `_WandbSink` after `close()` is a no-op (no double-finish).

3. **`tests/agents/metrics_factory_test.py`** — the auto-activation
   table:
   - `tb_log_dir=...` only → 1 sink, type `_TensorBoardSink`.
   - `WANDB_API_KEY` only → 1 sink, type `_WandbSink`, project = algo.
   - both → 2 sinks.
   - nothing → 0 sinks; `add_scalar` whitelist still enforced; no error.
   - `wandb_project=` without `WANDB_API_KEY` → `wandb.login()` called.

4. **`tests/agents/metrics_dual_sink_smoke_test.py`** — runs SAC for 1
   episode with both sinks active (wandb mocked); asserts the TB event
   file contains canonical tags AND the mocked `wandb.log(...)` was
   called with the same set of canonical tags.

5. **`tests/agents/a3c_wandb_parent_only_test.py`** — mock
   `multiprocessing.current_process().name` to a child name; assert that
   `create_metric_writer(wandb_project=...)` returns a writer with
   `_WandbSink` **omitted** from `_sinks`.

6. **All 12 existing per-agent smoke tests** (`*_metrics_smoke_test.py`)
   continue to pass with no edits — default test environment has no
   `WANDB_API_KEY`, so behaviour is identical to today.

---

## Out of scope (explicitly)

- Wandb Sweeps integration.
- Wandb Artifacts (model uploads).
- Offline-mode toggle (users set `WANDB_MODE=offline` themselves).
- MLflow / CSV / console sinks (the `_Sink` protocol allows them later).
- Custom wandb panels / charts pre-configured per algo.

---

## Acceptance criteria

1. `create_metric_writer(tb_log_dir="runs/sac")` behaves exactly as today
   when `WANDB_API_KEY` is unset (no wandb dependency exercised, no extra
   network calls).
2. With `WANDB_API_KEY` set, the same call additionally creates an
   active wandb run with project = `algo`.
3. `create_metric_writer(tb_log_dir="runs/sac", wandb_project="exp1")`
   creates two sinks regardless of env var; missing `WANDB_API_KEY`
   triggers `wandb.login()`.
4. `MetricWriter.add_scalar(schema.LOSS_ACTOR, 0.5, env_step=10)` results
   in (a) a TB event file entry AND (b) a `wandb.log(...)` call when
   both sinks are active.
5. A3C training does not call `wandb.init()` from worker processes.
6. All 12 RL agents accept the five wandb kwargs in their `__init__` and
   forward them to `create_metric_writer(...)`.
7. New unit-test files (4) and updated existing tests pass; full
   `pytest tests/` continues to pass (currently 1724 / 0).
8. Documentation page `docs/en/guide/metrics.md` gains a new section
   "Wandb backend" covering: enabling, env-var auto-detect, kwargs,
   A3C limitation.
