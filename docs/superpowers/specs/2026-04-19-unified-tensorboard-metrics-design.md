# Unified TensorBoard Metrics — Design

**Date:** 2026-04-19
**Status:** Approved (brainstorming)
**Scope:** All RL agents in `tensoraerospace/agent/` with a training loop.

---

## Problem

TensorBoard metric names across RL agents are inconsistent. Examples observed
in the codebase:

- Episode reward is logged as `Performance/Episode_Reward` (A2C),
  `Performance/Reward` (DDPG, SAC, DSAC, PPO), `Performance/EpisodeReward`
  (DSAC vector), `performance/episode_reward` (ADHDP, ADP), `episode_reward`
  (A2C-NARX), `Performance/{name}/episode_reward` (A3C).
- Loss prefixes mix `Loss/`, `loss/`, `losses/`.
- Episode length is `Performance/Episode Length`, `Performance/EpisodeLength`,
  `performance/episode_length` depending on the agent.
- The X-axis (TensorBoard `global_step`) is sometimes the episode index,
  sometimes the cumulative environment step — making cross-agent comparisons
  on the same axis incorrect.

There is already a partial normalization layer (`MetricWriter` +
`normalize_tag()` + alias maps in `tensoraerospace/agent/metrics.py`). It is
incomplete and only used by 8 of the agents. A3C and A2C-NARX bypass it
entirely; ET-DHP and GAIL do not log to TensorBoard at all.

## Goal

Make TensorBoard runs across every RL agent in TensorAeroSpace directly
comparable: same names, same axis, same minimum metric set — plus
algorithm-specific extras under the same naming conventions.

## Non-goals

- Keeping backward compatibility with old TensorBoard runs. Old runs were
  written under the old names; the unification is a hard rename. Old runs
  will not align with new runs on the same chart.
- Changing the underlying logging backend (still `torch.utils.tensorboard`).
- Touching pure control-law modules that have no RL training loop
  (`AA-INDI`, `HDP`, `iADP`, `iHDP`, `IM-GDHP`, `MPC`, `NARX`, `PID`).

---

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Scope of unification | Mandatory minimum every RL agent must log + per-algorithm extras under the same conventions. |
| Naming style | `lowercase_snake_case`, `/` as group separator. |
| Group taxonomy | `rollout/`, `loss/`, `policy/`, `value/`, `diagnostics/`, `train/`, `eval/`, plus `weights/` and `grads/` for histograms. |
| Backward compatibility | Hard rename. Existing alias map in `metrics.py` is removed. |
| X-axis (`global_step` argument) | Always `global_env_step` — cumulative environment interactions. Per-episode metrics are logged at episode-end against the env-step at that moment. |
| API style | Named constants in a schema module + a `MetricWriter` whitelist (`strict=True` raises `ValueError` for unregistered tags). |
| Agent scope | All RL agents with a training loop, in a single phase: A2C, A2C-NARX, A3C, ADHDP, ADP, DDPG, DQN, DSAC, ET-DHP, GAIL, PPO, SAC. |

---

## Architecture

### Module layout

The current single-file `tensoraerospace/agent/metrics.py` becomes a package:

```
tensoraerospace/agent/metrics/
├── __init__.py      # public re-exports (MetricWriter, schema, contract)
├── schema.py        # canonical names as constants + REGISTRY
├── writer.py        # MetricWriter (strict whitelist, env-step-required API)
└── contract.py      # MANDATORY_METRICS + assert_contract_satisfied helper
```

Removed from current file:

- `_GROUP_ALIASES`
- `_METRIC_ALIASES`
- `normalize_tag()`
- All deprecation/normalization branches inside `MetricWriter.add_scalar`
  and `add_histogram`.

### Public API surface

```python
from tensoraerospace.agent.metrics import MetricWriter, schema
from tensoraerospace.agent.metrics.schema import (
    ROLLOUT_EPISODE_REWARD, ROLLOUT_EPISODE_LENGTH, ROLLOUT_TOTAL_STEPS,
    LOSS_ACTOR, LOSS_CRITIC, LOSS_ENTROPY,
    # ... etc
)
```

---

## Canonical metric schema

### Tier 1 — Mandatory minimum (every RL agent must log)

| Constant | Tag | Notes |
|---|---|---|
| `ROLLOUT_EPISODE_REWARD` | `rollout/episode_reward` | logged at episode end |
| `ROLLOUT_EPISODE_LENGTH` | `rollout/episode_length` | logged at episode end |
| `ROLLOUT_TOTAL_STEPS` | `rollout/total_steps` | cumulative env steps |
| `TRAIN_UPDATES` | `train/updates` | cumulative gradient updates |
| `TRAIN_LR` | `train/lr` | current learning rate; agents with a constant LR may write it once at the start of training to satisfy the contract |

`eval/episode_reward` and `eval/episode_length` are mandatory **only if** the
agent runs an evaluation loop.

### Tier 2 — Common (logged when applicable to the algorithm family)

```python
# loss/*
LOSS_ACTOR    = "loss/actor"
LOSS_CRITIC   = "loss/critic"
LOSS_ENTROPY  = "loss/entropy"
LOSS_VALUE    = "loss/value"

# policy/*
POLICY_ENTROPY         = "policy/entropy"
POLICY_ACTION_STD      = "policy/action_std"
POLICY_ACTION_ABS_MEAN = "policy/action_abs_mean"

# value/*
VALUE_MEAN          = "value/mean"
VALUE_TD_TARGET     = "value/td_target_mean"
VALUE_TD_ERROR_MEAN = "value/td_error_mean"
VALUE_TD_ERROR_MAX  = "value/td_error_max"
VALUE_TD_ERROR_MIN  = "value/td_error_min"

# diagnostics/*
DIAG_TERMINATED_COUNT = "diagnostics/terminated_count"
DIAG_TRUNCATED_COUNT  = "diagnostics/truncated_count"

# eval/*
EVAL_EPISODE_REWARD = "eval/episode_reward"
EVAL_EPISODE_LENGTH = "eval/episode_length"
```

### Tier 3 — Per-algorithm extras

Each algorithm has its own namespaced class inside `schema.py`. Constants
follow the same group prefixes (`loss/`, `policy/`, `train/`, `value/`,
`diagnostics/`) so TensorBoard groups stay coherent.

```python
class PPO:
    APPROX_KL          = "diagnostics/approx_kl"
    CLIP_FRACTION      = "diagnostics/clip_fraction"
    EXPLAINED_VARIANCE = "diagnostics/explained_variance"
    REWARD_MEDIAN      = "rollout/episode_reward_median"
    REWARD_P10         = "rollout/episode_reward_p10"
    REWARD_P90         = "rollout/episode_reward_p90"

class SAC:
    LOSS_Q1     = "loss/q1"
    LOSS_Q2     = "loss/q2"
    LOSS_POLICY = "loss/policy"
    LOSS_ALPHA  = "loss/alpha"
    ALPHA_VALUE = "policy/alpha"
    Q_MEAN      = "value/q_mean"
    LOG_PI_MEAN = "policy/log_pi_mean"
    REPLAY_SIZE = "train/replay_size"

class DSAC(SAC):
    CAPS_SPATIAL  = "loss/caps_spatial"
    CAPS_TEMPORAL = "loss/caps_temporal"

class DQN:
    LOSS_Q           = "loss/q"
    Q_PRED_SA_MEAN   = "value/q_pred_mean"
    Q_TARGET_SA_MEAN = "value/q_target_mean"
    EPSILON          = "train/epsilon"
    PER_BETA         = "train/per_beta"
    REPLAY_SIZE      = "train/replay_size"
    TARGET_UPDATE    = "train/target_update"

class DDPG:
    LOSS_POLICY = "loss/policy"
    LOSS_VALUE  = "loss/value"
    REPLAY_SIZE = "train/replay_size"

class A2C:
    ADVANTAGE_MEAN            = "value/advantage_mean"
    ADVANTAGE_STD             = "value/advantage_std"
    ADVANTAGE_NORMALIZED_MEAN = "value/advantage_normalized_mean"
    VALUE_BEFORE_UPDATE       = "value/before_update_mean"
    ENTROPY_BETA              = "policy/entropy_beta"

class ADP:
    DHP_PHASE_EPISODE  = "train/dhp_phase_episode"
    LOSS_ACTOR_HDP     = "loss/actor_hdp"
    LOSS_ACTOR_GDHP    = "loss/actor_adgdhp"
    LOSS_CRITIC_HDP    = "loss/critic_hdp"
    LOSS_CRITIC_GDHP   = "loss/critic_gdhp"
    LOSS_CRITIC_LAMBDA = "loss/critic_lambda"

class ADHDP:
    DO_CRITIC       = "train/do_critic"
    DO_ACTOR        = "train/do_actor"
    ACTION_SAT_FRAC = "policy/action_sat_frac"

class GAIL:
    LOSS_DISCRIMINATOR = "loss/discriminator"
    LOSS_GENERATOR     = "loss/generator"
    EXPERT_ACCURACY    = "diagnostics/expert_accuracy"
    POLICY_ACCURACY    = "diagnostics/policy_accuracy"
```

### Multi-worker convention (A3C)

Per-worker scalars use a `/worker_<id>` **suffix** so the group prefix
remains shared and TensorBoard groups them together:

```
rollout/episode_reward/worker_0
rollout/episode_reward/worker_1
loss/actor/worker_0
```

`MetricWriter` validates these by stripping the trailing `/worker_<id>`
segment before checking the registry.

### Histogram convention

Histograms use dedicated group prefixes with two-level nesting:

```
weights/actor/<param_name>
weights/critic/<param_name>
grads/actor/<param_name>
grads/critic/<param_name>
```

`MetricWriter.add_histogram` validates the first two segments
(`weights/<group>/...`, `grads/<group>/...`) against a registered set of
groups (`actor`, `critic`, `policy`, `q1`, `q2`, `discriminator`).

### Registry assembly

`schema.REGISTRY` is a `frozenset[str]` collected at import time from all
module-level constants and all per-algorithm class constants. Template
patterns (`weights/<group>/<param>`, `<base>/worker_<N>`) are validated by
prefix rules outside `REGISTRY`.

---

## MetricWriter API

```python
class MetricWriter:
    def __init__(
        self,
        log_dir: str,
        *,
        strict: bool = True,
        required: Iterable[str] = MANDATORY_METRICS,
        algo: str | None = None,
    ) -> None: ...

    def add_scalar(self, tag: str, value: float, env_step: int) -> None: ...
    def add_histogram(self, tag: str, values, env_step: int) -> None: ...

    def log_episode(
        self,
        *,
        reward: float,
        length: int,
        env_step: int,
        terminated: bool | None = None,
        truncated: bool | None = None,
    ) -> None: ...

    def assert_contract_satisfied(self) -> None: ...
    def close(self) -> None: ...
```

### Behaviour

- **Whitelist enforcement.** With `strict=True`, `add_scalar` raises
  `ValueError("Unknown metric tag '<tag>' — register in schema.py or pass "
  "strict=False")` for tags not in `REGISTRY` (after stripping multi-worker
  suffix and after applying the histogram prefix rule for `add_histogram`).
- **`env_step` is a positional/keyword required argument**, no default.
  This forces the caller to think about the X-axis. In multi-worker setups
  (A3C), `env_step` is the global cumulative env step, sourced from a
  shared counter.
- **`log_episode(...)`** is a sugar method that writes the mandatory rollout
  group atomically. Calling it once per episode end satisfies the
  rollout/* part of the contract.
- **`assert_contract_satisfied()`** is invoked at the end of `train()` and
  raises if any name in `required` was never written during the run.
  Catches agents that silently skip mandatory metrics.
- **`close()`** flushes and closes the underlying `SummaryWriter`.

### Example use

```python
from tensoraerospace.agent.metrics import MetricWriter, schema

self.writer = MetricWriter(log_dir=self.log_dir, algo="ppo")

# inside update()
self.writer.add_scalar(schema.LOSS_ACTOR, actor_loss, env_step=self.global_step)
self.writer.add_scalar(schema.PPO.APPROX_KL, kl, env_step=self.global_step)

# at episode end
self.writer.log_episode(
    reward=ep_reward,
    length=ep_length,
    env_step=self.global_step,
    terminated=terminated,
    truncated=truncated,
)

# at end of train()
self.writer.assert_contract_satisfied()
self.writer.close()
```

---

## TensorBoard export of per-algorithm metrics

Per-algorithm constants are written through the same `MetricWriter.add_scalar`
into the same event file. They appear in TensorBoard alongside common
metrics, grouped by prefix:

- `loss/` group expands to common `loss/actor`, `loss/critic` plus
  algorithm extras `loss/q1`, `loss/q2`, `loss/alpha`, `loss/caps_spatial`,
  `loss/q`, `loss/discriminator`, etc.
- `policy/` group expands to common `policy/entropy`, `policy/action_std`
  plus extras `policy/alpha`, `policy/action_sat_frac`, `policy/log_pi_mean`.
- `train/` group expands to common `train/updates`, `train/lr` plus extras
  `train/replay_size`, `train/epsilon`, `train/per_beta`,
  `train/dhp_phase_episode`.

No separate export channel is required — TensorBoard's standard tag
hierarchy handles it.

---

## Migration impact (high level — implementation plan handles details)

| Agent | Change |
|---|---|
| A2C | Rename ~17 tags. Was `Performance/Episode_Reward` etc., now canonical. Already uses `MetricWriter`. |
| A2C-NARX | Rename `losses/*`, `parameters/*`, `gradients/*`, `episode_reward`. Switch to `MetricWriter`. |
| A3C | Switch from raw `SummaryWriter` to `MetricWriter`. Adopt `/worker_<id>` suffix convention. Add a shared global env-step counter. |
| ADHDP | Rename `loss/*`, `performance/*`, `train/*`, `action/*` to canonical (some already match). |
| ADP | Rename `loss/*` (incl. variant losses), `performance/*`, `train/*`. |
| DDPG | Rename `Performance/Reward` to `rollout/episode_reward`, expand minimum set. |
| DQN | Rename `Loss/DQN` → `loss/q`, `Q/PredSA/Mean` → `value/q_pred_mean`, `TD-Error/*` → `value/td_error_*`, `Exploration/Epsilon` → `train/epsilon`, `PER/Beta` → `train/per_beta`. Add mandatory minimum. |
| DSAC | Rename `Loss/Z*` → `loss/q*`, `Loss/Policy`/`Alpha`, `Train/*` → `train/*`, `Performance/*` → `rollout/*` and `eval/*`. |
| ET-DHP | Add full logging from scratch using `MetricWriter`. Mandatory minimum + reuse `LOSS_ACTOR`, `LOSS_CRITIC` from common tier. No per-algo extras initially; add a `schema.ETDHP` block only when a genuinely ET-DHP-specific metric is identified. |
| GAIL | Add full logging from scratch. Discriminator/generator losses + mandatory minimum. |
| PPO | Rename `Performance/*` → `rollout/*` and `eval/*`, `Loss/Actor`/`Critic` → `loss/actor`/`loss/critic`, `Diagnostics/Approx KL` → `diagnostics/approx_kl` (no spaces). |
| SAC | Rename `Loss/QF*` → `loss/q*`, `Performance/*` → `rollout/*` and `eval/*`, switch X-axis to `env_step`. |

All rename diffs are mechanical: replace string literal with constant import.

---

## X-axis correction

For agents currently logging against episode index (A2C, PPO, parts of
A3C), every `add_scalar` call is updated to pass `env_step=self.global_step`
where `self.global_step` is incremented on every environment interaction.

For multi-worker A3C, a shared `mp.Value('i', 0)` (or equivalent) is added
to the worker pool; each worker increments it on every env step. All
workers log against this shared counter so curves overlay correctly.

---

## Testing strategy

1. **Schema unit tests** (`tests/agent/metrics/test_schema.py`):
   - Every constant value matches `^[a-z][a-z0-9_]*(\/[a-z0-9_]+)+$` (lowercase, snake_case, slash-separated, no spaces).
   - No duplicate values across modules and per-algo classes.
   - `REGISTRY` contains every defined constant.

2. **Writer unit tests** (`tests/agent/metrics/test_writer.py`):
   - `add_scalar("not/in/schema", ...)` raises `ValueError` when `strict=True`.
   - `add_scalar("not/in/schema", ...)` is silently logged when `strict=False`.
   - Multi-worker suffix `tag/worker_3` validates if `tag` is in registry.
   - Histogram tag `weights/actor/foo.weight` validates by prefix rule.
   - Calling `add_scalar` without `env_step` is a `TypeError` (signature-enforced).
   - `assert_contract_satisfied()` raises if a mandatory tag was never written; passes otherwise.

3. **Per-agent smoke tests** (`tests/agent/<name>/test_metrics_contract.py`):
   - Run a tiny `train(num_episodes=2, max_steps=8)` with a stub env.
   - After training, parse the event file and assert every mandatory tag is present at least once.
   - For each agent, also assert that its declared per-algo tags are present.

4. **No regression on existing agent tests.** Existing unit/integration
   tests under `tests/` continue to pass after the rename.

---

## Out of scope (explicitly)

- Renaming or restructuring agent classes themselves.
- Changing reward shaping, environment APIs, or training hyperparameters.
- Adding new functional metrics that the agent does not currently compute.
  (Mandatory minimum was chosen so that every agent already has — or can
  trivially compute — those values.)
- Building a custom dashboard, web UI, or external metrics service.
- Migration of historical TensorBoard runs to the new naming. Old runs
  remain readable under their old names; they simply will not align on the
  same chart with new runs.

---

## Acceptance criteria

1. `tensoraerospace/agent/metrics/` exists with `schema.py`, `writer.py`,
   `contract.py`, and a re-exporting `__init__.py`. The old single-file
   `metrics.py` no longer exists.
2. `_GROUP_ALIASES`, `_METRIC_ALIASES`, and `normalize_tag()` are not
   present anywhere in the codebase.
3. Every `writer.add_scalar(...)` and `writer.add_histogram(...)` call in
   `tensoraerospace/agent/**/*.py` uses a constant imported from
   `tensoraerospace.agent.metrics.schema`. No free-form string literals
   for metric tags remain in agent code.
4. Every RL agent's `train()` writes the mandatory minimum at least once
   per run; verified by per-agent smoke tests.
5. Every `add_scalar` / `add_histogram` call passes a cumulative
   environment step as its step argument.
6. Schema unit tests, writer unit tests, and per-agent smoke tests pass.
7. Documentation page lists the canonical schema (mandatory + per-algo)
   for each algorithm.
