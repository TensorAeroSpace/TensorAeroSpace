# TensorBoard Metrics

> :material-chart-line: Unified TensorBoard schema for every RL agent in TensorAeroSpace.

All RL agents in TensorAeroSpace write TensorBoard scalars and histograms through a
single `MetricWriter` whose tag namespace is defined in
`tensoraerospace.agent.metrics.schema`. Because every agent uses the same names,
the same axis (cumulative environment steps), and the same minimum metric set,
runs from PPO, SAC, DQN, ADP, GAIL, and others overlay correctly on the same
TensorBoard chart.

!!! note
    Old runs written under the previous, agent-specific naming will not align
    with new runs on the same chart. The unification is a hard rename — the
    legacy alias map has been removed.

## Why a unified schema

Before unification, the same quantity was logged under multiple names depending
on the agent — `Performance/Episode_Reward`, `Performance/Reward`,
`performance/episode_reward`, `episode_reward`, and so on. Loss prefixes mixed
`Loss/`, `loss/`, and `losses/`. The X-axis was sometimes the episode index,
sometimes cumulative environment steps, which made cross-agent comparisons on
the same chart incorrect.

The unified schema fixes all three issues at once:

- One canonical name per quantity (`lowercase_snake_case`, `/` as group separator).
- One axis: `env_step` is a required argument on every `add_scalar` call.
- One mandatory minimum that every RL agent must log, so any two agents are
  always comparable on at least the basic rollout/training metrics.

## Group prefixes

| Prefix | Purpose |
|---|---|
| `rollout/` | Per-episode environment statistics (reward, length, total steps). |
| `loss/` | Training losses (actor, critic, entropy, value, policy, plus algo-specific). |
| `policy/` | Policy / action statistics (entropy, action std, log-pi mean). |
| `value/` | Value-function statistics (mean V, TD targets, TD errors, advantages). |
| `diagnostics/` | Algorithm-specific diagnostics (KL, clip fraction, accuracy). |
| `train/` | Training progress counters (updates, learning rate, replay size). |
| `eval/` | Evaluation episode statistics (reward, length). |
| `weights/` | Network weight histograms (`weights/<group>/<param>`). |
| `grads/` | Gradient histograms (`grads/<group>/<param>`). |

## Tier 1 — Mandatory minimum

Every RL agent must log these. They are checked at the end of `train()` by
`assert_contract_satisfied()`.

| Constant | Tag | Notes |
|---|---|---|
| `ROLLOUT_EPISODE_REWARD` | `rollout/episode_reward` | logged at episode end |
| `ROLLOUT_EPISODE_LENGTH` | `rollout/episode_length` | logged at episode end |
| `ROLLOUT_TOTAL_STEPS` | `rollout/total_steps` | cumulative env steps |
| `TRAIN_UPDATES` | `train/updates` | cumulative gradient updates |
| `TRAIN_LR` | `train/lr` | current learning rate (constant-LR agents may write it once at start) |

`EVAL_EPISODE_REWARD` (`eval/episode_reward`) and `EVAL_EPISODE_LENGTH`
(`eval/episode_length`) are mandatory only if the agent runs an evaluation loop.

## Tier 2 — Common constants

These are logged when the algorithm has a corresponding quantity. They live as
module-level constants in `tensoraerospace.agent.metrics.schema`.

### `loss/*`

| Constant | Tag | One-liner |
|---|---|---|
| `LOSS_ACTOR` | `loss/actor` | Actor / policy gradient loss. |
| `LOSS_CRITIC` | `loss/critic` | Critic / value loss for actor-critic methods. |
| `LOSS_ENTROPY` | `loss/entropy` | Entropy regularization term. |
| `LOSS_VALUE` | `loss/value` | Standalone value loss (when distinct from `loss/critic`). |
| `LOSS_POLICY` | `loss/policy` | Policy loss for off-policy actor-critic (SAC, DDPG). |

### `train/*`

| Constant | Tag | One-liner |
|---|---|---|
| `TRAIN_REPLAY_SIZE` | `train/replay_size` | Current replay-buffer occupancy (off-policy). |

### `policy/*`

| Constant | Tag | One-liner |
|---|---|---|
| `POLICY_ENTROPY` | `policy/entropy` | Differential entropy of the current policy. |
| `POLICY_ACTION_STD` | `policy/action_std` | Mean per-dim standard deviation of the action distribution. |
| `POLICY_ACTION_ABS_MEAN` | `policy/action_abs_mean` | Mean absolute action magnitude (saturation indicator). |

### `value/*`

| Constant | Tag | One-liner |
|---|---|---|
| `VALUE_MEAN` | `value/mean` | Mean V(s) over the batch. |
| `VALUE_TD_TARGET` | `value/td_target_mean` | Mean of the TD target. |
| `VALUE_TD_ERROR_MEAN` | `value/td_error_mean` | Mean TD error (target - prediction). |
| `VALUE_TD_ERROR_MAX` | `value/td_error_max` | Max TD error in the batch. |
| `VALUE_TD_ERROR_MIN` | `value/td_error_min` | Min TD error in the batch. |

### `diagnostics/*`

| Constant | Tag | One-liner |
|---|---|---|
| `DIAG_TERMINATED_COUNT` | `diagnostics/terminated_count` | Number of terminated episodes since last log. |
| `DIAG_TRUNCATED_COUNT` | `diagnostics/truncated_count` | Number of truncated episodes since last log. |

### `eval/*`

| Constant | Tag | One-liner |
|---|---|---|
| `EVAL_EPISODE_REWARD` | `eval/episode_reward` | Reward of an evaluation episode. |
| `EVAL_EPISODE_LENGTH` | `eval/episode_length` | Length of an evaluation episode. |

## Tier 3 — Per-algorithm extras

Each algorithm has a namespaced class inside `schema.py`. Tags continue to use
the shared group prefixes, so TensorBoard groups stay coherent across runs.

### PPO

`from tensoraerospace.agent.metrics.schema import PPO`

| Constant | Tag | One-liner |
|---|---|---|
| `PPO.APPROX_KL` | `diagnostics/approx_kl` | Empirical KL between old and new policy. |
| `PPO.CLIP_FRACTION` | `diagnostics/clip_fraction` | Fraction of samples hitting the clip bound. |
| `PPO.EXPLAINED_VARIANCE` | `diagnostics/explained_variance` | 1 − Var(returns − V) / Var(returns). |
| `PPO.REWARD_MEDIAN` | `rollout/episode_reward_median` | Median episode reward over the rollout window. |
| `PPO.REWARD_P10` | `rollout/episode_reward_p10` | 10th percentile of episode rewards (downside). |
| `PPO.REWARD_P90` | `rollout/episode_reward_p90` | 90th percentile of episode rewards (upside). |

### SAC

`from tensoraerospace.agent.metrics.schema import SAC`

| Constant | Tag | One-liner |
|---|---|---|
| `SAC.LOSS_Q1` | `loss/q1` | Loss of the first Q-network. |
| `SAC.LOSS_Q2` | `loss/q2` | Loss of the second Q-network. |
| `SAC.LOSS_ALPHA` | `loss/alpha` | Loss for the temperature parameter. |
| `SAC.ALPHA_VALUE` | `policy/alpha` | Current entropy temperature α. |
| `SAC.Q_MEAN` | `value/q_mean` | Mean Q(s, a) over the batch. |
| `SAC.LOG_PI_MEAN` | `policy/log_pi_mean` | Mean log π(a|s) of sampled actions. |

SAC also uses the common `LOSS_POLICY` (`loss/policy`) and `TRAIN_REPLAY_SIZE`
(`train/replay_size`) constants.

### DSAC

`from tensoraerospace.agent.metrics.schema import DSAC`

DSAC inherits all SAC names and adds CAPS regularization terms.

| Constant | Tag | One-liner |
|---|---|---|
| `DSAC.CAPS_SPATIAL` | `loss/caps_spatial` | CAPS spatial smoothness penalty. |
| `DSAC.CAPS_TEMPORAL` | `loss/caps_temporal` | CAPS temporal smoothness penalty. |

### DQN

`from tensoraerospace.agent.metrics.schema import DQN`

| Constant | Tag | One-liner |
|---|---|---|
| `DQN.LOSS_Q` | `loss/q` | Bellman / Huber loss on Q-values. |
| `DQN.Q_PRED_SA_MEAN` | `value/q_pred_mean` | Mean predicted Q(s, a). |
| `DQN.Q_TARGET_SA_MEAN` | `value/q_target_mean` | Mean target Q(s, a). |
| `DQN.EPSILON` | `train/epsilon` | Current ε for ε-greedy exploration. |
| `DQN.PER_BETA` | `train/per_beta` | Importance-sampling exponent β for prioritized replay. |
| `DQN.TARGET_UPDATE` | `train/target_update` | Counter of target-network updates. |

DQN also uses common `TRAIN_REPLAY_SIZE` (`train/replay_size`).

### DDPG

`from tensoraerospace.agent.metrics.schema import DDPG`

DDPG has no DDPG-specific tags. It uses only common Tier 2 constants:
`LOSS_POLICY` (`loss/policy`), `LOSS_VALUE` (`loss/value`), and
`TRAIN_REPLAY_SIZE` (`train/replay_size`).

### A2C

`from tensoraerospace.agent.metrics.schema import A2C`

| Constant | Tag | One-liner |
|---|---|---|
| `A2C.ADVANTAGE_MEAN` | `value/advantage_mean` | Mean advantage before normalization. |
| `A2C.ADVANTAGE_STD` | `value/advantage_std` | Std of advantage before normalization. |
| `A2C.ADVANTAGE_NORMALIZED_MEAN` | `value/advantage_normalized_mean` | Mean of normalized advantage. |
| `A2C.VALUE_BEFORE_UPDATE` | `value/before_update_mean` | Mean V(s) before the gradient update. |
| `A2C.ENTROPY_BETA` | `policy/entropy_beta` | Current entropy-regularization coefficient β. |

### ADP

`from tensoraerospace.agent.metrics.schema import ADP`

| Constant | Tag | One-liner |
|---|---|---|
| `ADP.DHP_PHASE_EPISODE` | `train/dhp_phase_episode` | Episode index within the current DHP phase. |
| `ADP.LOSS_ACTOR_HDP` | `loss/actor_hdp` | Actor loss in the HDP variant. |
| `ADP.LOSS_ACTOR_GDHP` | `loss/actor_adgdhp` | Actor loss in the AD-GDHP variant. |
| `ADP.LOSS_CRITIC_HDP` | `loss/critic_hdp` | Critic loss in the HDP variant. |
| `ADP.LOSS_CRITIC_GDHP` | `loss/critic_gdhp` | Critic loss in the GDHP variant. |
| `ADP.LOSS_CRITIC_LAMBDA` | `loss/critic_lambda` | λ-weighted critic loss. |

### ADHDP

`from tensoraerospace.agent.metrics.schema import ADHDP`

| Constant | Tag | One-liner |
|---|---|---|
| `ADHDP.DO_CRITIC` | `train/do_critic` | 1 if the critic was updated this step, else 0. |
| `ADHDP.DO_ACTOR` | `train/do_actor` | 1 if the actor was updated this step, else 0. |
| `ADHDP.ACTION_SAT_FRAC` | `policy/action_sat_frac` | Fraction of actions hitting saturation bounds. |

### GAIL

`from tensoraerospace.agent.metrics.schema import GAIL`

| Constant | Tag | One-liner |
|---|---|---|
| `GAIL.LOSS_DISCRIMINATOR` | `loss/discriminator` | Discriminator BCE loss. |
| `GAIL.LOSS_GENERATOR` | `loss/generator` | Generator (policy) loss against discriminator score. |
| `GAIL.EXPERT_ACCURACY` | `diagnostics/expert_accuracy` | Discriminator accuracy on expert transitions. |
| `GAIL.POLICY_ACCURACY` | `diagnostics/policy_accuracy` | Discriminator accuracy on policy transitions. |

## Multi-worker convention

For agents with multiple parallel workers (A3C in particular), per-worker
scalars use a `/worker_<id>` **suffix** so that the group prefix remains shared
and TensorBoard places all workers in the same group.

```text
rollout/episode_reward/worker_0
rollout/episode_reward/worker_1
loss/actor/worker_0
loss/actor/worker_1
```

`MetricWriter` validates these tags by stripping the trailing `/worker_<N>`
segment before checking the registry. Any registered scalar tag may be suffixed
with `/worker_<N>` where `<N>` is a non-negative integer.

## Histogram convention

Histograms use dedicated top-level groups (`weights/`, `grads/`) with two
levels of nesting:

```text
weights/actor/<param_name>
weights/critic/<param_name>
grads/actor/<param_name>
grads/critic/<param_name>
```

`MetricWriter.add_histogram` validates the first two segments. Allowed
top-level groups (`HISTOGRAM_GROUPS`) are `weights` and `grads`. Allowed
sub-groups (`HISTOGRAM_SUBGROUPS`) are:

| Sub-group | Used by |
|---|---|
| `actor` | actor-critic agents (PPO, A2C, DDPG, SAC, DSAC, ADP, ADHDP) |
| `critic` | actor-critic agents (PPO, A2C, DDPG, SAC, DSAC, ADP, ADHDP) |
| `policy` | policy-only agents and SAC-style policies |
| `value` | standalone value networks |
| `q` | DQN Q-network |
| `q1`, `q2` | SAC / DSAC twin Q-networks |
| `discriminator` | GAIL discriminator |

## Example usage

```python
from tensoraerospace.agent.metrics import MetricWriter, schema
from tensoraerospace.agent.metrics.schema import (
    LOSS_ACTOR,
    LOSS_CRITIC,
    POLICY_ENTROPY,
    PPO,
)

# Construct the writer with a strict whitelist and the algo tag.
self.writer = MetricWriter(log_dir=self.log_dir, algo="ppo")

# Inside update() — every scalar requires env_step.
self.writer.add_scalar(LOSS_ACTOR,    actor_loss,  env_step=self.global_step)
self.writer.add_scalar(LOSS_CRITIC,   critic_loss, env_step=self.global_step)
self.writer.add_scalar(POLICY_ENTROPY, entropy,    env_step=self.global_step)

# Per-algorithm extras (PPO).
self.writer.add_scalar(PPO.APPROX_KL,     kl,          env_step=self.global_step)
self.writer.add_scalar(PPO.CLIP_FRACTION, clip_frac,   env_step=self.global_step)

# Histograms (validated by the weights/<group>/<param> rule).
for name, p in self.actor.named_parameters():
    self.writer.add_histogram(f"weights/actor/{name}", p, env_step=self.global_step)
    if p.grad is not None:
        self.writer.add_histogram(f"grads/actor/{name}", p.grad, env_step=self.global_step)

# At episode end — atomic write of the rollout/* mandatory minimum.
self.writer.log_episode(
    reward=ep_reward,
    length=ep_length,
    env_step=self.global_step,
    terminated=terminated,
    truncated=truncated,
)

# At the end of train() — fail loudly if any mandatory tag was never written.
self.writer.assert_contract_satisfied()
self.writer.close()
```

!!! tip
    `env_step` is a required keyword argument on every `add_scalar` and
    `add_histogram` call. This forces every agent to think about the X-axis
    and guarantees that runs overlay correctly on shared charts.

## Adding a new algorithm

1. **Edit `tensoraerospace/agent/metrics/schema.py`.** Add a new class named
   after the algorithm:

   ```python
   class MyAlgo:
       MY_LOSS = "loss/my_term"
       MY_DIAG = "diagnostics/my_signal"
   ```

   Reuse existing group prefixes (`loss/`, `policy/`, `value/`, `train/`,
   `diagnostics/`, `rollout/`, `eval/`) so TensorBoard groups stay coherent.
   Tags must be `lowercase_snake_case` with `/` as the group separator and no
   spaces.

2. **Register the class in `_build_registry()`.** Add `MyAlgo` to the tuple of
   classes iterated inside `_build_registry()` so its constants land in
   `REGISTRY`:

   ```python
   for cls in (PPO, SAC, DSAC, DQN, DDPG, A2C, ADP, ADHDP, GAIL, MyAlgo):
       parts.append(_collect_constants(cls))
   ```

3. **Use the constants from your agent.** Import them by name; never use
   free-form string literals for tags:

   ```python
   from tensoraerospace.agent.metrics.schema import MyAlgo
   self.writer.add_scalar(MyAlgo.MY_LOSS, loss, env_step=self.global_step)
   ```

4. **Run the schema and writer unit tests.** They will catch duplicate tag
   values, malformed names, and any new constant missing from `REGISTRY`.

## Reference

- Canonical source: `tensoraerospace/agent/metrics/schema.py`
- Writer implementation: `tensoraerospace/agent/metrics/writer.py`
- Contract helper: `tensoraerospace/agent/metrics/contract.py`
