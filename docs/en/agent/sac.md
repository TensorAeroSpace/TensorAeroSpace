# Soft Actor‑Critic (SAC)

!!! note "Vector transitions"
    `train_vector()` samples warmup actions within the environment's action bounds. With auto-reset, `info["final_observation"]` and its optional mask `info["_final_observation"]` preserve the last observation for replay; only true termination removes bootstrapping. Older environments without that metadata retain the conservative terminal mask. `ImprovedB747VecEnvTorch` supplies both fields.

SAC is an off-policy actor-critic with entropy maximization: it learns a stochastic policy while increasing expected reward and entropy (exploration). Our implementation employs twin Q-networks, a target critic, Gaussian/deterministic policy options, a replay buffer, soft updates, and optional automatic entropy tuning.

![SAC Diagram](../agent/img/sac/sac.png){ width=800 }

## Scalar transition collection

`train()` snapshots the observation before `env.step()`, so environments that
reuse a mutable observation array cannot corrupt the replay transition.
At episode boundaries, `final_observation` or `terminal_observation` supplies
the actual next state when an environment resets automatically. Only true
termination suppresses bootstrapping; user step caps are logged as truncation.
Short training runs that finish before replay warmup explicitly log zero
optimizer updates and satisfy the metrics contract.

## Components

- Twin Q-networks: `QNetwork(state, action) -> (Q1, Q2)` plus target `critic_target`
- Policy: `GaussianPolicy` (default) or `DeterministicPolicy` (no entropy term)
- Replay buffer: `ReplayMemory` for mini-batch sampling
- Soft target update: `soft_update(target, source, tau)`
- Automatic entropy tuning: optimizes `alpha` toward \(H_{\text{target}} = -\dim(\mathcal{A})\)

## Theory (as implemented)

- Soft Q target (double Q + entropy):

$$
\begin{aligned}
& a' \sim \pi_\theta(\cdot|s')\ ,\ \log \pi_\theta(a'|s'), \\
& Q_{\text{targ}}(s,a) = r + \gamma\, \big( \min(Q_1(s',a'), Q_2(s',a')) - \alpha\, \log \pi_\theta(a'|s') \big)
\end{aligned}
$$

- Critic update (MSE to target): \(\mathcal{L}_{Q_i} = \mathbb{E}[(Q_i(s,a) - Q_{\text{targ}})^2]\)

- Policy update (reparameterization):

$$
\mathcal{L}_\pi = \mathbb{E}_{s\sim \mathcal{D},\ \epsilon\sim\mathcal{N}}\big[ \alpha\, \log \pi_\theta(f_\theta(\epsilon; s) | s) - Q_{\min}(s, f_\theta(\epsilon; s)) \big]
$$

- Automatic \(\alpha\) tuning (optional):

$$
\mathcal{L}_\alpha = -\,\mathbb{E}_{a\sim\pi}\big[\log \alpha\, (\log \pi_\theta(a|s) + H_{\text{target}})\big]\ ,\quad \alpha \leftarrow e^{\log \alpha}
$$

## Quick start

```python
import gymnasium as gym
from tensoraerospace.agent.sac.sac import SAC

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
agent = SAC(env,
            updates_per_step=1,
            batch_size=64,
            memory_capacity=100000,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            policy_type='Gaussian',
            target_update_interval=1,
            automatic_entropy_tuning=True,
            hidden_size=256,
            device='cpu')

agent.train(num_episodes=100)
agent.save('./runs')
```

!!! tip
    For continuous action spaces keep `GaussianPolicy` with `automatic_entropy_tuning=True` to stabilize exploration.

## Unified training interface

All TensorAeroSpace RL agents share a common `train()` signature defined on
`BaseRLModel`:

```python
def train(
    self,
    num_episodes: int = 100,
    *,
    max_steps: Optional[int] = None,
    save_best: bool = False,
    save_path: Optional[str] = None,
    verbose: bool = True,
    **kwargs,
) -> dict
```

For SAC the algorithm-specific options accepted via `**kwargs` are:

- `save_best_with_gradients` (`bool`): include optimizer gradients in
  best-model checkpoints.

Example:

```python
stats = agent.train(
    num_episodes=100,
    max_steps=500,
    save_best=True,
    save_path='./runs/sac_best',
)
print(stats['best_reward'], len(stats['episode_rewards']))
```

## Practical tips

- Increase `batch_size` and `memory_capacity` for steadier gradients
- Choose `tau` around 0.005–0.02 for soft target updates
- With a deterministic policy set `alpha=0` and disable auto tuning
- When using `DeterministicPolicy` with `action_space=None`, note that `action_scale` and `action_bias` are now `torch.Tensor` values (not Python floats)

!!! warning "Gymnasium 5-tuple API"
    This implementation uses the modern Gymnasium 5-tuple step API internally:
    ```python
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ```
    If you are migrating from older code that used the 4-tuple API (`next_state, reward, done, info = env.step(action)`), ensure your environment is compatible with Gymnasium and returns the 5-tuple.

## API reference

::: tensoraerospace.agent.sac.sac.SAC

::: tensoraerospace.agent.sac.replay_memory.ReplayMemory

::: tensoraerospace.agent.sac.model.ValueNetwork

::: tensoraerospace.agent.sac.model.QNetwork

::: tensoraerospace.agent.sac.model.GaussianPolicy

::: tensoraerospace.agent.sac.model.DeterministicPolicy

## Continuing training and saving settings

`train()` and `train_vector()` retain `total_updates` and `total_env_steps` across
calls. Target-network updates follow the cumulative gradient-update count.
Splitting scalar training at episode boundaries preserves the result with the
same environment data and random-number stream. `train()` returns updates and
the best reward **for that call**; `best_reward` is computed with
`save_best=False` as well.

`policy_lr` controls both Gaussian and Deterministic policy optimizers separately
from the critic's `lr`. Checkpoints retain both rates, logging frequency and
cumulative counters. Legacy checkpoints without counters start at zero. Replay,
environment and RNG states are not saved: loading retains the target-update
phase, but does not reproduce an uninterrupted run exactly.

Each `train_vector()` call resets the environment and applies its own
`warmup_steps` budget. Comparisons of vector training chunks must align episode
boundaries and warmup. With direct `update_parameters(..., updates)` calls, the
caller owns the update index.

### Evaluation without changing training

`select_action(..., evaluate=True)` and `select_action_batch(..., evaluate=True)`
compute the mean action without sampling noise. They do not advance the PyTorch
RNG, mutate Deterministic-policy noise buffers or build gradient graphs.
Evaluation calls therefore do not change the subsequent training RNG stream.
If an evaluation environment uses global randomness, isolate that randomness
separately. `evaluate=False` retains exploratory action selection.

### Deterministic-policy exploration

`policy_type="Deterministic"` draws independent noise for every batch row and
action component: standard deviation 0.1 in normalized coordinates, clipped to
±0.25. Noise is then scaled by each action's half-range and the resulting action
is clipped to the exact action-space bounds. This gives consistent relative
exploration in radians, degrees and Newtons. The legacy `noise` buffer is retained
for checkpoint loading but is no longer used to generate samples.

Evaluation remains deterministic. This correction changes exploration sequences
and training; it does not guarantee a higher reward.

### Squashed Gaussian density near actuator limits

The Gaussian SAC policy evaluates the `tanh` change-of-variables Jacobian from
the latent sample using a stable softplus expression and accounts separately for
physical action scaling. This preserves the entropy gradient when float32
`tanh` rounds to ±1. Action intervals must have finite, positive widths.
Network parameters and deterministic actions for existing checkpoints are
unchanged; stochastic log probabilities and subsequent optimization do change.
This numerical correction does not guarantee closed-loop stability: compare
tracking errors and boundary violations across seeds and evaluation horizons.

The formula follows [PyTorch's TanhTransform](https://github.com/pytorch/pytorch/blob/main/torch/distributions/transforms.py)
and the density correction in [SAC](https://arxiv.org/abs/1801.01290).
