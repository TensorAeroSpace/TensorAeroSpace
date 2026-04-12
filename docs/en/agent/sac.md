# Soft Actor‑Critic (SAC)

SAC is an off-policy actor-critic with entropy maximization: it learns a stochastic policy while increasing expected reward and entropy (exploration). Our implementation employs twin Q-networks, a target critic, Gaussian/deterministic policy options, a replay buffer, soft updates, and optional automatic entropy tuning.

![SAC Diagram](../agent/img/sac/sac.png){ width=800 }

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
