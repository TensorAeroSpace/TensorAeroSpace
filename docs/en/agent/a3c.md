# A3C (Asynchronous Advantage Actor‑Critic)

A3C combines the strengths of policy-based and value-based methods: multiple asynchronous workers explore the environment in parallel and update a shared (global) network using the advantage function.

![A3C схема](../agent/img/a3c/a3c.png){ width=800 }

## Components

- Global network: shared parameters for the Actor (policy) and Critic (value)
- Workers: independent environment instances and local network copies collecting experience
- Advantage update: the policy gradient is weighted by \(A_t\)

## Theory (based on the implementation)

### Policy (Actor) — Gaussian, parameterized via \(\mu\) and \(\sigma\)

The actor network outputs the mean \(\mu(s)\) (through `tanh`, scaled to `action_bound`) and \(\sigma(s)\) (via `softplus`, clipped to `[std_min, std_max]`). The action is sampled as:

$$
 a \sim \mathcal{N}\big(\mu(s),\ \sigma^2(s)\big)
$$

Gaussian log-density (for multidimensional actions, summed across axes):

$$
\log \pi_\theta(a|s) = -\tfrac{1}{2}\,\frac{(a-\mu)^2}{\sigma^2} - \tfrac{1}{2}\,\log(2\pi\sigma^2)
$$

Actor loss (with a negative sign for minimization):

$$
\mathcal{L}_\text{actor}(\theta) = -\,\mathbb{E}\big[\log \pi_\theta(a_t|s_t)\, A_t\big]
$$

(policy entropy can be added as a regularizer: \(+\beta\,\mathbb{E}[\mathcal{H}[\pi]]\)).

### Critic — scalar V-network

The critic estimates \(V_\phi(s)\). Its loss is the MSE between the n-step target and the prediction:

$$
\mathcal{L}_\text{critic}(\phi) = \mathbb{E}\big[\big(R_t^{(n)} - V_\phi(s_t)\big)^2\big]
$$

n-step target (the code uses a simplified version without \(\gamma\) inside the loop and without bootstrapping from \(V(s_{t+n})\); acceptable as a heuristic):

$$
R_t^{(n)} \approx \sum_{k=0}^{n-1} r_{t+k} \quad (\text{эпизодический финал обнуляет «хвост»})
$$

Classic form, to revisit if needed:

$$
R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\phi(s_{t+n})
$$

### Advantage

$$
A_t = R_t^{(n)} - V_\phi(s_t)
$$

It scales the policy log-gradient.

### Asynchrony and synchronization

- Each worker periodically (every `update_interval` steps or at episode end) batches transitions and updates the global networks (in the implementation the actor/critic update calls are commented out, leaving only local weight synchronization).
- Synchronizing local copies with the global networks: `sync_with_global()` copies global actor/critic weights into the local networks.

### Schedules and hyperparameters

- Learning rates: `actor_lr`, `critic_lr`
- Discount factor: `gamma`
- Hidden-layer width: `hidden_size`
- Update interval: `update_interval`
- Episode limit: `max_episodes`
- Action/standard deviation bounds: `action_bound`, `std_bound`

## Asynchronous training (pseudocode)

```text
parallel for worker in 1..W:
  sync local nets from global
  s = env.reset()
  trajectory = []
  for t in range(T_max):
    a ~ N(mu_theta(s), sigma_theta(s)) ; a = clip(a, action_bound)
    s', r, done = env.step(a)
    push (s,a,r,s') into trajectory
    if done or len(trajectory) == update_interval:
      # current implementation: R^n is the accumulated reward sum
      compute R^n from rewards (and optionally bootstrap V(s'))
      A = R^n - V_phi(s)
      # (обновления глобальной сети могут быть добавлены здесь)
      sync local nets from global
      clear trajectory
    s = s'
    if done: s = env.reset()
```

## Quick start

```python
import gymnasium as gym
import numpy as np

from tensoraerospace.agent.a3c.model import Agent, setup_global_params

# Global hyperparameters
actor_lr = 0.0005
critic_lr = 0.001
gamma = 0.99
hidden_size = 128
update_interval = 5
max_episodes = 100

setup_global_params(actor_lr, critic_lr, gamma, hidden_size, update_interval, max_episodes)

# Environment factory (example)
def env_function(worker_id: int):
    env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
    return env

agent = Agent(env_function, gamma)
agent.train()
```

!!! tip
    For continuous actions clip outputs according to `action_bound`; enforce a lower bound on `std` (e.g., `1e-2`) for numerical stability.

## API reference

::: tensoraerospace.agent.a3c.model.Agent

::: tensoraerospace.agent.a3c.model.Worker

::: tensoraerospace.agent.a3c.model.Actor

::: tensoraerospace.agent.a3c.model.Critic

## References

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

## Tested on

- Unity environment
