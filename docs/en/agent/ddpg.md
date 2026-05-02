# Deep Deterministic Policy Gradient (DDPG)

DDPG is an off-policy actor-critic for continuous actions: it trains a deterministic policy and a Q-function using a replay buffer and target networks with soft updates.

## Components

- Policy (Actor): `PolicyNetwork(s) -> a`, deterministic action via `tanh`
- Critic (Q-network): `ValueNetwork(s,a) -> Q(s,a)`
- Target networks: `target_policy_net`, `target_value_net` for stability
- Replay buffer: `ReplayBuffer` to sample mini-batches
- Exploration: Ornstein–Uhlenbeck noise `OUNoise`

## Theory (from the implementation)

- Policy gradient (DPG):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s\sim \mathcal{D}}\Big[\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)}\, \nabla_\theta \pi_\theta(s)\Big]
$$

In the code we minimize \(-Q(s,\pi(s))\), which is equivalent to gradient ascent on \(J\).

- Critic update (Bellman target with target networks):

$$
\hat{Q}(s,a) = r + \gamma\,(1-\text{done})\, Q_{\text{target}}(s', \pi_{\text{target}}(s'))
$$

Critic loss is the MSE: \(\mathcal{L}_Q = (Q(s,a) - \hat{Q})^2\).

- Soft update of target networks:

$$
\theta^- \leftarrow (1-\tau)\,\theta^- + \tau\,\theta
$$

## Quick start

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

# Time grid and reference
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=1000, output_rad=True).reshape(1, -1)

# F‑16 environment
env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=number_time_steps,
               initial_state=[[0],[0],[0]],
               reference_signal=reference_signals,
               use_reward=True,
               state_space=["theta","alpha","q"],
               output_space=["theta","alpha","q"],
               control_space=["ele"],
               tracking_states=["alpha"],)

agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1_000_000)
agent.learn(max_frames=12000, max_steps=500, batch_size=128)
```

!!! tip
    Exploration relies on OU noise: tune `sigma` and `decay_period` to gradually reduce noise intensity.

## Unified training interface

DDPG also exposes the shared unified `train()` API from `BaseRLModel`:

```python
agent.train(
    num_episodes=24,
    max_steps=500,
    batch_size=128,
    warmup_frames=2_000,
)
```

Under the hood `train()` converts `num_episodes * max_steps` into a
`max_frames` budget and calls the legacy `learn()` method. Accepted
DDPG-specific keyword arguments (passed via `**kwargs`):

- `max_frames`, `batch_size`, `gamma`, `soft_tau`, `warmup_frames`,
  `updates_per_step`, `target_value_clip`.

The legacy `agent.learn(max_frames=..., max_steps=..., batch_size=...)`
call continues to work unchanged.

## API reference

::: tensoraerospace.agent.ddpg.model.DDPG

## References

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

## Tested on

- Unity environment
- LinearLongitudinalF16‑v0 (repository example)
