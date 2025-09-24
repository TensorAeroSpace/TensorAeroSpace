# Generative Adversarial Imitation Learning (GAIL)

GAIL performs imitation learning via an actor–discriminator adversarial game: the policy learns to generate trajectories indistinguishable from expert demonstrations without an explicit reward function.

## Components

- Actor-Critic: `ActorCritic` outputs actions and estimates \(V(s)\)
- Discriminator: `Discriminator` distinguishes expert from agent pairs \((s,a)\)
- Policy optimizer: PPO-style clipped surrogate updates

## Theory

- Min–max objective:

$$
\min_{\pi} \max_{D} \; \mathbb{E}_{(s,a)\sim \pi_E}[\log D(s,a)] + \mathbb{E}_{(s,a)\sim \pi}[\log (1 - D(s,a))]
$$

- Discriminator-derived pseudo reward (for the actor):

$$
 r_D(s,a) = -\log D(s,a)
$$

- PPO actor update (as implemented):

$$
\mathcal{L}_\text{actor} = -\,\mathbb{E}\Big[ \min\big(r_t A_t,\ \mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon) A_t\big) \Big],\quad
r_t = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})
$$

- Advantage via GAE:

$$
\delta_t = r_D(s_t,a_t) + \gamma V(s_{t+1}) - V(s_t),\quad
\hat{A}_t = \sum_{l\ge 0} (\gamma\lambda)^l\, \delta_{t+l}
$$

## Expert data

Expect `expert_data` as an array of shape `[N, obs_dim + act_dim]`: state concatenated with action.

## Training loop

1. Generate agent rollouts \(s_t, a_t \sim \pi\); store `log_prob`, `V(s)`
2. Compute pseudo rewards `r_D = -log D([s,a])`, then GAE returns/advantages
3. Update actor/critic with PPO mini-batches
4. Train the discriminator with BCE: `D(fake)=1`, `D(real)=0`
5. Periodically evaluate the policy and early-stop based on `max_reward`

## Пример (LinearLongitudinalF16‑v0)

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.gail.model import GAIL
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=1000, output_rad=True).reshape(1, -1)

env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=number_time_steps,
               initial_state=[[0],[0],[0]],
               reference_signal=reference_signals,
               use_reward=False,
               state_space=["theta","alpha","q"],
               output_space=["theta","alpha","q"],
               control_space=["ele"],
               tracking_states=["alpha"],)

expert_data = np.load('expert_f16.npy')
agent = GAIL(env, learning_rate=3e-3, max_steps=20, mini_batch_size=16, epochs=4, data=expert_data)
agent.learn(max_frames=5000, max_reward=-1)
```

!!! tip
    High-quality `expert_data` is crucial—include demonstrations with varied initial states and maneuvers.

## API reference

::: tensoraerospace.agent.gail.model.GAIL

## References

- [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)

## Tested on

- Unity environment
- LinearLongitudinalF16‑v0 (repository example)
