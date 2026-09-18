# Generative Adversarial Imitation Learning (GAIL)

GAIL performs imitation learning via an actor–discriminator adversarial game: the policy learns to generate trajectories indistinguishable from expert demonstrations without an explicit reward function.

## Components

- Actor-Critic: `ActorCritic` outputs actions and estimates \(V(s)\)
- Discriminator: `Discriminator` distinguishes expert from agent pairs \((s,a)\)
- Policy optimizer: PPO-style clipped surrogate updates

## Theory

- Min–max objective:

$$
\min_{\pi} \max_{D} \; \mathbb{E}_{(s,a)\sim \pi}[\log D(s,a)] + \mathbb{E}_{(s,a)\sim \pi_E}[\log (1 - D(s,a))]
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

1. Collect policy rollouts; store sampled actions, executed commands, log probabilities and values.
2. Fit the discriminator on those commands and expert data: `D(fake)=1`, `D(real)=0`.
3. Recompute imitation rewards using the updated discriminator, then GAE returns/advantages.
4. Update actor/critic with PPO mini-batches.
5. Evaluate between rollouts and apply early stopping based on `max_reward`.

`Discriminator.logits()` returns the unbounded score `z`. Training uses
`BCEWithLogitsLoss`; rewards use `softplus(-z)`, equivalent to
`-log(sigmoid(z))` without probability clamping. This preserves gradients for
confidently wrong discriminator predictions. `forward()` still returns a
probability, and saved parameter names/shapes are unchanged.

Updating the discriminator before forming policy rewards follows
[Algorithm 1 of the GAIL paper](https://arxiv.org/html/1606.03476v1).
This implementation uses PPO for the policy step. Stable losses and the correct
update order do not guarantee convergence; evaluate tracking and constraint
violations across seeds and training budgets.

## Example (LinearLongitudinalF16‑v0)

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.gail.model import GAIL
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

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

# Unified API (wraps learn)
agent.train(num_episodes=250, max_steps=20, max_reward=-1)
```

## Unified training interface

GAIL exposes the shared unified `train()` API from `BaseRLModel`.
Internally it delegates to the legacy `learn()` method, translating
`num_episodes * max_steps` into a `max_frames` budget. GAIL-specific
options accepted via `**kwargs`:

- `max_frames` (`int`): override the computed step budget.
- `max_reward` (`float`): early-stop threshold for the mean test
  reward.

!!! tip
    High-quality `expert_data` is crucial—include demonstrations with varied initial states and maneuvers.

!!! warning "Gymnasium 5-tuple API"
    This implementation uses the modern Gymnasium 5-tuple step API internally:
    ```python
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ```
    If you are migrating from older code that used the 4-tuple API (`next_state, reward, done, info = env.step(action)`), ensure your environment is compatible with Gymnasium and returns the 5-tuple.

## API reference

::: tensoraerospace.agent.gail.model.GAIL

## References

- [Generative Adversarial Imitation Learning](https://arxiv.org/pdf/1606.03476)

## Tested on

- Unity environment
- LinearLongitudinalF16‑v0 (repository example)

## Rollout boundaries

Training snapshots observations so that environments can reuse numpy buffers.
Time limits bootstrap from the final observation; true terminations do not.
The training step budget is exact. Every 1,000 training steps, the current
rollout is finalized before evaluation uses the same environment. Training
then resets and starts a fresh episode; a partial episode is logged as truncated.
The imitation reward remains finite when the discriminator outputs zero.

Each rollout updates the policy for the configured `epochs`. Minibatches are
shuffled without replacement and include the final partial batch. PPO uses the
joint action likelihood ratio, normalized advantages and gradient clipping.
The environment and discriminator receive clipped physical commands; PPO
retains the sampled action for its Gaussian log probability.

Timeout bootstrap accepts `final_observation` and `terminal_observation`.
A missing or `None` final observation falls back to the returned observation.
