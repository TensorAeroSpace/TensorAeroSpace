# A2C with NARX‑Critic

A2C (Advantage Actor-Critic) uses an actor to select actions and a critic to evaluate states. Our implementation employs a NARX (Nonlinear AutoRegressive with eXogenous inputs) critic, enabling better modeling of dynamics and history by explicitly incorporating past states.

![A2C-NARX Diagram](../agent/img/a2c_narx.png){ width=800 }

## Components

- Actor: Gaussian policy \(\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)\); implemented in the `Actor` class (PyTorch)
- Critic (NARX): evaluates \(V(s)\) using an extended input (current state + past signals); see `Critic` (A2C) and `NARX` (modular NARX network)
- Experience collection: `Runner` gathers trajectories and clips actions to the `action_space`
- Training: `A2CLearner.learn` updates actor/critic with stabilization (gradient clipping, entropy bonus)

## Targets and history

The critic receives `z_t = [s_t, s_(t-1)]`; the next critic state is
`[s_(t+1), s_t]`. History persists between calls to `Runner.run` and resets
at episode boundaries. Collection snapshots observations.

With `discount_rewards=False`, the one-step target is:

$$
y_t = r_t + \gamma (1-\mathrm{terminated}_t) V_\phi(z_{t+1}),
\qquad A_t = y_t - V_\phi(z_t).
$$

With `discount_rewards=True`, returns accumulate until an episode or rollout
boundary. True termination has zero continuation; time limits and incomplete
rollouts bootstrap from the critic at the actual final observation. Rewards
from the next episode do not enter the return. Targets have no gradient.
The critic minimizes MSE; the actor minimizes
`-mean(log_prob * advantage) - entropy_beta * entropy`.
Log probabilities are summed across components of a multidimensional action.

`Runner` returns `NARXTransition` records that still unpack as
`(action, reward, state, next_state, done)` and carry `terminated` and
`previous_state` metadata. Legacy five-tuples remain accepted, but their
`done=True` is treated as true termination and their initial history is zero.
Use `Runner` or explicit `NARXTransition` records for time-limit handling.

## Quick start

```python
import gymnasium as gym
import torch
from tensoraerospace.agent.a2c.narx import Actor, Critic, A2CLearner, Runner

env = gym.make('LinearLongitudinalF16-v0', number_time_steps=2000)
actor = Actor(state_dim=env.observation_space.shape[0], n_actions=env.action_space.shape[0])
critic = Critic(state_dim=env.observation_space.shape[0])
learner = A2CLearner(actor, critic, gamma=0.99, entropy_beta=0.01)
runner = Runner(env, actor, learner.writer)

memory = runner.run(max_steps=2048)
learner.learn(memory, steps=2048, discount_rewards=True)
```

!!! tip
    For systems with strong inertia set `discount_rewards=False` so the critic trains on the TD target with \(V(s')\).

## Main A2C agent and action history

`agent.a2c.model.A2C` and `A2CWithNARXCritic` use `run_episode()` and `learn()`.
They share the terminal/time-limit target rules above. `RolloutTransition`
still unpacks into five values; its action is the original Gaussian sample.
The separate `executed_action` stores the bounded command passed to the plant.
Policy likelihoods use the sample, while the NARX critic uses executed commands.

For history length `h`, `A2CWithNARXCritic` receives
`[s_t, ..., s_(t-h+1), u_(t-1), ..., u_(t-h)]`.
The next input advances both histories using the current observation and
executed command. History survives rollout boundaries, resets between episodes,
and is copied into each transition. Legacy tuples lack this history metadata;
the first row then uses zero history. A single-transition update retains its
advantage instead of centering it to zero or producing an undefined variance.

## API reference

::: tensoraerospace.agent.a2c.narx.A2CLearner

::: tensoraerospace.agent.a2c.narx.Runner

<!-- ::: tensoraerospace.agent.narx.model.NARX -->
