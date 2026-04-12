# A2C with NARX‑Critic

A2C (Advantage Actor-Critic) uses an actor to select actions and a critic to evaluate states. Our implementation employs a NARX (Nonlinear AutoRegressive with eXogenous inputs) critic, enabling better modeling of dynamics and history by explicitly incorporating past states.

![A2C-NARX Diagram](../agent/img/a2c_narx.png){ width=800 }

## Components

- Actor: Gaussian policy \(\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2)\); implemented in the `Actor` class (PyTorch)
- Critic (NARX): evaluates \(V(s)\) using an extended input (current state + past signals); see `Critic` (A2C) and `NARX` (modular NARX network)
- Experience collection: `Runner` gathers trajectories and clips actions to the `action_space`
- Training: `A2CLearner.learn` updates actor/critic with stabilization (gradient clipping, entropy bonus)

## Theory (based on the implementation)

- Discounted returns:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}, \quad G^{\text{episodic}}_t = r_t + \gamma (1-\text{done}) G_{t+1}
$$

- Advantage in the code:

$$
A_t = \begin{cases}
    r_t, & \text{if } discount\_rewards=True \text{ (pure returns)}\\
    r_t + \gamma V(s_{t+1}) - V(s_t), & \text{otherwise (TD target)}
\end{cases}
$$

 and then \(A_t = \text{td\_target} - V(s_t)\).

- Losses:

$$
\mathcal{L}_\text{actor} = -\,\mathbb{E}[\log \pi_\theta(a_t|s_t)\, A_t] - \beta\,\mathbb{E}[\mathcal{H}[\pi_\theta(\cdot|s_t)]]
$$

$$
\mathcal{L}_\text{critic} = \mathbb{E}\big[(\text{td\_target} - V_\phi(s_t))^2\big]
$$

### NARX as critic

The NARX network (`tensoraerospace/agent/narx/model.py`) explicitly feeds previous outputs/states as inputs to predict the next step. In A2C the critic input is the concatenation of the current and previous states (`process_memory_narx` builds `critic_states`). This improves \(V(s)\) estimates for systems with significant dynamic memory.

Identification (simplified):

- Training NARX minimizes the MSE between predicted and target outputs over sequences (`NARX.train`).
- In A2C the critic minimizes the MSE between the target (returns or TD) and the current estimate \(V(s)\).

## Training loop

1. `Runner.run` collects tuples \((s_t, a_t, r_t, s_{t+1}, done_t)\), clipping actions to the `action_space`.
2. `process_memory_narx` prepares tensors: actions, rewards (optionally discounted), states, next states, termination flags, and `critic_states = [s_t, s_{t-1}]`.
3. `A2CLearner.learn`:
   - If `discount_rewards=True`, critic target `td_target = rewards` (returns); otherwise `r + γ V(s')`.
   - Advantage: `advantage = td_target - V(s)`.
   - Actor: log-probabilities from `Normal(mean, std)`, entropy; gradient-clipped updates.
   - Critic: MSE; gradient-clipped updates.
   - TensorBoard logging (losses, gradients/parameters, rewards).

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

## API reference

::: tensoraerospace.agent.a2c.narx.A2CLearner

::: tensoraerospace.agent.a2c.narx.Runner

<!-- ::: tensoraerospace.agent.narx.model.NARX -->
