# Incremental Model-based Global Dual Heuristic Programming (IMGDHP)

IMGDHP is an incremental model-based variant of Global Dual Heuristic Programming from the Adaptive Critic Designs (ACD) family. It is designed for online adaptive control of nonlinear systems under partial observability. The agent combines recursive least squares (RLS) system identification with a dual-head critic that estimates both the cost-to-go \(J\) and the costate vector \(\lambda = \partial J / \partial y\), enabling richer gradient information for the actor. See also the nonlinear F-16 model: [NonlinearLongitudinalF16](../model/f16_nonlinear_longitudinal.md).

## Key ideas

- **Incremental model**: online identification of local linearization \(\Delta y_{t+1} = A \Delta y_t + B \Delta u_t\) via RLS — lightweight, interpretable, and does not require a neural network for system ID
- **GDHP dual critic**: the critic outputs both \(J(o)\) (scalar cost-to-go) and \(\lambda(o)\) (costate vector), providing a richer gradient signal to the actor compared to standard HDP/DHP
- **Model-predictive actor update**: the actor gradient flows through the identified model matrices \(A\), \(B\), enabling one-step lookahead optimization
- **Partial observability**: the augmented observation \(o = [y; r; e]\) allows the agent to operate when the environment observation is not the full state

## Key differences from related agents

| Aspect | HDP | IHDP | **IMGDHP** |
| --- | --- | --- | --- |
| System ID | Fixed/known model | Online NN | Online RLS (incremental linear) |
| Critic output | \(J(o)\) only | \(J(o)\) only | \(J(o)\) + \(\lambda(o)\) (dual) |
| Actor update | Direct gradient | Model-based | Model-predictive via \(A\), \(B\) |
| Partial observability | No | Limited | Core design feature |
| Framework | NumPy | NumPy | PyTorch |

## IMGDHP components

| Component | Role | Implementation |
| --- | --- | --- |
| IncrementalModelRLS | Online identification of \(A\), \(B\) matrices via RLS | `tensoraerospace.agent.im_gdhp.IncrementalModelRLS` |
| GDHPActor | Deterministic policy network \(u = u_{\max} \tanh(\pi_\theta(o))\) | `tensoraerospace.agent.im_gdhp.GDHPActor` |
| GDHPCritic | Dual-head critic: shared backbone with \(J\)-head and \(\lambda\)-head | `tensoraerospace.agent.im_gdhp.GDHPCritic` |
| IMGDHPAgent | Orchestrates all components, training loop, predict/learn interface | `tensoraerospace.agent.im_gdhp.IMGDHPAgent` |

## Algorithm

At each time step \(t\), given observation \(y_t\) and reference \(r_t\):

1. **Augment observation**: \(o_t = [y_t;\; r_t;\; e_t]\), where \(e_t = y_t[\text{tracking}] - r_t\)
2. **Actor produces action**: \(u_t = \pi_\theta(o_t)\)
3. **Execute** \(u_t\) in the environment, observe \(y_{t+1}\)
4. **Compute one-step cost**: \(c_t = e_t^\top Q e_t + \rho \| u_t - u_{t-1} \|^2\)
5. **RLS update** (if \(t \geq 2\)): update incremental model using \((y_{t-2}, y_{t-1}, y_t, u_{t-2}, u_{t-1})\) to obtain \(A_t\), \(B_t\)
6. **Critic update** (GDHP dual loss):

\[
L = \underbrace{\left( J(o_t) - (c_t + \gamma J(o_{t+1})) \right)^2}_{L_J} + \beta \underbrace{\left\| \lambda(o_t) - \left( \frac{\partial c_t}{\partial y} + \gamma A_t^\top \lambda(o_{t+1}) \right) \right\|^2}_{L_\lambda}
\]

7. **Actor update** (model-predictive):

\[
\min_\theta \; c_t + \gamma \, J\!\left(\hat{o}_{t+1}\right), \quad \text{gradient flows through } B_t
\]

## Quick start

```python
import numpy as np
import gymnasium as gym
from tensoraerospace.agent.im_gdhp import IMGDHPAgent, IMGDHPConfig
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import sinusoid

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signal = sinusoid(
    degree=3, tp=tp, frequency=0.1, output_rad=True
).reshape(1, -1)

env = gym.make(
    "NonlinearLongitudinalF16-v0",
    number_time_steps=number_time_steps,
    initial_state=np.array([0.0, 0.0]),
    reference_signal=reference_signal,
    dt=dt,
)

config = IMGDHPConfig(
    gamma=0.95,
    actor_hidden=(32, 32),
    critic_hidden=(64, 64),
    actor_lr=1e-3,
    critic_lr=5e-3,
    track_Q=(1.0,),
    warmup_steps=5,
    forgetting=0.9995,
    u_max=25.0,
)

agent = IMGDHPAgent(
    n_obs=2,
    n_action=1,
    reference_size=1,
    tracking_indices=[0],
    config=config,
)

obs, info = env.reset()
for t in range(number_time_steps - 1):
    action = agent.predict(obs, reference_signal, t)
    obs_next, reward, terminated, truncated, info = env.step(action)
    metrics = agent.learn(obs_next, reference_signal, t)
    obs = obs_next
    if terminated or truncated:
        break
```

!!! tip
    `tracking_indices` must align with the observation indices that correspond to the tracked reference signal. For example, if the observation is `[alpha, wz]` and you track `alpha`, use `tracking_indices=[0]`.

## Hyperparameters

### General

| Parameter | Default | Description |
| --- | --- | --- |
| `gamma` | 0.95 | Discount factor |
| `warmup_steps` | 5 | Steps with frozen actor/critic (exploration only) |
| `critic_only_steps` | 0 | Additional steps with frozen actor after warmup |
| `seed` | None | RNG seed for reproducibility |
| `device` | `"cpu"` | PyTorch device |

### Actor

| Parameter | Default | Description |
| --- | --- | --- |
| `actor_hidden` | (32, 32) | Hidden layer sizes |
| `actor_lr` | 1e-3 | Learning rate |
| `u_max` | 25.0 | Per-channel control bound |
| `exploration_noise_std` | 0.0 | Gaussian exploration noise during training |

### Critic

| Parameter | Default | Description |
| --- | --- | --- |
| `critic_hidden` | (64, 64) | Backbone hidden layer sizes |
| `critic_lr` | 5e-3 | Learning rate |
| `beta_lambda` | 1.0 | Weight of \(\lambda\)-loss in GDHP dual objective |
| `critic_updates_per_step` | 1 | Gradient steps per environment transition |
| `target_update_tau` | 0.0 | Polyak coefficient for target critic (0 = no target network) |
| `critic_weight_decay` | 0.0 | L2 regularization |
| `max_grad_norm` | 5.0 | Gradient clipping threshold |

### Cost function

| Parameter | Default | Description |
| --- | --- | --- |
| `track_Q` | (1.0,) | Diagonal weights of tracking cost \(e^\top Q e\) |
| `action_rate_penalty` | 1e-3 | Coefficient \(\rho\) for \(\| \Delta u \|^2\) penalty |

### Incremental model (RLS)

| Parameter | Default | Description |
| --- | --- | --- |
| `forgetting` | 0.9995 | RLS forgetting factor \(\in (0, 1]\) |
| `cov_init` | 1e2 | Initial covariance matrix scale |

### Observation

| Parameter | Default | Description |
| --- | --- | --- |
| `obs_scale` | None | Per-component observation scaling factors |

## Supported environments

- `NonlinearLongitudinalF16-v0`
- `LinearLongitudinalF16-v0`

## API reference

::: tensoraerospace.agent.im_gdhp.model.IMGDHPAgent

::: tensoraerospace.agent.im_gdhp.model.IMGDHPConfig

::: tensoraerospace.agent.im_gdhp.incremental_model.IncrementalModelRLS

::: tensoraerospace.agent.im_gdhp.networks.GDHPActor

::: tensoraerospace.agent.im_gdhp.networks.GDHPCritic

## Sources

- Sun, Z. & van Kampen, E.-J. (2021). *Intelligent adaptive optimal control using incremental model-based global dual heuristic programming subject to partial observability*. Applied Soft Computing, 103, 107153.
- Zhou, Y., van Kampen, E.-J., & Chu, Q. P. (2020). *Incremental model based online dual heuristic programming for nonlinear adaptive control*. Control Engineering Practice, 95, 104242.
