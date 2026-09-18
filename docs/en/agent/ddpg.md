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

## Observation and checkpoint contract

DDPG stores independent copies of **raw observations** in replay. Both sides
of a sampled transition use the current running mean and variance. Statistics
are updated from snapshots collected before `env.step()`, so environments that
reuse an observation array cannot rewrite past states. When filling replay
manually, pass unnormalized observations.

A time limit permits bootstrapping; a true terminal state does not. For an
auto-reset environment, `final_observation` (or `terminal_observation`) supplies
the final state instead of the next episode's initial observation.

File checkpoints mark raw replay as `replay_observation_format="raw_v1"`.
Loading restores the checkpoint's normalization setting and statistics. Older
normalized replay cannot be converted reliably because each transition may have
used different statistics; it is skipped with a warning and training starts with
empty replay. Network weights remain loadable. Legacy replay collected with
normalization disabled can be restored. `load_replay=False` explicitly skips
replay loading.

These changes correct transition semantics; they do not guarantee policy
convergence or preserve the learning curve under unchanged hyperparameters.

When `min_sigma < max_sigma`, OU noise decays by the global step count within
`learn()`, without restarting the schedule at episode boundaries. The scale
for a step is applied before drawing its noise sample.

## Applying a trained policy

Use `agent.predict(observation)` with raw observations. It applies the saved
normalization and returns deterministic actions for one observation or a batch.
It does not update running statistics or add exploration noise. Direct
`policy_net.get_action()` calls expect network inputs and bypass normalization;
passing raw observations there can change the deployed policy.

## Validated B747 tracking experiment

For the linear B747 tracking scenario in `scripts/validate_ddpg.py`, reducing
critic LR to `1e-4` and decaying OU sigma from `0.3` to `0.05` over 15,000 steps
improved the mean final RMSE at 60,000 steps from 5.13° to 1.14° across seeds
11, 29 and 47. Actor LR remained `1e-4`; observation normalization was enabled.
There were no final pitch-bound violations. A PD reference achieved 0.954°;
this experiment does not establish superiority to a tuned classical controller
or convergence on other tasks. Shorter runs and intermediate evaluations remain
variable. Removing observation normalization alone did not resolve the issue.

```bash
.venv/bin/python scripts/validate_ddpg.py --repo . --env-source tensoraerospace/envs/b747_vec_torch.py --seed 11 --frames 60000 --value-lr 0.0001 --noise-min-sigma 0.05 --noise-decay-period 15000 --output /tmp/ddpg-b747-11.json
```

Repeat for seeds 29 and 47. The script records all final results and learning
curves, without selecting the best checkpoint.

## API reference

::: tensoraerospace.agent.ddpg.model.DDPG

## References

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

## Tested on

- Unity environment
- LinearLongitudinalF16‑v0 (repository example)
