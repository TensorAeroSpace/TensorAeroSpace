# Example: SAC on the linear F-16 — step α tracking

This example trains a **Soft Actor-Critic (SAC)** agent to track a 5° angle-of-attack step on the `LinearLongitudinalF16-v0` Gymnasium environment. SAC is an off-policy, stochastic actor-critic algorithm with a maximum-entropy objective — well suited to continuous-action control problems where exploration matters and gradient-free alternatives (like IHDP) struggle to cover the state space.

Source notebook: `example/reinforcement_learning/example-sac-f16.ipynb`.

## When SAC vs the adaptive-critic family

| Aspect | SAC | IHDP / DHP |
|---|---|---|
| Learning style | Off-policy, replay-buffer | Online, single-sample |
| Plant model | None | Online incremental linearisation |
| Exploration | Stochastic policy + entropy bonus | Persistent-excitation injection |
| Tuning effort | Low-to-medium | Low (given good seed / FF) |
| Sample efficiency | Low (10⁴–10⁶ env steps) | High (1–10 episodes) |
| Interpretability | Low | High (explicit \(F, G\)) |

Use SAC when you can afford long training and want a robust black-box controller. Use IHDP (see the [linear IHDP example](../ihdp/example_ihdp.md)) when you need fast online adaptation.

## 1. Imports

```python
import itertools

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from tensoraerospace.agent.sac import SAC
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.utils import convert_tp_to_sec_tp, generate_time_period
```

## 2. Time grid and reference signal

20-second episode at \(dt = 0.01\) s (2 000 steps). The reference is a 5° step at \(t = 10\) s — same task as the linear-IHDP example for head-to-head comparison.

```python
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1],
)
```

## 3. Build the environment

```python
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)
state, info = env.reset()
```

The env exposes the 2-D observation `[alpha, wz]` (radians, rad/s) and a 1-D action space for elevator command (radians). The built-in reward function penalises \(\alpha\) tracking error plus a small pitch-rate term; see `LinearLongitudinalF16.default_reward`.

## 4. Build the SAC agent

The SAC class owns its own replay buffer, critic pair, target networks, and stochastic Gaussian policy. Default hyperparameters follow the reference paper; we only override `hidden_size` and `device` for a small-footprint run.

```python
seed = 42
batch_size = 256
updates_per_step = 1
num_steps = 100_000   # total env steps across all training episodes

torch.manual_seed(seed)
np.random.seed(seed)

agent = SAC(
    env=env,
    hidden_size=32,
    batch_size=batch_size,
    updates_per_step=updates_per_step,
    memory_capacity=1_000_000,
    device="cpu",
    seed=seed,
)
```

Key hyperparameters (all defaulted unless overridden):

| Parameter | Default | Meaning |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Polyak averaging for target networks |
| `alpha` | 0.2 | Entropy temperature (entropy bonus weight) |
| `policy_type` | "Gaussian" | Squashed-Gaussian stochastic policy |
| `automatic_entropy_tuning` | False | If True, alpha is learned online |
| `lr` / `policy_lr` | 3e-4 | Critic / policy optimiser learning rates |

## 5. Training loop

Classic off-policy rollout. Each step: sample action → transition → push to replay → take `updates_per_step` gradient steps on the critic and policy once the replay buffer has enough samples.

```python
total_numsteps = 0
updates = 0

for _ in itertools.count(1):
    episode_reward = 0.0
    episode_steps = 0
    state, info = env.reset()
    state = np.array(state, dtype=np.float32).reshape(-1)
    reward_per_step = []

    for _ in tqdm(range(number_time_steps - 1)):
        action = agent.select_action(state)

        # Take gradient steps once we have enough experience.
        if len(agent.memory) > batch_size:
            for _ in range(updates_per_step):
                (c1_loss, c2_loss, policy_loss,
                 ent_loss, alpha_val) = agent.update_parameters(
                    agent.memory, batch_size, updates
                )
                updates += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += float(reward)
        reward_per_step.append(float(reward))

        # Mask 1 at truncated (bootstrap), 0 at terminated.
        mask = 1.0 if (episode_steps == number_time_steps - 1) else float(
            not (terminated or truncated)
        )
        agent.memory.push(state, action, float(reward), next_state, mask)
        state = next_state

        if terminated or truncated:
            break

    print(f"episode reward={episode_reward:+.3f}  avg/step={np.mean(reward_per_step):+.4f}")

    if total_numsteps > num_steps:
        break
```

!!! note "Training cost"
    100 000 env steps × 256-sample gradient updates take ~10–20 minutes on CPU. Reduce `num_steps` to 10 000 for a quick smoke test; the agent will not fully converge but you'll see the reward curve climb out of the noise floor within a few episodes. For serious runs use `device="cuda"` and raise `hidden_size` to 128 or 256.

## 6. Deterministic evaluation

After training, evaluate with `evaluate=True` so the policy outputs the mean of the squashed Gaussian instead of sampling.

```python
state, info = env.reset()
state = np.array(state, dtype=np.float32).reshape(-1)

alpha_log, action_log = [], []
total_rew = 0.0
for step in range(number_time_steps - 1):
    action = agent.select_action(state, evaluate=True)
    state, reward, terminated, truncated, info = env.step(action)
    state = np.array(state, dtype=np.float32).reshape(-1)
    alpha_log.append(float(state[0]))
    action_log.append(float(action[0]))
    total_rew += float(reward)
    if terminated or truncated:
        break

print(f"eval total reward = {total_rew:+.3f}")
```

## 7. Visualise tracking

```python
env.unwrapped.model.plot_transient_process(
    'alpha', tps, reference_signals[0], to_deg=True, figsize=(15, 4)
)
```

The env's built-in plotting helpers access the full state trajectory via `env.unwrapped.model` — this is the Gymnasium-wrapped form, so `unwrapped` is required to reach the underlying `ModelBase` instance.

## 8. Quantitative check

```python
alpha_hist = env.unwrapped.model.get_state('alpha', to_deg=True)
ref_deg = np.rad2deg(reference_signals[0, :len(alpha_hist)])

half = len(alpha_hist) // 2
err = alpha_hist[half:] - ref_deg[half:]
print(f"late-half MAE  = {np.mean(np.abs(err)):.4f}°")
print(f"late-half RMSE = {np.sqrt(np.mean(err ** 2)):.4f}°")
print(f"late-half max  = {np.max(np.abs(err)):.4f}°")
```

## Notes and gotchas

- **Small network.** `hidden_size=32` is aggressive — it converges faster than a 256-unit net on this low-dim task, but is more sensitive to `alpha` (entropy) tuning. Enable `automatic_entropy_tuning=True` if you see the policy collapse.
- **Replay buffer warm-up.** The first ~256 env steps make no gradient updates — `agent.memory` is being filled. Expect the early episodes to look like noise.
- **Reward scale.** `LinearLongitudinalF16.default_reward` returns values on the order of `-0.1` per step. If you see flat reward curves, sanity-check by switching to `use_reward=False` and supplying a custom reward function on env construction.
- **Action scaling.** SAC's Gaussian policy outputs actions already mapped to the env's `action_space.low`/`action_space.high`. No extra `Box` or clipping needed at the agent boundary.

## See also

- **SAC agent documentation:** [`docs/en/agent/sac.md`](../../../agent/sac.md) — theory and full API.
- **Alternative controller:** [`Example: IHDP on the linear F-16`](../ihdp/example_ihdp.md) — online adaptive approach, far fewer env steps.
- **Larger plant:** [`Example: SAC on B-747`](example-sac-b747.md) — same SAC recipe applied to the heavier B-747 longitudinal model.
