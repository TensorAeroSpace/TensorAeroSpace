# Lesson 10 -- Training Your First RL Agent for Aerospace Control

## 1. Goal

In this lesson we go through a complete end-to-end workflow:

1. Set up a Boeing 747 pitch-tracking environment.
2. Establish a **PID baseline** with automatic tuning.
3. Train a **PPO** (Proximal Policy Optimization) agent.
4. Train a **SAC** (Soft Actor-Critic) agent.
5. Evaluate all three controllers and compare them with standard benchmark metrics.
6. Save, load, and visualize results.

By the end you will have a working RL training pipeline that you can adapt to any
TensorAeroSpace environment.

---

## 2. Prerequisites

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| PyTorch | >= 2.0 |
| TensorAeroSpace | latest |
| matplotlib | any |
| numpy | any |

Install everything with:

```bash
pip install tensoraerospace matplotlib
```

---

## 3. Step 1 -- Setting Up the Environment

We use `ImprovedB747Env` -- a normalized Boeing 747 longitudinal channel
environment with a composite reward function designed for RL training.

Key properties:

* **Observation space**: normalized vector `[pitch_error, pitch_rate, pitch, previous_action]`, all in `[-1, 1]`.
* **Action space**: scalar elevator deflection in `[-1, 1]`.
* **State vector**: `[u, w, q, theta]` -- forward velocity, vertical velocity, pitch rate, pitch angle.
* **Reward**: composite signal penalizing tracking error, pitch rate, control effort, jitter, and (in `step_response` mode) overshoot and oscillations.

```python
import numpy as np
import torch
import matplotlib.pyplot as plt

from tensoraerospace.envs import ImprovedB747Env
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp

# --- Time base ---
dt = 0.1                                         # 10 Hz sampling
tp = generate_time_period(tn=40, dt=dt)           # 40-second simulation
tps = convert_tp_to_sec_tp(tp, dt=dt)             # time axis in seconds
number_time_steps = len(tp)

# --- Reference signal: 1-degree step at t = 5 s ---
reference = np.reshape(
    unit_step(degree=1, tp=tp, time_step=50, output_rad=True),
    (1, -1),
)

# --- Initial state [u, w, q, theta] (all zeros) ---
initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# --- Create environment ---
env = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference,
    number_time_steps=number_time_steps,
    dt=dt,
)

obs, info = env.reset()
print(f"Observation shape : {np.array(obs).shape}")
print(f"Action space      : {env.action_space}")
print(f"Observation space : {env.observation_space}")
```

### What the parameters mean

| Parameter | Description |
|---|---|
| `degree=1` | Step amplitude in degrees (converted to radians by `output_rad=True`). |
| `time_step=50` | Index at which the step starts (50 * dt = 5 s). |
| `initial_state` | Aircraft starts at trim: zero perturbation in all channels. |
| `dt=0.1` | Discretization step; 0.1 s is a good balance between resolution and speed. |

---

## 4. Step 2 -- PID Baseline

Before training neural controllers it is essential to have a baseline.
`tensoraerospace.agent.pid.PID` provides a classic PID controller with
MATLAB-style automatic tuning.

```python
from tensoraerospace.agent.pid import PID

# Create PID controller
pid = PID(env=env, kp=1.0, ki=1.0, kd=0.5, dt=dt)

# Automatic tuning via state-space optimization
# track_state_idx=3 means we track theta (index 3 in [u, w, q, theta])
tune_result = pid.tune_matlab_style(
    track_state_idx=3,
    target_settling_time=5.0,   # aim for 5-second settling
    target_overshoot=5.0,       # aim for 5 % max overshoot
    n_iterations=100,
    verbose=True,
)
print(tune_result)
```

### Running a PID episode

```python
pid.reset()
obs_pid, info_pid = env.reset()
done = False
pid_states = []
pid_rewards = []
step_idx = 0

while not done:
    # Reference value at current time step (in radians)
    ref_val = float(reference[0, min(step_idx, reference.shape[1] - 1)])

    # The environment observation contains the normalized pitch at index 2;
    # for PID we need the raw theta from the model.
    raw_theta = float(env.unwrapped.model.get_output()[3])

    # Compute PID action (returns a float)
    action_pid = pid.select_action(setpoint=ref_val, measurement=raw_theta)

    # Normalize action to [-1, 1] for ImprovedB747Env
    max_ele_deg = env.unwrapped.max_stabilizer_angle_deg
    action_norm = np.clip(
        np.array([action_pid / np.deg2rad(max_ele_deg)], dtype=np.float32),
        -1.0, 1.0,
    )

    obs_pid, reward, terminated, truncated, info_pid = env.step(action_norm)
    done = terminated or truncated
    pid_states.append(raw_theta)
    pid_rewards.append(reward)
    step_idx += 1

pid_response = np.array(pid_states)
print(f"PID total reward: {sum(pid_rewards):.2f}")
```

---

## 5. Step 3 -- Training PPO

PPO is an on-policy algorithm that collects a batch of trajectories,
computes advantages with GAE, and updates the policy with a clipped
surrogate loss. It is a solid first choice for continuous control.

### 5.1 Create the agent

```python
from tensoraerospace.agent.ppo.model import PPO

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

ppo_agent = PPO(
    env=env,
    gamma=0.99,              # discount factor -- high value for long-horizon tracking
    max_episodes=50,         # total training episodes (increase to 200+ for better results)
    rollout_len=2048,        # steps collected before each policy update
    clip_pram=0.2,           # PPO clipping parameter epsilon
    num_epochs=10,           # SGD passes over each rollout
    batch_size=64,           # mini-batch size
    entropy_coef=0.01,       # entropy bonus -- keeps exploration alive
    actor_lr=3e-4,           # actor learning rate
    critic_lr=1e-3,          # critic learning rate
    gae_lambda=0.95,         # GAE lambda for variance reduction
    actor_hidden_dim=256,    # hidden layer size for actor network
    critic_hidden_dim=256,   # hidden layer size for critic network
    normalize_obs=True,      # running-mean normalization of observations
    seed=42,
    device=device,
)
```

### 5.2 Key hyperparameters explained

| Parameter | Role | Typical range |
|---|---|---|
| `gamma` | Discount factor. Higher values make the agent care more about future rewards. | 0.95 -- 0.999 |
| `clip_pram` | Limits the policy update magnitude per step. | 0.1 -- 0.3 |
| `rollout_len` | Number of environment steps before an update. Longer rollouts give more diverse data but slow training. | 512 -- 4096 |
| `num_epochs` | How many times the algorithm iterates over the collected rollout. | 3 -- 30 |
| `entropy_coef` | Bonus for policy entropy. Prevents premature convergence but too high leads to random behavior. | 0.001 -- 0.05 |
| `actor_lr` / `critic_lr` | Learning rates. Critic is often trained with a higher rate. | 1e-4 -- 3e-3 |
| `gae_lambda` | Bias-variance trade-off in advantage estimation. 1.0 = Monte Carlo, 0.0 = 1-step TD. | 0.9 -- 0.99 |

### 5.3 Train

```python
print("Starting PPO training ...")
ppo_agent.train()
print("PPO training complete.")
```

Training progress is logged to TensorBoard automatically. Launch the
dashboard with:

```bash
tensorboard --logdir runs
```

---

## 6. Step 4 -- Training SAC

SAC is an off-policy algorithm that maximizes a trade-off between
expected return and entropy. It uses a replay buffer, twin Q-networks,
and soft target updates. For many continuous-control tasks SAC is more
sample-efficient than PPO.

### 6.1 Create the agent

```python
from tensoraerospace.agent.sac import SAC

# Reset environment for SAC (same setup)
env_sac = ImprovedB747Env(
    initial_state=initial_state,
    reference_signal=reference,
    number_time_steps=number_time_steps,
    dt=dt,
)

sac_agent = SAC(
    env=env_sac,
    hidden_size=256,                   # width of critic and policy networks
    lr=3e-4,                           # critic learning rate
    policy_lr=3e-4,                    # policy learning rate
    gamma=0.99,                        # discount factor
    tau=0.005,                         # soft-update coefficient for target network
    alpha=0.2,                         # initial entropy coefficient
    automatic_entropy_tuning=True,     # let SAC adapt alpha automatically
    batch_size=64,                     # mini-batch size for replay sampling
    memory_capacity=100000,            # replay buffer capacity
    updates_per_step=1,                # gradient steps per environment step
    policy_type="Gaussian",            # stochastic Gaussian policy
    seed=42,
    device=device,
)
```

### 6.2 Key concepts

* **Replay buffer** (`memory_capacity`): Stores past transitions `(s, a, r, s', done)`.
  Off-policy learning reuses old data, improving sample efficiency.
* **Soft updates** (`tau`): The target Q-network is slowly blended toward the
  online network: `theta_target = tau * theta + (1 - tau) * theta_target`.
* **Automatic entropy tuning**: SAC adjusts the entropy coefficient `alpha`
  so the policy keeps a target entropy level. This removes one hyperparameter.
* **Gaussian policy**: The actor outputs mean and log-std of a squashed
  Gaussian distribution, guaranteeing actions stay within bounds.

### 6.3 Train

```python
print("Starting SAC training ...")
sac_agent.train(num_episodes=300)
print("SAC training complete.")
```

---

## 7. Step 5 -- Evaluation

After training we evaluate each agent deterministically (no exploration noise)
and collect trajectories for comparison.

### 7.1 Helper function

```python
def evaluate_agent_sac(agent, env, reference_signal, label="Agent"):
    """Run one evaluation episode with a SAC agent.

    Returns:
        response (np.ndarray): Recorded pitch angle at each time step.
        total_reward (float): Cumulative episode reward.
    """
    obs, _ = env.reset()
    done = False
    states = []
    total_reward = 0.0
    step_idx = 0

    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        raw_theta = float(env.unwrapped.model.get_output()[3])
        states.append(raw_theta)
        total_reward += float(reward)
        step_idx += 1

    print(f"{label} -- reward: {total_reward:.2f}, steps: {step_idx}")
    return np.array(states), total_reward


def evaluate_agent_ppo(agent, env, reference_signal, label="Agent"):
    """Run one evaluation episode with a PPO agent.

    Returns:
        response (np.ndarray): Recorded pitch angle at each time step.
        total_reward (float): Cumulative episode reward.
    """
    obs, _ = env.reset()
    done = False
    states = []
    total_reward = 0.0
    step_idx = 0

    while not done:
        # PPO uses act() which returns (action_tensor, mean_action, log_prob)
        _, mean_action, _ = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(mean_action[0])
        done = terminated or truncated
        raw_theta = float(env.unwrapped.model.get_output()[3])
        states.append(raw_theta)
        total_reward += float(reward)
        step_idx += 1

    print(f"{label} -- reward: {total_reward:.2f}, steps: {step_idx}")
    return np.array(states), total_reward
```

### 7.2 Run evaluation

```python
# Re-create a fresh environment for each evaluation run
def make_env():
    return ImprovedB747Env(
        initial_state=initial_state,
        reference_signal=reference,
        number_time_steps=number_time_steps,
        dt=dt,
    )

# PPO evaluation
env_eval_ppo = make_env()
ppo_response, ppo_reward = evaluate_agent_ppo(
    ppo_agent, env_eval_ppo, reference, label="PPO"
)

# SAC evaluation
env_eval_sac = make_env()
sac_response, sac_reward = evaluate_agent_sac(
    sac_agent, env_eval_sac, reference, label="SAC"
)
```

---

## 8. Step 6 -- Benchmarking and Comparison

TensorAeroSpace provides dedicated functions for control quality evaluation.

### 8.1 Compute metrics

```python
from tensoraerospace.benchmark.function import (
    overshoot,
    settling_time,
    static_error,
)

# Build reference array in degrees (same length as response arrays)
n_pid = len(pid_response)
n_ppo = len(ppo_response)
n_sac = len(sac_response)

ref_deg = np.rad2deg(reference[0])  # full reference in degrees

ref_pid = ref_deg[:n_pid]
ref_ppo = ref_deg[:n_ppo]
ref_sac = ref_deg[:n_sac]

pid_deg = np.rad2deg(pid_response)
ppo_deg = np.rad2deg(ppo_response)
sac_deg = np.rad2deg(sac_response)

# Overshoot (%)
os_pid = overshoot(ref_pid, pid_deg)
os_ppo = overshoot(ref_ppo, ppo_deg)
os_sac = overshoot(ref_sac, sac_deg)

# Settling time (index -> multiply by dt to get seconds)
st_pid = settling_time(ref_pid, pid_deg)
st_ppo = settling_time(ref_ppo, ppo_deg)
st_sac = settling_time(ref_sac, sac_deg)

# Static error (degrees)
se_pid = static_error(ref_pid, pid_deg)
se_ppo = static_error(ref_ppo, ppo_deg)
se_sac = static_error(ref_sac, sac_deg)
```

### 8.2 Results table

```python
def fmt_settling(st_index, dt):
    """Format settling time: index * dt or 'N/A'."""
    if st_index is None:
        return "N/A"
    return f"{st_index * dt:.2f} s"


print(f"{'Metric':<20} {'PID':>12} {'PPO':>12} {'SAC':>12}")
print("-" * 58)
print(f"{'Overshoot (%)':<20} {os_pid:>12.2f} {os_ppo:>12.2f} {os_sac:>12.2f}")
print(f"{'Settling time':<20} {fmt_settling(st_pid, dt):>12} "
      f"{fmt_settling(st_ppo, dt):>12} {fmt_settling(st_sac, dt):>12}")
print(f"{'Static error (deg)':<20} {se_pid:>12.4f} {se_ppo:>12.4f} {se_sac:>12.4f}")
print(f"{'Total reward':<20} {sum(pid_rewards):>12.2f} {ppo_reward:>12.2f} "
      f"{sac_reward:>12.2f}")
```

### 8.3 Using ControlBenchmark

For a more comprehensive report you can use the `ControlBenchmark` class:

```python
from tensoraerospace.benchmark import ControlBenchmark

bench = ControlBenchmark()

# signal_val=0 tells the function where the step starts (before the step,
# the reference is zero).
pid_metrics = bench.benchmarking_one_step(
    control_signal=ref_pid,
    system_signal=pid_deg,
    signal_val=0,
    dt=dt,
)
print("PID benchmark:", pid_metrics)
```

---

## 9. Step 7 -- Saving and Loading Models

### 9.1 Save

```python
# PPO
ppo_agent.save("./checkpoints/ppo_b747")

# SAC
sac_agent.save("./checkpoints/sac_b747")
```

### 9.2 Load from local checkpoint

```python
# SAC -- use from_pretrained with a local directory path
loaded_sac = SAC.from_pretrained("./checkpoints/sac_b747/<timestamp_folder>")

# PPO -- same pattern
loaded_ppo = PPO.from_pretrained("./checkpoints/ppo_b747/<timestamp_folder>")
```

### 9.3 Load from Hugging Face Hub

If a pretrained model is available on the Hub, loading is just:

```python
agent = SAC.from_pretrained("TensorAeroSpace/sac-b747")
```

---

## 10. Step 8 -- Visualization

### 10.1 Pitch tracking comparison

```python
time_axis = np.array(tps[:max(n_pid, n_ppo, n_sac)])

fig, ax = plt.subplots(figsize=(14, 5))

# Reference
ax.plot(tps[:reference.shape[1]], np.rad2deg(reference[0]),
        "k--", linewidth=2, label="Reference")

# PID
ax.plot(tps[:n_pid], pid_deg,
        linewidth=1.5, label=f"PID (OS={os_pid:.1f}%)")

# PPO
ax.plot(tps[:n_ppo], ppo_deg,
        linewidth=1.5, label=f"PPO (OS={os_ppo:.1f}%)")

# SAC
ax.plot(tps[:n_sac], sac_deg,
        linewidth=1.5, label=f"SAC (OS={os_sac:.1f}%)")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Pitch angle (deg)")
ax.set_title("Pitch Tracking: PID vs PPO vs SAC")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_pitch_tracking.png", dpi=150)
plt.show()
```

### 10.2 Reward curves

During training, reward histories are logged to TensorBoard. You can view
them interactively:

```bash
tensorboard --logdir runs
```

Alternatively, record rewards manually during training and plot with
matplotlib.

---

## 11. Hyperparameter Tips for Aerospace Tasks

### 11.1 Recommended starting points

| Hyperparameter | PPO | SAC |
|---|---|---|
| `gamma` | 0.99 | 0.99 |
| `learning_rate` (actor) | 3e-4 | 3e-4 |
| `learning_rate` (critic) | 1e-3 | 3e-4 |
| `hidden_size` | 256 | 256 |
| `batch_size` | 64 | 64 |
| `entropy_coef` / `alpha` | 0.01 | 0.2 (auto-tuned) |
| `episodes` | 50 -- 200 | 200 -- 500 |

### 11.2 Common pitfalls

1. **Learning rate too high** -- policy updates become unstable and the agent
   diverges (rewards collapse to large negative numbers). Start with 3e-4
   and reduce if needed.

2. **Not enough training episodes** -- aerospace environments have long
   horizons. With `rollout_len=2048` and 50 episodes the agent may not see
   enough variety. Scale up to 200+ episodes for reliable convergence.

3. **Sparse reward** -- if the reward only fires at the end of the episode
   the agent struggles to learn. `ImprovedB747Env` already provides a dense
   shaped reward, but if you build a custom environment make sure you give
   per-step feedback.

4. **Ignoring the PID baseline** -- if PID with automatic tuning already
   achieves good tracking, RL needs many more episodes to match it. Use PID
   metrics as a target.

5. **Forgetting to normalize** -- PPO benefits strongly from observation
   normalization (`normalize_obs=True`). SAC is less sensitive because the
   replay buffer contains diverse data, but normalized observations help
   network training in general.

6. **Terminal penalty too low** -- if the agent learns to terminate early to
   avoid accumulating negative rewards, increase `early_termination_penalty`
   and `early_termination_penalty_per_step` in the environment constructor.

---

## 12. What is Next

* **DSAC (Distributional SAC)**: Risk-aware control where the critic predicts
  a full return distribution. See `tensoraerospace.agent.dsac` and the example
  `example/reinforcement_learning/deep_rl/train_dsac_b747_step_response.py`.
* **MPC (Model Predictive Control)**: Use a learned or analytical dynamics model
  with receding-horizon optimization. See `tensoraerospace.agent.mpc`.
* **ADP / ADHDP**: Adaptive Dynamic Programming methods that combine neural
  networks with the Bellman principle. See `tensoraerospace.agent.adp`.
* **Comparison studies**: The `example/comparison/` directory contains notebooks
  that benchmark multiple agents head-to-head on the same environment.
* **Custom environments**: Adapt this workflow to other aircraft models --
  `ImprovedX15Env`, `F4CPitchEnvNormalized`, `ImprovedComSatEnv` -- by
  swapping the environment class and adjusting the initial state and reference
  signal.

---

## 13. Full Script

Below is the complete script assembled from the steps above. Copy it into a
`.py` file and run it.

```python
"""Lesson 10 -- Training Your First RL Agent for Aerospace Control.

End-to-end pipeline: PID baseline -> PPO training -> SAC training ->
evaluation -> benchmark comparison -> visualization.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from tensoraerospace.envs import ImprovedB747Env
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.agent.pid import PID
from tensoraerospace.agent.ppo.model import PPO
from tensoraerospace.agent.sac import SAC
from tensoraerospace.benchmark.function import overshoot, settling_time, static_error
from tensoraerospace.benchmark import ControlBenchmark

# =====================================================================
# 1. Environment Setup
# =====================================================================
dt = 0.1
tp = generate_time_period(tn=40, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)

reference = np.reshape(
    unit_step(degree=1, tp=tp, time_step=50, output_rad=True),
    (1, -1),
)
initial_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_env():
    return ImprovedB747Env(
        initial_state=initial_state,
        reference_signal=reference,
        number_time_steps=number_time_steps,
        dt=dt,
    )


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {device}")

# =====================================================================
# 2. PID Baseline
# =====================================================================
env_pid = make_env()
pid = PID(env=env_pid, dt=dt)
pid.tune_matlab_style(
    track_state_idx=3,
    target_settling_time=5.0,
    target_overshoot=5.0,
    n_iterations=100,
    verbose=True,
)

pid.reset()
obs_pid, _ = env_pid.reset()
done = False
pid_states, pid_rewards = [], []
step_idx = 0

while not done:
    ref_val = float(reference[0, min(step_idx, reference.shape[1] - 1)])
    raw_theta = float(env_pid.unwrapped.model.get_output()[3])
    action_pid = pid.select_action(setpoint=ref_val, measurement=raw_theta)
    max_ele_deg = env_pid.unwrapped.max_stabilizer_angle_deg
    action_norm = np.clip(
        np.array([action_pid / np.deg2rad(max_ele_deg)], dtype=np.float32),
        -1.0, 1.0,
    )
    obs_pid, reward, terminated, truncated, _ = env_pid.step(action_norm)
    done = terminated or truncated
    pid_states.append(raw_theta)
    pid_rewards.append(reward)
    step_idx += 1

pid_response = np.array(pid_states)
print(f"PID total reward: {sum(pid_rewards):.2f}")

# =====================================================================
# 3. PPO Training
# =====================================================================
env_ppo = make_env()
ppo_agent = PPO(
    env=env_ppo,
    gamma=0.99,
    max_episodes=50,
    rollout_len=2048,
    clip_pram=0.2,
    num_epochs=10,
    batch_size=64,
    entropy_coef=0.01,
    actor_lr=3e-4,
    critic_lr=1e-3,
    gae_lambda=0.95,
    actor_hidden_dim=256,
    critic_hidden_dim=256,
    normalize_obs=True,
    seed=42,
    device=device,
)

print("Starting PPO training ...")
ppo_agent.train()
print("PPO training complete.")

# =====================================================================
# 4. SAC Training
# =====================================================================
env_sac = make_env()
sac_agent = SAC(
    env=env_sac,
    hidden_size=256,
    lr=3e-4,
    policy_lr=3e-4,
    gamma=0.99,
    tau=0.005,
    alpha=0.2,
    automatic_entropy_tuning=True,
    batch_size=64,
    memory_capacity=100000,
    updates_per_step=1,
    policy_type="Gaussian",
    seed=42,
    device=device,
)

print("Starting SAC training ...")
sac_agent.train(num_episodes=300)
print("SAC training complete.")

# =====================================================================
# 5. Evaluation
# =====================================================================

def evaluate_ppo(agent, env):
    obs, _ = env.reset()
    done, states, total = False, [], 0.0
    while not done:
        _, mean_action, _ = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(mean_action[0])
        done = terminated or truncated
        states.append(float(env.unwrapped.model.get_output()[3]))
        total += float(reward)
    return np.array(states), total


def evaluate_sac(agent, env):
    obs, _ = env.reset()
    done, states, total = False, [], 0.0
    while not done:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(float(env.unwrapped.model.get_output()[3]))
        total += float(reward)
    return np.array(states), total


ppo_response, ppo_reward = evaluate_ppo(ppo_agent, make_env())
sac_response, sac_reward = evaluate_sac(sac_agent, make_env())

# =====================================================================
# 6. Benchmark
# =====================================================================
ref_deg = np.rad2deg(reference[0])

n_pid, n_ppo, n_sac = len(pid_response), len(ppo_response), len(sac_response)
pid_deg = np.rad2deg(pid_response)
ppo_deg = np.rad2deg(ppo_response)
sac_deg = np.rad2deg(sac_response)

os_pid = overshoot(ref_deg[:n_pid], pid_deg)
os_ppo = overshoot(ref_deg[:n_ppo], ppo_deg)
os_sac = overshoot(ref_deg[:n_sac], sac_deg)

st_pid = settling_time(ref_deg[:n_pid], pid_deg)
st_ppo = settling_time(ref_deg[:n_ppo], ppo_deg)
st_sac = settling_time(ref_deg[:n_sac], sac_deg)

se_pid = static_error(ref_deg[:n_pid], pid_deg)
se_ppo = static_error(ref_deg[:n_ppo], ppo_deg)
se_sac = static_error(ref_deg[:n_sac], sac_deg)


def fmt_st(idx):
    return f"{idx * dt:.2f} s" if idx is not None else "N/A"


print(f"\n{'Metric':<20} {'PID':>12} {'PPO':>12} {'SAC':>12}")
print("-" * 58)
print(f"{'Overshoot (%)':<20} {os_pid:>12.2f} {os_ppo:>12.2f} {os_sac:>12.2f}")
print(f"{'Settling time':<20} {fmt_st(st_pid):>12} {fmt_st(st_ppo):>12} "
      f"{fmt_st(st_sac):>12}")
print(f"{'Static error (deg)':<20} {se_pid:>12.4f} {se_ppo:>12.4f} "
      f"{se_sac:>12.4f}")
print(f"{'Total reward':<20} {sum(pid_rewards):>12.2f} {ppo_reward:>12.2f} "
      f"{sac_reward:>12.2f}")

# =====================================================================
# 7. Visualization
# =====================================================================
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(tps[:reference.shape[1]], np.rad2deg(reference[0]),
        "k--", linewidth=2, label="Reference")
ax.plot(tps[:n_pid], pid_deg, linewidth=1.5, label=f"PID (OS={os_pid:.1f}%)")
ax.plot(tps[:n_ppo], ppo_deg, linewidth=1.5, label=f"PPO (OS={os_ppo:.1f}%)")
ax.plot(tps[:n_sac], sac_deg, linewidth=1.5, label=f"SAC (OS={os_sac:.1f}%)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pitch angle (deg)")
ax.set_title("Pitch Tracking: PID vs PPO vs SAC on Boeing 747")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_pitch_tracking.png", dpi=150)
plt.show()

# =====================================================================
# 8. Save models
# =====================================================================
ppo_agent.save("./checkpoints/ppo_b747")
sac_agent.save("./checkpoints/sac_b747")
print("Models saved.")
```

---

## 14. References

1. Schulman, J. et al. *Proximal Policy Optimization Algorithms*, 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
2. Haarnoja, T. et al. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor*, 2018. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
3. TensorAeroSpace documentation and examples: [GitHub](https://github.com/TensorAeroSpace/TensorAeroSpace)
