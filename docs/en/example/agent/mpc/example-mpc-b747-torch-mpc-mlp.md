# MPC + MLP Dynamics for B747 — Step Response Tracking

This example demonstrates a complete **Model Predictive Control (MPC)** pipeline for the Boeing 747 longitudinal dynamics model using a learned **Multi-Layer Perceptron (MLP)** as the dynamics model.

## Problem Statement

We control the **pitch angle (θ)** of a Boeing 747 aircraft to track a step reference signal. The aircraft model is a 4th-order linear state-space representation of the longitudinal dynamics at cruise conditions.

### State Vector

The B747 longitudinal model has 4 state variables:

| Index | Variable | Description | Units |
| --- | --- | --- | --- |
| 0 | u | Forward velocity perturbation | m/s |
| 1 | w | Vertical velocity perturbation | m/s |
| 2 | q | Pitch rate | rad/s |
| 3 | θ | Pitch angle | rad |

### Control Input

| Index | Variable | Description | Units |
| --- | --- | --- | --- |
| 0 | δe | Elevator deflection | deg (env) / rad (internal) |

!!! warning "Unit Convention"
    The B747 environment expects **actions in degrees**, while the internal linear model and MPC work in **radians**. The `MPCAgent` handles this conversion via `action_to_env` and `action_from_env` adapters.

## Method Overview

1. **Environment Setup**: Create `LinearLongitudinalB747-v0` with a step reference for pitch (θ)
2. **Data Collection**: Collect state transitions \((x_t, u_t) \to x_{t+1}\) using diverse exploration signals
3. **Dynamics Learning**: Train `OneStepMLP` to predict \(\Delta x = x_{t+1} - x_t\)
4. **MPC Control**: Use gradient-based optimization to find optimal control sequence
5. **Evaluation**: Assess control quality via `ControlBenchmark`

## Configuration

### Simulation Parameters

```python
# Time discretization
DT = 0.1          # Time step [s]
TN = 20.0         # Simulation duration [s]
N_STEPS = 201     # Total number of steps

# Step reference signal
REF_STEP_DEG = 5.0      # Target pitch [deg]
REF_STEP_TIME_S = 5.0   # Step occurs at t=5s
```

### Data Collection

```python
COLLECT_EPISODES = 1500     # Number of exploration episodes
ACTION_RANGE_DEG = 25.0     # Maximum control amplitude [deg]
```

### Neural Network (OneStepMLP)

```python
HIDDEN = 256        # Hidden layer size
LR = 1e-4           # Learning rate
EPOCHS = 120        # Training epochs
BATCH_SIZE = 1024   # Mini-batch size
```

### MPC Parameters

```python
HORIZON = 20        # Prediction horizon [steps]
MPC_ITERS = 60      # Optimization iterations per step
MPC_LR = 0.02       # Optimizer learning rate (Adam)
```

## Imports

```python
import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from tensoraerospace.signals.standard import unit_step
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
)
from tensoraerospace.benchmark import ControlBenchmark
```

## Environment Setup

```python
# Time vector
T = np.arange(N_STEPS, dtype=np.float32) * DT

# Generate step reference signal (converted to radians internally)
reference_signal = unit_step(
    tp=T,
    degree=float(REF_STEP_DEG),
    time_step=float(REF_STEP_TIME_S),
    output_rad=True,
).reshape(1, -1)

# Create environment
env = gym.make(
    "LinearLongitudinalB747-v0",
    number_time_steps=N_STEPS,
    initial_state=np.array([[0.0], [0.0], [0.0], [0.0]], dtype=np.float32),
    reference_signal=reference_signal,
    dt=float(DT),
)
```

## MPCAgent Configuration

### State/Action Adapters

These functions handle the conversion between environment observations and internal MPC states:

```python
def obs_to_state(env, obs) -> np.ndarray:
    """Extract true state from environment (in radians)."""
    return np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)

def action_from_env(a_env_deg: np.ndarray) -> np.ndarray:
    """Convert action from env units (deg) to internal (rad)."""
    return np.deg2rad(a_env_deg.astype(np.float32)).astype(np.float32)

def action_to_env(u_rad: np.ndarray) -> np.ndarray:
    """Convert action from internal (rad) to env units (deg)."""
    return np.rad2deg(u_rad.astype(np.float32)).astype(np.float32)
```

### MPC Weights

The cost function is:
$$J = \sum_{k=0}^{H} \left[ (x_k - x_{ref})^T Q (x_k - x_{ref}) + u_k^T R\, u_k + \Delta u_k^T S\, \Delta u_k \right] + J_{extra}$$

```python
weights = MPCWeights(
    Q_diag=np.array([0.0, 0.0, 0.2, 2000.0], dtype=np.float32),  # [u, w, q, θ]
    R_diag=np.array([0.01], dtype=np.float32),                   # control effort
    S_diag=np.array([5.0], dtype=np.float32),                    # control rate
    terminal_weight=10.0,                                         # terminal cost multiplier
)
```

- **Q_diag**: State tracking weights. High weight on θ (2000.0) for precise pitch tracking.
- **R_diag**: Control magnitude penalty. Small (0.01) to allow aggressive control when needed.
- **S_diag**: Control rate penalty. Moderate (5.0) for smooth elevator commands.

### Constraints

```python
u_lim = float(np.deg2rad(25.0))    # ±25° elevator limit
du_max = float(np.deg2rad(10.0))   # ±10°/step rate limit

constraints = MPCConstraints(
    u_min=np.array([-u_lim], dtype=np.float32),
    u_max=np.array([u_lim], dtype=np.float32),
    du_min=np.array([-du_max], dtype=np.float32),
    du_max=np.array([du_max], dtype=np.float32),
)
```

### Step Response Extra Cost

The step response configuration adds penalties for:

- **Overshoot**: Penalize exceeding the setpoint
- **Settling time**: Encourage fast convergence
- **Oscillations**: Reduce hunting around setpoint
- **Jerk**: Smooth control transitions

```python
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=3,              # Index of θ in state vector
    rate_idx=2,                 # Index of q (pitch rate)
    dt=float(DT),
    overshoot_limit_deg=0.05,   # Allowable overshoot [deg]
    settle_band_deg=0.10,       # Settling band [deg]
    settle_time_target_s=1.0,   # Target settling time [s]
    w_overshoot=8000.0,         # Overshoot penalty weight
    w_settle=8000.0,            # Settling penalty weight
)
```

### Agent Initialization

```python
agent = MPCAgent(
    env,
    state_dim=4,
    action_dim=1,
    horizon=HORIZON,
    weights=weights,
    constraints=constraints,
    tracking_type="step_response",
    step_response_config=step_cfg,
    # MLP configuration
    hidden_layers=(HIDDEN, HIDDEN),  # Two hidden layers
    model_predict_delta=True,         # Predict Δx, not x_next
    normalize=True,                   # Normalize inputs/outputs
    dynamics_lr=LR,
    # MPC optimization
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    mpc_optimizer="adam",
    warm_start=True,                  # Initialize from previous solution
    # Adapters
    obs_to_state=obs_to_state,
    action_to_env=action_to_env,
    action_from_env=action_from_env,
    device="cuda",
)
```

## Data Collection

The agent collects transitions using diverse exploration signals to ensure good coverage of the state-action space:

```python
agent.collect_data(
    num_episodes=COLLECT_EPISODES,
    exploration="signals",
    signal_kinds=[
        "random_steps",      # Random step sequences
        "unit_step",         # Simple step inputs
        "multi_step",        # Multiple steps per episode
        "ramp",              # Linear ramps
        "sinusoid",          # Sinusoidal excitation
        "multisine",         # Sum of sinusoids
        "chirp",             # Frequency sweep
        "square_wave",       # Square wave
        "triangular_wave",   # Triangle wave
        "sawtooth",          # Sawtooth wave
        "doublet",           # Doublet maneuver (flight test)
        "pulse",             # Rectangular pulse
        "gaussian_pulse",    # Gaussian-shaped pulse
        "damped_sinusoid",   # Decaying oscillation
    ],
    action_amplitude_frac=1.0,
)
print(f"Collected {len(agent.memory)} transitions")
```

## Dynamics Training

```python
metrics = agent.train_dynamics(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    loss="mse",
)
print(f"Final training loss: {metrics['loss']:.2e}")
```

**Expected output:**
```
Train dynamics: 100%|██████████| 23520/23520 [02:31<00:00, 155.23step/s, loss=1.28e-6]
```

## MPC Rollout

```python
_ = env.reset()
agent.reset()

hist_theta_deg, hist_ref_deg, hist_u_deg = [], [], []
ref_theta_rad = np.asarray(env.unwrapped.reference_signal).reshape(-1)

for step in tqdm(range(env.unwrapped.number_time_steps - 2)):
    k = int(env.unwrapped.current_step)
    x0 = np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)
    
    # Build reference trajectory for horizon
    target = float(ref_theta_rad[min(k, len(ref_theta_rad)-1)])
    x_ref = np.zeros((HORIZON + 1, 4), dtype=np.float32)
    x_ref[:, 3] = target  # Only θ reference matters
    
    # Get optimal action from MPC
    action = agent.select_action(x0, x_ref=x_ref)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Log results
    theta_deg = float(np.rad2deg(env.unwrapped.model.xt[3]))
    hist_theta_deg.append(theta_deg)
    hist_ref_deg.append(float(np.rad2deg(target)))
    hist_u_deg.append(float(action[0]))
    
    if terminated or truncated:
        break
```

## Results

### Step Response Visualization

<!-- TODO: INSERT STEP RESPONSE PLOT HERE -->
<!-- ![Step Response MLP](../../../../../assets/mpc/mlp_step_response.png) -->

```python
t_plot = np.arange(len(hist_theta_deg)) * DT

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Pitch angle tracking
axes[0].plot(t_plot, hist_ref_deg, 'r--', linewidth=2, label="θ_ref")
axes[0].plot(t_plot, hist_theta_deg, 'b-', linewidth=1.5, label="θ")
axes[0].axhline(y=REF_STEP_DEG * 0.9, color='g', linestyle=':', alpha=0.5, label="90% rise")
axes[0].axhline(y=REF_STEP_DEG * 1.02, color='orange', linestyle=':', alpha=0.5, label="2% band")
axes[0].axhline(y=REF_STEP_DEG * 0.98, color='orange', linestyle=':', alpha=0.5)
axes[0].set_ylabel("Pitch angle θ [deg]")
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_title("MPC + MLP Dynamics: Step Response Tracking")

# Control signal
axes[1].plot(t_plot, hist_u_deg, 'g-', linewidth=1.5, label="δe")
axes[1].axhline(y=25, color='r', linestyle='--', alpha=0.5, label="limits")
axes[1].axhline(y=-25, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Elevator δe [deg]")
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Benchmark Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **Overshoot** | ~0.30% | Excellent — well within 2% typical requirement |
| **Settling time** | ~1.7 s | Fast convergence to ±2% band |
| **Rise time** | ~1.1 s | Time to reach 90% of setpoint |
| **Static error** | ~0.001 | Negligible steady-state error |

```python
bench = ControlBenchmark()
metrics = bench.becnchmarking_one_step(
    control_signal=np.array(hist_ref_deg),
    system_signal=np.array(hist_theta_deg),
    signal_val=0.0,
    dt=DT,
)
print(bench.generate_report(metrics))
```

### Analysis

The MLP-based MPC achieves excellent step response characteristics:

1. **Minimal overshoot** (~0.3%): The step response cost penalties effectively suppress overshoot.
2. **Fast settling** (~1.7s): The high weight on θ tracking (Q[3]=2000) drives quick convergence.
3. **Smooth control**: The rate penalty (S=5.0) prevents abrupt elevator movements.
4. **Near-zero static error**: The learned dynamics accurately capture the system behavior.

## Key Takeaways

- **OneStepMLP** is the simplest dynamics model, suitable for systems with smooth, approximately linear dynamics
- **Delta prediction** (`model_predict_delta=True`) improves training stability
- **Diverse exploration signals** are critical for good generalization
- **Step response extra cost** helps achieve tight overshoot/settling specifications

## Source Code

Full notebook: [`example/mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-mlp.ipynb)
