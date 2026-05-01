# MPC + NARX Dynamics for B747 — Step Response Tracking

This example demonstrates a complete **Model Predictive Control (MPC)** pipeline for the Boeing 747 longitudinal dynamics model using a learned **NARX (Nonlinear AutoRegressive with eXogenous inputs)** model.

## Problem Statement

We control the **pitch angle (θ)** of a Boeing 747 aircraft to track a step reference signal using a NARX neural network as the dynamics model.

### What is NARX?

**NARX** (Nonlinear AutoRegressive with eXogenous inputs) is a recurrent architecture that predicts future outputs based on past inputs and outputs:

$$x_{t+1} = f(x_t, x_{t-1}, ..., x_{t-n_x}, u_t, u_{t-1}, ..., u_{t-n_u})$$

Where:

- \(n_x\) = `state_lags` — number of past states to consider
- \(n_u\) = `control_lags` — number of past controls to consider

!!! info "One-Step vs Multi-Lag NARX"
    In this example we use NARX in **one-step mode** (`state_lags=1`, `control_lags=1`), making it equivalent to a standard MLP that receives `concat([x_t, u_t])`. For systems with significant memory effects, use higher lags and provide the augmented history vector to MPC.

### State Vector

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

## Method Overview

1. **Environment Setup**: Create `LinearLongitudinalB747-v0` with a step reference for pitch (θ)
2. **Data Collection**: Collect state transitions using diverse exploration signals
3. **NARX Training**: Train `NARXDynamicsModel` to predict state deltas
4. **MPC Control**: Use gradient-based optimization with NARX predictions
5. **Evaluation**: Assess control quality via `ControlBenchmark`

## Configuration

### Simulation Parameters

```python
DT = 0.1            # Time step [s]
TN = 20.0           # Simulation duration [s]
N_STEPS = 201       # Total number of steps

REF_STEP_DEG = 1.0      # Target pitch [deg] — smaller step for NARX demo
REF_STEP_TIME_S = 5.0   # Step occurs at t=5s
```

### Data Collection

```python
COLLECT_EPISODES = 1500     # Number of exploration episodes
ACTION_RANGE_DEG = 25.0     # Maximum control amplitude [deg]
```

### NARX Architecture

```python
NARX_HIDDEN = 256     # Hidden layer size
NARX_LAYERS = 3       # Number of hidden layers
STATE_LAGS = 1        # History of states (1 = current only)
CONTROL_LAGS = 1      # History of controls (1 = current only)
```

### Training

```python
EPOCHS = 120          # Training epochs
BATCH_SIZE = 512      # Mini-batch size
LR = 1e-4             # Learning rate
```

### MPC Parameters

```python
HORIZON = 20          # Prediction horizon [steps]
MPC_ITERS = 60        # Optimization iterations per step
MPC_LR = 0.02         # Optimizer learning rate
DU_MAX_DEG = 3.0      # Control rate limit [deg/step]
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
    NARXDynamicsModel,
)
from tensoraerospace.benchmark import ControlBenchmark
```

## NARX Model Architecture

The `NARXDynamicsModel` internally constructs:

```
Input: concat([x_{t}, ..., x_{t-n_x+1}, u_{t}, ..., u_{t-n_u+1}])
       ↓
   Linear(input_dim → hidden_size)
       ↓
   [LayerNorm → ReLU → Linear(hidden_size → hidden_size)] × (num_layers - 1)
       ↓
   Linear(hidden_size → output_dim)
       ↓
Output: Δx (state delta)
```

```python
narx_model = NARXDynamicsModel(
    state_dim=4,
    action_dim=1,
    hidden_size=NARX_HIDDEN,      # 256 neurons per layer
    num_layers=NARX_LAYERS,       # 3 hidden layers
    state_lags=STATE_LAGS,        # 1 (current state only)
    control_lags=CONTROL_LAGS,    # 1 (current control only)
)
```

## MPCAgent Configuration

### Weights and Constraints

```python
weights = MPCWeights(
    Q_diag=np.array([0.0, 0.0, 0.2, 2000.0], dtype=np.float32),
    R_diag=np.array([0.01], dtype=np.float32),
    S_diag=np.array([5.0], dtype=np.float32),
    terminal_weight=10.0,
)

u_lim = float(np.deg2rad(25.0))
du_max = float(np.deg2rad(DU_MAX_DEG))

constraints = MPCConstraints(
    u_min=np.array([-u_lim], dtype=np.float32),
    u_max=np.array([u_lim], dtype=np.float32),
    du_min=np.array([-du_max], dtype=np.float32),
    du_max=np.array([du_max], dtype=np.float32),
)
```

### Step Response Configuration

```python
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=3,
    rate_idx=2,
    dt=float(DT),
    overshoot_limit_deg=0.05,
    settle_band_deg=0.10,
    settle_time_target_s=1.0,
    w_overshoot=8000.0,
    w_settle=8000.0,
    w_sse_steady=40000.0,
    w_osc=500.0,
)
```

### Agent with Custom NARX Model

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
    # Custom NARX model
    model=narx_model,
    model_predict_delta=True,
    normalize=True,
    dynamics_lr=LR,
    # MPC settings
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    warm_start=True,
    mpc_track_best=True,
    # Adapters
    obs_to_state=obs_to_state,
    action_to_env=action_to_env,
    action_from_env=action_from_env,
    device="cuda",
)
```

## Data Collection

```python
agent.collect_data(
    num_episodes=COLLECT_EPISODES,
    exploration="signals",
    signal_kinds=[
        "random_steps",
        "unit_step",
        "multi_step",
        "ramp",
        "sinusoid",
        "multisine",
        "chirp",
        "square_wave",
        "triangular_wave",
        "sawtooth",
        "doublet",
        "pulse",
        "gaussian_pulse",
        "damped_sinusoid",
    ],
)
print(f"Collected {len(agent.memory)} transitions")
```

## Training NARX Dynamics

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
Train dynamics: 100%|██████████| 69600/69600 [08:54<00:00, 130.33step/s, loss=4.36e-6]
```

!!! note "Training Time"
    NARX with 3 hidden layers takes longer to train than a simple 2-layer MLP. Expect ~9 minutes on GPU for 1500 episodes.

## MPC Rollout

The rollout procedure is identical to the MLP example — `MPCAgent` handles the dynamics model internally:

```python
_ = env.reset()
agent.reset()

hist_theta_deg, hist_ref_deg, hist_u_deg = [], [], []
ref_theta_rad = np.asarray(env.unwrapped.reference_signal).reshape(-1)

for step in tqdm(range(env.unwrapped.number_time_steps - 2)):
    k = int(env.unwrapped.current_step)
    x0 = np.asarray(env.unwrapped.model.xt, dtype=np.float32).reshape(-1)
    
    target = float(ref_theta_rad[min(k, len(ref_theta_rad)-1)])
    x_ref = np.zeros((HORIZON + 1, 4), dtype=np.float32)
    x_ref[:, 3] = target
    
    action = agent.select_action(x0, x_ref=x_ref)
    obs, reward, terminated, truncated, info = env.step(action)
    
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
<!-- ![Step Response NARX](../../../../../assets/mpc/narx_step_response.png) -->

### Benchmark Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **Overshoot** | ~-1.87% | Slight undershoot (negative overshoot) |
| **Settling time** | ~3.0 s | Slower than MLP due to conservative control |
| **Rise time** | ~1.0 s | Comparable to MLP |
| **Peak time** | ~1.7 s | Time to first peak |
| **Static error** | ~0.026 | Small but noticeable steady-state offset |

### Analysis

The NARX model produces different control behavior compared to MLP:

1. **Undershoot instead of overshoot**: The system approaches the setpoint from below, indicating conservative control. This manifests as negative "overshoot" (~-1.87%).

2. **Slower settling**: At 3.0s, NARX takes almost twice as long as MLP (1.7s) to settle. This is due to:
   - Deeper network (3 layers vs 2) may have slightly different gradients
   - More conservative predictions at the start of step response

3. **Higher static error**: The 0.026 static error is larger than MLP's 0.001. This could be improved by:
   - Increasing the `w_sse_steady` weight
   - Using integral action (not implemented in base MPC)

4. **Good generalization**: Despite slower settling, NARX tracks the reference reliably and avoids instability.

### When to Use NARX

NARX is particularly useful when:

- **System has memory**: Higher lags (`state_lags > 1`) can capture delayed effects
- **Nonlinear dynamics**: The deeper architecture better approximates complex functions
- **Noisy observations**: LayerNorm in NARX helps with input normalization

For the B747 linear model, MLP performs better, but NARX demonstrates the modular architecture where any compatible model can be used.

## Comparison with MLP

| Metric | MLP | NARX |
| --- | --- | --- |
| Overshoot | +0.30% | -1.87% |
| Settling time | 1.7 s | 3.0 s |
| Rise time | 1.1 s | 1.0 s |
| Static error | 0.001 | 0.026 |
| Training time | ~2.5 min | ~9 min |

## Source Code

Full notebook: [`example/mpc_controllers/example-mpc-b747-torch-mpc-narx.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-narx.ipynb)
