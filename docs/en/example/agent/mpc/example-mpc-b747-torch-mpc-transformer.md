# MPC + Transformer Dynamics for B747 — Step Response Tracking

This example demonstrates a complete **Model Predictive Control (MPC)** pipeline for the Boeing 747 longitudinal dynamics model using a learned **Transformer-based dynamics model**.

## Problem Statement

We control the **pitch angle (θ)** of a Boeing 747 aircraft to track a step reference signal using a Transformer encoder architecture as the dynamics model.

### What is Transformer Dynamics?

The **TransformerDynamicsModel** applies the Transformer encoder architecture to dynamics learning:

$$\hat{x}_{t+1} = \text{MLP}(\text{TransformerEncoder}(\text{Embed}([x_t, u_t])))$$

Key components:

- **Input embedding**: Projects `concat([x_t, u_t])` to `d_model` dimensions
- **Positional encoding**: Optional (disabled for `seq_len=1`)
- **Self-attention layers**: Capture relationships within the input
- **Feed-forward layers**: Transform attention output
- **Output projection**: Maps back to state dimension

!!! info "Simplified Configuration"
    In this example we use `seq_len=1` (no explicit history), so the Transformer acts as a sophisticated nonlinear function approximator. For sequential data, increase `seq_len` and provide past state-action pairs.

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

1. **Environment Setup**: Create `LinearLongitudinalB747-v0` with step reference
2. **Data Collection**: Collect state transitions using diverse exploration signals
3. **Transformer Training**: Train `TransformerDynamicsModel` to predict state deltas
4. **MPC Control**: Use gradient-based optimization with Transformer predictions
5. **Evaluation**: Assess control quality via `ControlBenchmark`

## Configuration

### Simulation Parameters

```python
DT = 0.1            # Time step [s]
TN = 20.0           # Simulation duration [s]
N_STEPS = 201       # Total number of steps

REF_STEP_DEG = 5.0      # Target pitch [deg]
REF_STEP_TIME_S = 5.0   # Step occurs at t=5s
```

### Transformer Architecture

```python
D_MODEL = 64          # Embedding dimension
N_HEAD = 4            # Number of attention heads
N_LAYERS = 2          # Number of encoder layers
FF_DIM = 256          # Feed-forward hidden dimension
DROPOUT = 0.1         # Dropout rate (regularization)
```

!!! note "Architecture Choices"
    - **d_model=64**: Small embedding for fast inference. Increase for complex systems.
    - **nhead=4**: Allows 4 parallel attention patterns.
    - **num_layers=2**: Minimal depth; deeper models can capture more complex dynamics.
    - **dim_feedforward=256**: 4× `d_model` is a common choice.

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

from tensoraerospace.signals.standart import unit_step
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
    MPCTrackingExtraCostConfig,
    MPCWeights,
    TransformerDynamicsModel,
)
from tensoraerospace.benchmark import ControlBenchmark
```

## Transformer Model Architecture

```
Input: [x_t, u_t] ∈ ℝ^5
    ↓
Input Embedding: Linear(5 → 64)
    ↓
Positional Encoding (optional)
    ↓
┌─────────────────────────────────────────┐
│ TransformerEncoderLayer × 2:            │
│   • Multi-Head Self-Attention (4 heads) │
│   • Add & Norm                          │
│   • Feed-Forward (64 → 256 → 64)        │
│   • Add & Norm                          │
│   • Dropout (0.1)                       │
└─────────────────────────────────────────┘
    ↓
Output Projection: Linear(64 → 4)
    ↓
Output: Δx ∈ ℝ^4
```

```python
transformer_model = TransformerDynamicsModel(
    input_dim=4 + 1,              # state_dim + action_dim
    output_dim=4,                 # state_dim
    d_model=D_MODEL,              # 64
    nhead=N_HEAD,                 # 4
    num_encoder_layers=N_LAYERS,  # 2
    dim_feedforward=FF_DIM,       # 256
    dropout=DROPOUT,              # 0.1
    seq_len=1,                    # Single time step
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

### Agent with Custom Transformer Model

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
    # Custom Transformer model
    model=transformer_model,
    model_predict_delta=True,
    normalize=True,
    dynamics_lr=LR,
    # MPC settings
    iters=MPC_ITERS,
    mpc_lr=MPC_LR,
    warm_start=True,
    mpc_track_best=True,
    mpc_compile_dynamics=True,  # JIT compile for speed (CUDA only)
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

## Training Transformer Dynamics

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
Train dynamics: 100%|██████████| 69600/69600 [18:18<00:00, 63.37step/s, loss=1.57e-5]
```

!!! warning "Training Time"
    Transformer models take significantly longer to train than MLPs due to the attention mechanism. Expect ~18 minutes on GPU for 1500 episodes (vs ~2.5 min for MLP).

## MPC Rollout

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
<!-- ![Step Response Transformer](../../../../../assets/mpc/transformer_step_response.png) -->

### Benchmark Metrics

| Metric | Value | Description |
| --- | --- | --- |
| **Overshoot** | ~-0.10% | Minimal undershoot |
| **Settling time** | ~1.5 s | **Fastest** among all models |
| **Rise time** | ~0.8 s | **Fastest** among all models |
| **Peak time** | ~2.5 s | Time to first peak |
| **Static error** | ~0.009 | Low steady-state error |
| **Oscillation count** | 6 | More oscillations than MLP |

### Analysis

The Transformer model achieves **the best overall performance**:

1. **Fastest rise time (0.8s)**: The attention mechanism enables accurate short-horizon predictions, allowing aggressive control without instability.

2. **Fastest settling time (1.5s)**: Despite more oscillations, the Transformer converges quickly to the setpoint.

3. **Minimal overshoot (-0.10%)**: The slight undershoot is negligible and well within typical specifications.

4. **More oscillations (6)**: The Transformer's ability to capture fine details leads to small oscillations around the setpoint. This can be tuned by:
   - Increasing `w_osc` in step response config
   - Reducing MPC horizon
   - Adding more smoothing in weights

5. **Good static error (0.009)**: Better than NARX but slightly worse than MLP.

### Why Transformer Performs Well

1. **Self-attention**: Can model complex input-output relationships in a single forward pass
2. **Residual connections**: Improve gradient flow during training and inference
3. **Layer normalization**: Stabilizes activations, important for MPC gradient-based optimization
4. **Flexible capacity**: The 2-layer encoder provides enough capacity for B747 dynamics

### When to Use Transformer

Transformer dynamics is recommended when:

- **Complex nonlinear systems**: The attention mechanism can capture intricate dynamics
- **Longer sequences** (`seq_len > 1`): Natural fit for sequential data
- **Sufficient compute**: Training is slower but inference is parallelizable on GPU
- **Research/prototyping**: Modern architecture for exploring new control methods

## Comparison with Other Models

| Metric | MLP | NARX | **Transformer** |
| --- | --- | --- | --- |
| **Overshoot** | +0.30% | -1.87% | **-0.10%** |
| **Settling time** | 1.7 s | 3.0 s | **1.5 s** |
| **Rise time** | 1.1 s | 1.0 s | **0.8 s** |
| **Static error** | **0.001** | 0.026 | 0.009 |
| **Oscillations** | Low | Low | Medium |
| **Training time** | **~2.5 min** | ~9 min | ~18 min |

### Recommendations

- **For production**: Start with **MLP** (fastest training, good performance)
- **For systems with memory**: Use **NARX** with appropriate lags
- **For best tracking**: Use **Transformer** if training time is not critical
- **For research**: **Transformer** offers the most flexibility for architectural changes

## Source Code

Full notebook: [`example/mpc_controllers/example-mpc-b747-torch-mpc-transformer.ipynb`](https://github.com/TensorAeroSpace/TensorAeroSpace/blob/main/example/mpc_controllers/example-mpc-b747-torch-mpc-transformer.ipynb)
