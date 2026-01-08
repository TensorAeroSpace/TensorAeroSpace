# Model Predictive Control (MPC)

MPC uses a dynamics model to predict system behavior and choose an optimal control sequence under constraints. At each step it solves an optimization problem, applies the first control input from the optimal sequence, and repeats with a shifted horizon.

![MPC diagram](../agent/img/mpc/mpc.png){ width=800 }

## Theory (brief)

- Discrete dynamics: \(x_{k+1} = f(x_k, u_k)\)
- Horizon cost \(N\):

$$
J = \sum_{i=0}^{N-1} (x_{k+i} - x^{\mathrm{ref}}_{k+i})^\top Q (x_{k+i} - x^{\mathrm{ref}}_{k+i})
    + u_{k+i}^\top R\, u_{k+i} + \Delta u_{k+i}^\top S\, \Delta u_{k+i}
    + \text{terminal\_weight} \cdot (x_{k+N}-x^{\mathrm{ref}}_{k+N})^\top Q (x_{k+N}-x^{\mathrm{ref}}_{k+N})
$$

- Control increment:

$$
\Delta u_{k+i} = u_{k+i} - u_{k+i-1}
$$

- Constraints:

$$
\begin{aligned}
u_{\min} \le u_{k+i} \le u_{\max}, &\quad \Delta u_{\min} \le \Delta u_{k+i} \le \Delta u_{\max}, \\
\end{aligned}
$$

- Receding horizon: solve → apply \(u_k\) → shift window → repeat
- Stability: terminal weight, sufficient \(N\), feasibility

## Architecture

The MPC module consists of:

| Component | Class | Description |
| --- | --- | --- |
| **Low-level solver** | `MPC` | Projected-gradient optimizer over a differentiable dynamics |
| **High-level agent** | `MPCAgent` | DSAC-like wrapper with learned dynamics, buffer, training |
| **Weights config** | `MPCWeights` | Diagonal Q, R, S weights and terminal weight |
| **Constraints** | `MPCConstraints` | Box constraints for u and du |
| **Extra costs** | `MPCTrackingExtraCostConfig`, `MPCStepResponseExtraCostConfig` | Additional penalties for smoothness, overshoot, settling |
| **Dynamics models** | `OneStepMLP`, `NARXDynamicsModel`, `TransformerDynamicsModel` | Neural network models for learning dynamics |
| **Scaler** | `MPCStandardScaler` | Feature normalization (mean/std) |

## Quick Start

### Basic MPC with a custom dynamics function

```python
import numpy as np
import torch
from tensoraerospace.agent.mpc import MPC, MPCWeights, MPCConstraints

state_dim = 4
action_dim = 1

# Define dynamics: x_{t+1} = f(x_t, u_t)
def dynamics(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # Simple linear dynamics for example
    A = torch.eye(state_dim)
    B = torch.zeros(state_dim, action_dim)
    B[-1, 0] = 1.0  # control affects last state
    return x @ A.T + u @ B.T

# Configure weights
weights = MPCWeights(
    Q_diag=np.array([1.0, 1.0, 1.0, 10.0]),  # state tracking weights
    R_diag=np.array([0.01]),                   # control effort
    S_diag=np.array([0.1]),                    # control smoothness
    terminal_weight=2.0,
)

# Configure constraints
constraints = MPCConstraints(
    u_min=np.array([-1.0]),
    u_max=np.array([1.0]),
    du_min=np.array([-0.2]),
    du_max=np.array([0.2]),
)

# Create MPC solver
mpc = MPC(
    dynamics=dynamics,
    state_dim=state_dim,
    action_dim=action_dim,
    horizon=20,
    weights=weights,
    constraints=constraints,
    iters=60,
    lr=0.05,
    optimizer="adam",
    warm_start=True,
)

# Solve
x0 = np.zeros(state_dim)
x_ref = np.zeros((21, state_dim))  # horizon+1 reference trajectory
x_ref[:, -1] = 0.1  # target for last state component

result = mpc.solve(x0=x0, x_ref=x_ref, u_prev=None)
print("First control:", result.u0)
print("Predicted trajectory shape:", result.x_seq.shape)
```

### MPCAgent with learned dynamics (recommended)

`MPCAgent` provides a complete workflow: data collection, dynamics training, and MPC control.

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.mpc import (
    MPCAgent,
    MPCWeights,
    MPCConstraints,
    MPCStepResponseExtraCostConfig,
)

# Create environment
env = gym.make("LinearLongitudinalB747-v0", ...)

# Configure weights
weights = MPCWeights(
    Q_diag=np.array([1.0, 1.0, 10.0, 100.0]),
    R_diag=np.array([0.01]),
    S_diag=np.array([0.5]),
    terminal_weight=1.0,
)

# Configure constraints
constraints = MPCConstraints(
    u_min=np.array([-0.3]),
    u_max=np.array([0.3]),
    du_min=np.array([-0.05]),
    du_max=np.array([0.05]),
)

# Extra cost for step response quality
step_cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=-1,        # last state = theta
    rate_idx=-2,           # second-to-last = q (pitch rate)
    dt=0.01,
    overshoot_limit_deg=0.05,
    settle_band_deg=0.1,
    settle_time_target_s=1.0,
)

# Create agent
agent = MPCAgent(
    env,
    horizon=30,
    weights=weights,
    constraints=constraints,
    tracking_type="step_response",
    step_response_config=step_cfg,
    hidden_layers=(256, 256),
    normalize=True,
    device="cuda",  # or "cpu"
)

# Collect training data
agent.collect_data(num_episodes=50, exploration="signals")

# Train dynamics model
agent.train_dynamics(epochs=10, batch_size=1024)

# Use in control loop
obs, info = env.reset()
state = ...  # extract internal state from env
x_ref = ...  # reference trajectory (horizon+1, state_dim)

action = agent.select_action(state, x_ref=x_ref)
obs, reward, done, truncated, info = env.step(action)

# Save/load checkpoints
path = agent.save("./runs")
agent.load(path)
```

### Using custom dynamics models

You can plug in different neural network architectures:

=== "MLP (default)"

    ```python
    from tensoraerospace.agent.mpc import OneStepMLP

    model = OneStepMLP(
        input_dim=state_dim + action_dim,
        output_dim=state_dim,
        hidden_layers=(256, 256, 128),
        activation="relu",  # or "tanh", "gelu"
    )

    agent = MPCAgent(env, model=model, ...)
    ```

=== "NARX"

    ```python
    from tensoraerospace.agent.mpc import NARXDynamicsModel

    model = NARXDynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        num_layers=3,
        state_lags=1,
        control_lags=1,
    )

    agent = MPCAgent(env, model=model, ...)
    ```

=== "Transformer"

    ```python
    from tensoraerospace.agent.mpc import TransformerDynamicsModel

    model = TransformerDynamicsModel(
        input_dim=state_dim + action_dim,
        output_dim=state_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    agent = MPCAgent(env, model=model, ...)
    ```

## Extra Cost Functions

### Tracking mode (`tracking`)

Adds penalties for control smoothness:

- `w_du`: weight for \(\sum (\Delta u)^2\)
- `w_jerk`: weight for \(\sum (\Delta^2 u)^2\)

```python
from tensoraerospace.agent.mpc import MPCTrackingExtraCostConfig

cfg = MPCTrackingExtraCostConfig(w_du=50.0, w_jerk=10.0)
agent = MPCAgent(env, tracking_type="tracking", tracking_config=cfg, ...)
```

### Step response mode (`step_response`)

Adds penalties for overshoot, settling time, oscillations:

```python
from tensoraerospace.agent.mpc import MPCStepResponseExtraCostConfig

cfg = MPCStepResponseExtraCostConfig.from_degrees(
    tracked_idx=-1,               # index of tracked state (e.g., theta)
    rate_idx=-2,                  # index of rate state (e.g., q)
    dt=0.01,                      # timestep
    overshoot_limit_deg=0.05,     # max overshoot in degrees
    settle_band_deg=0.10,         # settling band width
    settle_time_target_s=1.0,     # target settling time
    w_overshoot=8000.0,           # overshoot penalty weight
    w_settle=8000.0,              # settling penalty weight
    w_osc=500.0,                  # oscillation penalty weight
    w_jerk=50.0,                  # jerk penalty weight
)

agent = MPCAgent(env, tracking_type="step_response", step_response_config=cfg, ...)
```

You can switch modes at runtime:

```python
agent.set_tracking_type("tracking", tracking_config=tracking_cfg)
# or
agent.set_tracking_type("step_response", step_response_config=step_cfg)
```

## Data Collection

`MPCAgent.collect_data()` supports two exploration strategies:

| Strategy | Description |
| --- | --- |
| `"random"` | Random actions from `env.action_space.sample()` |
| `"signals"` | Rich signal library: steps, ramps, sinusoids, chirps, doublets, etc. |

```python
agent.collect_data(
    num_episodes=50,
    max_steps=1000,
    exploration="signals",
    signal_kinds=["random_steps", "sinusoid", "chirp", "doublet"],
    action_amplitude_frac=0.8,
)
```

Available signal types: `random_steps`, `unit_step`, `multi_step`, `ramp`, `sinusoid`, `multisine`, `chirp`, `square_wave`, `triangular_wave`, `sawtooth`, `doublet`, `pulse`, `gaussian_pulse`, `exponential`, `damped_sinusoid`.

## Hyperparameters

### MPC Solver (`MPC`)

| Parameter | Description | Default |
| --- | --- | --- |
| `horizon` | Prediction horizon | 20 |
| `iters` | Optimization iterations per solve | 60 |
| `lr` | Learning rate | 0.05 |
| `optimizer` | `"adam"` or `"sgd"` | `"adam"` |
| `warm_start` | Reuse previous solution | `True` |
| `track_best` | Track best solution during optimization | `True` |
| `compile_dynamics` | Use `torch.compile` (PyTorch 2.x) | `False` |

### MPCAgent

| Parameter | Description | Default |
| --- | --- | --- |
| `hidden_layers` | MLP hidden layer sizes | `(256, 256)` |
| `normalize` | Normalize inputs/outputs | `True` |
| `dynamics_lr` | Learning rate for dynamics model | `1e-3` |
| `grad_clip_norm` | Gradient clipping | `1.0` |
| `memory_capacity` | Replay buffer size | `200_000` |
| `model_predict_delta` | Predict \(\Delta x\) instead of \(x'\) | `True` |

!!! tip "Best practices"
    - Use `exploration="signals"` for better coverage of state-action space
    - Start with `horizon=20-30` and increase if needed
    - Enable `normalize=True` for neural dynamics
    - Use `tracking_type="step_response"` for aerospace control tasks
    - For real-time control, consider `compile_dynamics=True` on GPU

## Examples

Complete end-to-end examples demonstrating MPC with different dynamics models on the B747 longitudinal control task:

| Example | Dynamics Model | Description |
| --- | --- | --- |
| [MPC + MLP](../example/agent/mpc/example-mpc-b747-torch-mpc-mlp.md) | `OneStepMLP` | Standard MLP-based dynamics learning with step response tracking |
| [MPC + NARX](../example/agent/mpc/example-mpc-b747-torch-mpc-narx.md) | `NARXDynamicsModel` | Nonlinear autoregressive model with exogenous inputs |
| [MPC + Transformer](../example/agent/mpc/example-mpc-b747-torch-mpc-transformer.md) | `TransformerDynamicsModel` | Transformer encoder for dynamics prediction |

Each example demonstrates the full pipeline:

1. **Environment setup** — Create B747 environment with step reference signal for pitch (θ)
2. **Data collection** — Collect transitions using rich exploration signals
3. **Dynamics training** — Train neural network to predict state transitions
4. **MPC rollout** — Run closed-loop control using learned dynamics
5. **Evaluation** — Analyze step response quality (overshoot, settling time, etc.)

### Key results from examples

| Model | Overshoot | Settling Time | Rise Time | Static Error |
| --- | --- | --- | --- | --- |
| MLP | ~0.30% | ~1.7s | ~1.1s | ~0.001 |
| NARX | ~-1.9% | ~3.0s | ~1.0s | ~0.026 |
| Transformer | ~-0.10% | ~1.5s | ~0.8s | ~0.009 |

!!! note "Running examples"
    Examples are Jupyter notebooks located in `example/mpc_controllers/`. Run them to see full training logs, plots, and benchmark reports.

## API Reference

::: tensoraerospace.agent.mpc.MPC

::: tensoraerospace.agent.mpc.MPCAgent

::: tensoraerospace.agent.mpc.MPCWeights

::: tensoraerospace.agent.mpc.MPCConstraints

::: tensoraerospace.agent.mpc.MPCSolveResult

::: tensoraerospace.agent.mpc.MPCTrackingExtraCostConfig

::: tensoraerospace.agent.mpc.MPCStepResponseExtraCostConfig

::: tensoraerospace.agent.mpc.MPCStandardScaler

::: tensoraerospace.agent.mpc.OneStepMLP

::: tensoraerospace.agent.mpc.NARXDynamicsModel

::: tensoraerospace.agent.mpc.NARX

::: tensoraerospace.agent.mpc.TransformerDynamicsModel
