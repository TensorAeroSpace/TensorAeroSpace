# ImprovedComSatEnv — Enhanced Satellite Control Environment

`ImprovedComSatEnv` is an improved version of the communication satellite control environment designed specifically for reinforcement learning applications. It features normalized action and observation spaces, LQR-style reward function, and multiple control objectives.

## Key Features

<div class="grid cards" markdown>

-   :material-chart-bell-curve: **Normalized Spaces**

    Both action and observation spaces are normalized to [-1, 1] for better RL training stability and convergence.

-   :material-target: **Multi-Objective Control**

    Tracks angular velocity while maintaining orbital stability, minimizing energy consumption and ensuring smooth control.

-   :material-function-variant: **LQR-Style Rewards**

    Quadratic cost function with tunable weights for different control objectives.

-   :material-shield-check: **Safety Constraints**

    Automatic termination on constraint violations to ensure realistic satellite operation.

</div>

## Environment Description

### State Space

The environment tracks the following state variables:

| Variable | Symbol | Description | Units |
|----------|--------|-------------|-------|
| Radial position | ρ | Distance from Earth center | km |
| Radial velocity | ρ̇ | Rate of change of radius | m/s |
| Angular velocity | θ̇ | Orbital angular velocity | rad/s |

### Observation Space

Normalized observation vector in range [-1, 1] with 4 components:

```python
[
    norm_theta_dot_error,  # Normalized angular velocity tracking error
    norm_rho_error,        # Normalized radial position deviation
    norm_rho_dot,          # Normalized radial velocity
    norm_prev_action       # Previous control action (for smoothness)
]
```

### Action Space

Single continuous action normalized to [-1, 1]:

- **u₂**: Tangential thrust
  - Range: [-1, 1] (normalized)
  - Physical range: [-25, 25] (arbitrary units)
  - u₂ > 0: Acceleration (increase angular velocity)
  - u₂ < 0: Deceleration (decrease angular velocity)

### Reward Function

The reward is computed as negative cost with the following components:

\[
r = -\text{scale} \cdot \left( 
w_{\theta} e_{\theta}^2 + 
w_{\rho} e_{\rho}^2 + 
w_{\dot{\rho}} e_{\dot{\rho}}^2 + 
w_u u^2 + 
w_{\Delta u} (\Delta u)^2 + 
w_{\Delta^2 u} (\Delta^2 u)^2
\right)
\]

Where:

| Weight | Symbol | Default | Description |
|--------|--------|---------|-------------|
| Angular velocity | w_θ̇ | 10.0 | Tracking accuracy (primary objective) |
| Radial position | w_ρ | 2.0 | Orbital stability |
| Radial velocity | w_ρ̇ | 0.5 | Velocity damping |
| Action cost | w_u | 0.01 | Energy efficiency |
| Smoothness | w_Δu | 0.05 | Control smoothness |
| Jerk | w_Δ²u | 0.01 | Jitter suppression |

### Termination Conditions

The episode terminates (`terminated=True`) with penalty reward of -100 if:

1. **Excessive angular velocity**: |θ̇| > 0.02 rad/s (2× max)
2. **Orbital instability**: |ρ - ρ_nominal| > 500 km
3. **Excessive radial velocity**: |ρ̇| > 200 m/s (2× max)

The episode truncates (`truncated=True`) when reaching the maximum number of time steps.

## Quick Start

### Basic Usage

```python
import numpy as np
from tensoraerospace.envs import ImprovedComSatEnv
from tensoraerospace.signals.standard import unit_step
from tensoraerospace.utils import generate_time_period

# Generate reference signal
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)

# Step change in angular velocity at t=10s
reference_signal = unit_step(
    degree=0.002, 
    tp=tp, 
    time_step=int(10.0/dt),
    output_rad=True
).reshape(1, -1) + 0.001  # Baseline 0.001 rad/s

# Initial state: [rho (km), rho_dot (m/s), theta_dot (rad/s)]
initial_state = np.array([6371.0, 0.0, 0.001])

# Create environment
env = ImprovedComSatEnv(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=number_time_steps,
    dt=dt,
)

# Reset and run
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Training with PPO

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Wrap environment
env = DummyVecEnv([lambda: ImprovedComSatEnv(...)])

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

# Train
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_comsat")
```

## Configuration Parameters

### Normalization Bounds

```python
env.max_angular_velocity = 0.01  # rad/s
env.max_radial_velocity = 100.0  # m/s
env.max_radial_position_deviation = 500.0  # km
env.max_thrust = 25.0
env.nominal_rho = 6371.0  # km (Earth radius)
```

### Reward Weights Tuning

Adjust weights for different control priorities:

```python
# Aggressive tracking
env.w_theta_dot = 20.0  # Prioritize tracking
env.w_smooth = 0.001    # Less smoothness penalty

# Energy-efficient control
env.w_action = 0.1      # Higher energy cost
env.w_theta_dot = 5.0   # Relax tracking

# Smooth control
env.w_smooth = 0.1      # High smoothness penalty
env.w_jerk = 0.05       # High jitter suppression
```

## Comparison with Base ComSatEnv

| Feature | ComSatEnv | ImprovedComSatEnv |
|---------|-----------|-------------------|
| Action space | [-60, 60] | [-1, 1] (normalized) |
| Observation space | Unnormalized states | [-1, 1] (normalized) |
| Reward function | Simple tracking error | LQR-style multi-objective |
| Smoothness penalty | ❌ | ✅ |
| Energy efficiency | ❌ | ✅ |
| Safety constraints | ❌ | ✅ |
| Initial action handling | ❌ | ✅ |

## Tips for Training

1. **Start with default weights**: The default reward weights provide a good balance for most tasks.

2. **Tune learning rate**: If training is unstable, reduce learning rate to 1e-4 or lower.

3. **Monitor terminations**: If episodes terminate too often, relax safety constraints or improve initial policy.

4. **Use curriculum learning**: Start with easier reference signals (smaller step changes) and gradually increase difficulty.

5. **Normalize reference signal**: Ensure reference signal stays within reasonable bounds for the satellite dynamics.

## API Reference

::: tensoraerospace.envs.comsat.ImprovedComSatEnv

## See Also

- [ComSat Base Model](../model/comsat.md): Mathematical model description
- PPO Training Example: `example/reinforcement_learning/example_ppo_comsat_improved.py`
- [ImprovedB747Env](../model/b747_imporove.md): Similar improved environment for aircraft

