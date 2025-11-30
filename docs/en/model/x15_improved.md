# ImprovedX15Env - Enhanced X-15 RL Environment

## Overview

`ImprovedX15Env` is an enhanced reinforcement learning environment for the North American X-15 experimental rocket plane's longitudinal channel control. It features normalized action/observation spaces, a comprehensive reward function, and realistic termination conditions.

## Key Features

- **Normalized spaces**: Both action and observation spaces are normalized to [-1, 1]
- **Extended observations**: [pitch_error, pitch_rate, pitch_angle, previous_action]
- **LQR-like reward**: Comprehensive cost function including:
  - Pitch angle tracking accuracy
  - Angular velocity damping
  - Control energy minimization
  - Control smoothness (jitter suppression)
- **Realistic termination**: Episode ends if flight envelope is exceeded
- **Pygame visualization**: Real-time 2D visualization with telemetry plots

## Observation Space

The observation is a 4-dimensional vector (normalized to [-1, 1]):

1. **pitch_error_norm**: Normalized pitch angle error (target - actual)
2. **pitch_rate_norm**: Normalized pitch angular velocity (q)
3. **pitch_angle_norm**: Normalized pitch angle (theta)
4. **prev_action_norm**: Previous control action (helps with smoothness)

## Action Space

Single continuous action (normalized to [-1, 1]):

- **elevator**: Normalized elevator deflection
  - -1.0 corresponds to -25° (nose down)
  - +1.0 corresponds to +25° (nose up)

## Physical Constraints

The environment enforces realistic X-15 flight envelope limits:

- **Max pitch angle**: ±30° (experimental aircraft with larger envelope)
- **Max pitch rate**: ±10°/s
- **Max elevator deflection**: ±25°

## Reward Function

The reward is based on a quadratic cost function (LQR-style):

```
cost = w_pitch * e_theta² + w_q * e_q² + w_action * u² + 
       w_smooth * Δu² + w_jerk * Δ²u²

reward = -cost * reward_scale
```

Where:
- `e_theta`: Normalized pitch error
- `e_q`: Normalized pitch rate error relative to reference derivative
- `u`: Control action
- `Δu`: Control change (first derivative)
- `Δ²u`: Control jerk (second derivative)

Default weights:
- `w_pitch = 5.0` (primary objective: track pitch)
- `w_q = 0.2` (dampen oscillations)
- `w_action = 0.003` (minimize energy)
- `w_smooth = 0.01` (smooth control)
- `w_jerk = 0.001` (reduce control jitter)
- `reward_scale = 0.1` (Q-value stability)

**Termination penalty**: -100 if pitch angle exceeds ±30°

## Usage Example

```python
import numpy as np
from tensoraerospace.envs import ImprovedX15Env
from tensoraerospace.signals import unit_step

# Setup
dt = 0.01
duration = 10.0
num_steps = int(duration / dt)

# Initial state [u, w, q, theta] in SI units
initial_state = np.array([600.0, 0.0, 0.0, 0.0])

# Reference signal (10° pitch step at t=1s)
time_array = np.arange(0, duration, dt)
reference_signal = unit_step(
    tp=time_array,
    degree=10.0,
    time_step=1.0,
    output_rad=True,
).reshape(1, -1)

# Create environment
env = ImprovedX15Env(
    initial_state=initial_state,
    reference_signal=reference_signal,
    number_time_steps=num_steps,
    dt=dt,
    initial_elevator_deg=0.0,
)

# Training loop
obs, info = env.reset()
for _ in range(num_steps):
    action = your_agent.predict(obs)  # Your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

model.learn(total_timesteps=100000)
model.save("ppo_x15")
```

## Comparison with LinearLongitudinalX15

| Feature | LinearLongitudinalX15 | ImprovedX15Env |
|---------|----------------------|----------------|
| Action space | [-60, 60]° | [-1, 1] normalized |
| Observation | Raw states | Normalized + error |
| Reward | Simple MSE | LQR-style multi-term |
| Termination | Fixed steps only | Envelope + steps |
| Visualization | Not implemented | Pygame with plots |
| Smoothness | No penalty | Explicit smoothness terms |

## Tuning Recommendations

For different control objectives, adjust reward weights:

**Aggressive tracking** (fast response, may oscillate):
```python
env.w_pitch = 10.0
env.w_q = 0.1
```

**Smooth control** (slower, less overshoot):
```python
env.w_pitch = 3.0
env.w_smooth = 0.05
env.w_jerk = 0.005
```

**Energy-efficient**:
```python
env.w_action = 0.01
env.w_smooth = 0.02
```

## Technical Details

**State vector** (internal, SI units):
- `u`: Longitudinal velocity [m/s]
- `w`: Normal velocity [m/s]
- `q`: Pitch rate [rad/s]
- `theta`: Pitch angle [rad]

**Model**: Linear time-invariant (LTI) system based on X-15 flight data, discretized using zero-order hold.

## References

1. Original X-15 flight dynamics
2. LongitudinalX15 model in `tensoraerospace.aerospacemodel`
3. ImprovedB747Env (similar design pattern)

## See Also

- [LinearLongitudinalX15](x15.md) - Basic environment
- [ImprovedB747Env](b747.md) - Similar improved environment for B747
- Example script: `example/environments/example-env-x15-improved.py`

