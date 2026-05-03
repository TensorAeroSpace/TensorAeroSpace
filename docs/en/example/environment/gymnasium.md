# Gymnasium Environment Examples

Below are runnable snippets for key TensorAeroSpace Gymnasium environments. Every example follows the same template and uses the standard `reset()`/`step()` API.

## Common Setup

```python
import numpy as np
import gymnasium as gym

from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standard import unit_step

dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1]
)
```

## Environment Usage Template

```python
env = gym.make(
    'ENV_ID',
    number_time_steps=number_time_steps,
    initial_state=INITIAL_STATE,
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalF16-v0

```python
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    # Default state_space=["alpha", "q"], so initialize accordingly
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalB747-v0

```python
env = gym.make(
    'LinearLongitudinalB747-v0',
    number_time_steps=number_time_steps,
    # Full state: [u, w, q, theta]
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalF4C-v0

```python
env = gym.make(
    'LinearLongitudinalF4C-v0',
    number_time_steps=number_time_steps,
    # state_space=['theta','q','alpha','V']
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalUAV-v0

```python
env = gym.make(
    'LinearLongitudinalUAV-v0',
    number_time_steps=number_time_steps,
    # Full state: [u, w, q, theta]
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalX15-v0

```python
env = gym.make(
    'LinearLongitudinalX15-v0',
    number_time_steps=number_time_steps,
    # Recommended full initial state
    initial_state=[[0], [0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalELVRocket-v0

```python
env = gym.make(
    'LinearLongitudinalELVRocket-v0',
    number_time_steps=number_time_steps,
    # Model tracks two states; choose an appropriate initial state
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## LinearLongitudinalMissileModel-v0

```python
env = gym.make(
    'LinearLongitudinalMissileModel-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## GeoSat-v0

```python
env = gym.make(
    'GeoSat-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

## ComSat-v0

```python
env = gym.make(
    'ComSat-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0], [0]],
    reference_signal=reference_signals,
)

state, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

