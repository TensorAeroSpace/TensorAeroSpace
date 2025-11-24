Пример использования SAC
===========================================================

```python
import gymnasium as gym
import itertools
import torch
import numpy as np
from gym.spaces import Box
from tqdm import tqdm

from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step
from tensoraerospace.agent.sac import SAC, ReplayMemory
```

```python
# Параметры времени
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
tps = convert_tp_to_sec_tp(tp, dt=dt)
number_time_steps = len(tp)
reference_signals = np.reshape(
    unit_step(degree=5, tp=tp, time_step=10, output_rad=True),
    [1, -1]
)
```

```python
# Среда Gymnasium
env = gym.make(
    'LinearLongitudinalF16-v0',
    number_time_steps=number_time_steps,
    initial_state=[[0], [0]],
    reference_signal=reference_signals,
)
state, info = env.reset()
```

```python
# Настройки обучения
seed = 42
replay_size = 1_000_000
batch_size = 256
updates_per_step = 1
num_steps = 100_000

action_space_boxes = Box(low=np.array([-30], dtype=np.float32), high=np.array([30], dtype=np.float32))

torch.manual_seed(seed)
np.random.seed(seed)
```

```python
# Память и агент
memory = ReplayMemory(replay_size, seed)
agent = SAC(env=env, hidden_size=32, device="cpu")
```

```python
# Цикл обучения
total_numsteps = 0
updates = 0

for _ in itertools.count(1):
    episode_reward = 0.0
    episode_steps = 0
    terminated = False
    truncated = False
    state, info = env.reset()
    state = np.array(state, dtype=np.float32).reshape(-1)
    reward_per_step = []

    for _ in tqdm(range(number_time_steps - 1)):
        action = agent.select_action(state)

        if len(memory) > batch_size:
            for _ in range(updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory, batch_size, updates
                )
                updates += 1

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).reshape(-1)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += float(reward)
        reward_per_step.append(float(reward))
        mask = 1.0 if (episode_steps == number_time_steps - 1) else float(not (terminated or truncated))
        memory.push(state, action, float(reward), next_state, mask)
        state = next_state

        if terminated or truncated:
            break

    print("avg reward/step:", np.mean(reward_per_step))
    if total_numsteps > num_steps:
        break
```

```python
# Визуализация переходного процесса по alpha
env.model.plot_transient_process('alpha', tps, reference_signals[0], to_deg=True, figsize=(15,4))
```

<!-- Изображение отсутствует в репозитории; закомментировано, чтобы избежать предупреждения сборки. -->
<!-- ![alpha](output_10_0.png) -->

