# Deep Deterministic Policy Gradient (DDPG)

DDPG — off‑policy актор‑критик для непрерывных действий: обучает детерминированную стратегию и Q‑функцию, используя буфер повторов и целевые сети со «мягким» обновлением.

## Компоненты

- Политика (Actor): `PolicyNetwork(s) -> a`, детерминированное действие через `tanh`
- Критик (Q‑сеть): `ValueNetwork(s,a) -> Q(s,a)`
- Целевые сети: `target_policy_net`, `target_value_net` для стабильности
- Реплей‑буфер: `ReplayBuffer` для выборки мини‑батчей
- Эксплорейшн: орнштейн–уленбековский шум `OUNoise`

## Теория (на базе реализации)

- Градиент политики (DPG):

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s\sim \mathcal{D}}\Big[\nabla_a Q(s,a)\big|_{a=\pi_\theta(s)}\, \nabla_\theta \pi_\theta(s)\Big]
$$

В коде минимизируется \(-Q(s,\pi(s))\), что эквивалентно градиентному подъёму по \(J\).

- Обновление критика (таргет Беллмана с целевыми сетями):

$$
\hat{Q}(s,a) = r + \gamma\,(1-\text{done})\, Q_{\text{target}}(s', \pi_{\text{target}}(s'))
$$

Лосс критика — MSE: \(\mathcal{L}_Q = (Q(s,a) - \hat{Q})^2\).

- Мягкое обновление целевых сетей:

$$
\theta^- \leftarrow (1-\tau)\,\theta^- + \tau\,\theta
$$

## Быстрый старт

```python
import gymnasium as gym
import numpy as np
from tensoraerospace.agent.ddpg.model import DDPG
from tensoraerospace.utils import generate_time_period
from tensoraerospace.signals.standart import unit_step

# Временная сетка и референс
dt = 0.01
tp = generate_time_period(tn=20, dt=dt)
number_time_steps = len(tp)
reference_signals = unit_step(degree=5, tp=tp, time_step=1000, output_rad=True).reshape(1, -1)

# Среда F‑16
env = gym.make('LinearLongitudinalF16-v0',
               number_time_steps=number_time_steps,
               initial_state=[[0],[0],[0]],
               reference_signal=reference_signals,
               use_reward=True,
               state_space=["theta","alpha","q"],
               output_space=["theta","alpha","q"],
               control_space=["ele"],
               tracking_states=["alpha"],)

agent = DDPG(env, value_lr=1e-3, policy_lr=1e-4, replay_buffer_size=1_000_000)
agent.learn(max_frames=12000, max_steps=500, batch_size=128)
```

!!! tip
    Эксплорейшн обеспечивается OU‑шумом: контролируйте `sigma` и `decay_period`, чтобы плавно снижать силу шума.

## Документация API

::: tensoraerospace.agent.ddpg.model.DDPG

## Источники

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

## Где тестировалось

- Unity‑среда
- LinearLongitudinalF16‑v0 (пример в репозитории)
