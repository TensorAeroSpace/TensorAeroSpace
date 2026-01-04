# Distributional Soft Actor-Critic (DSAC)

DSAC расширяет SAC за счёт распределительных (квантильных) критиков и регуляризации CAPS. Каждый критик — Implicit Quantile Network (IQN), который предсказывает набор квантилей вместо одного Q-значения, что улучшает устойчивость на шумных аэрокосмических динамиках.

## Чем отличается от SAC

- IQN-критики: сдвоенные квантильные головы (`QuantileTwin`) с косинусными эмбеддингами
- Квантильный Huber-loss: обучает полное распределение вознаграждений
- CAPS-регуляризация: пространственная и временная гладкость действий
- Те же реплей-буфер, автонастройка энтропии и soft-обновление таргета, что и в SAC

## Быстрый старт

```python
import numpy as np
import torch
from tensoraerospace.agent import DSAC
from tensoraerospace.envs.b747 import ImprovedB747Env

def step_reference(steps: int, deg: float = 5.0) -> np.ndarray:
    ref = np.zeros((1, steps), dtype=np.float32)
    ref[:, steps // 5 :] = np.deg2rad(deg)
    return ref

device = "cuda" if torch.cuda.is_available() else "cpu"
num_steps = 800

env = ImprovedB747Env(
    initial_state=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
    reference_signal=step_reference(num_steps, deg=5.0),
    number_time_steps=num_steps,
    dt=0.02,
    reward_mode="step_response",
)

agent = DSAC(
    env,
    batch_size=128,
    memory_capacity=200_000,
    learning_starts=1_000,
    updates_per_step=1,
    num_quantiles=32,
    embedding_dim=32,
    hidden_layers=[64, 64],
    huber_threshold=1.0,
    lr=3e-4,
    policy_lr=3e-4,
    device=device,
    log_every_updates=50,
    automatic_entropy_tuning=True,
)

agent.train(num_episodes=5, save_best=False)
agent.save("./runs")
agent.close()
```

!!! tip
    Держите `num_quantiles` в диапазоне 16–64 и `huber_threshold` около 1.0 для стабильного обучения. CAPS встроен; если действия становятся слишком сглаженными, уменьшите `updates_per_step`.

## Документация API

:::: tensoraerospace.agent.dsac.dsac.DSAC

:::: tensoraerospace.agent.dsac.model.QuantileTwin

:::: tensoraerospace.agent.dsac.model.IQNCritic

